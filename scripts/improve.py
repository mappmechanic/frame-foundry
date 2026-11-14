#!/usr/bin/env python3
"""Explicit controlled paired experiments; no paid dispatch without authorization."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import os
from pathlib import Path
import re
import time

from pipelines.ffmpeg.contracts import atomic_write_json, read_json, resolve_episode
from pipelines.ffmpeg.episode import episode_lock
from pipelines.ffmpeg.memory_loop import digest, finite, config_hash


@contextmanager
def environment(values):
    previous={k:os.environ.get(k) for k in values}
    os.environ.update({k:str(v) for k,v in values.items()})
    try:yield
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def reserve_experiment(budget_root, episode_paths, *, authorized_total, prior_spend):
    """Atomically forecast/reserve ALL arms against both approval and global cap.

    Existing episode limits are not increased. Unknown charges stay reserved.
    Reported prior spend cannot reduce an existing ledger's initial spend.
    """
    from pipelines.ffmpeg.budget import PROJECT_BUDGET_USD
    from pipelines.ffmpeg.llm_provider import episode_budget_usd, ProviderBudgetError
    root=Path(budget_root);root.mkdir(parents=True,exist_ok=True)
    try:
        total,spent=Decimal(str(authorized_total)),Decimal(str(prior_spend))
        allowance=episode_budget_usd()
        if any(not x.is_finite() or x<0 for x in (total,spent,allowance)) or total<=0:raise ValueError()
        cap=min(total,PROJECT_BUDGET_USD)
        with (root/'.ai-project.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=root/'.ai-project-spend.json'
            data=read_json(path) if path.exists() else dict(initial_spend_usd=str(spent),episodes={})
            initial=Decimal(str(data['initial_spend_usd']))
            amounts={k:Decimal(str(v)) for k,v in data['episodes'].items()}
            if any(not x.is_finite() or x<0 for x in [initial,*amounts.values()]):raise ValueError()
            initial=max(initial,spent)
            for eid in episode_paths:
                if eid not in amounts:amounts[eid]=allowance
                elif amounts[eid]!=allowance:raise ValueError('Reservation configuration changed')
            forecast=initial+sum(amounts.values(),Decimal(0))
            if forecast>cap:raise ProviderBudgetError('Experiment forecast exceeds authorized project headroom')
            atomic_write_json(path,dict(initial_spend_usd=str(initial),episodes={k:str(v) for k,v in amounts.items()}))
            return dict(authorized_total_usd=float(cap),prior_spend_usd=float(initial),episode_reserve_usd=float(allowance),project_forecast_usd=float(forecast))
    except (ValueError,KeyError,TypeError,InvalidOperation):
        raise ProviderBudgetError('Invalid experiment authorization or project ledger') from None


def snapshot_seed(seed_root, target):
    """Freeze all validated local seed lessons once, never query shared history later."""
    from pipelines.ffmpeg.memory_store import LocalJsonlStore, _bounded
    rows=[]
    if seed_root is not None:
        store=LocalJsonlStore(Path(seed_root)/'lessons.jsonl')
        def read():
            fd=store._directory()
            try:return store._read(fd)
            finally:os.close(fd)
        rows=_bounded(read)
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    # Store batch size is bounded; preserve full seed in bounded chunks.
    store=LocalJsonlStore(target/'lessons.jsonl')
    for start in range(0,len(rows),128):
        status=store.put_many(rows[start:start+128])
        if status['status']!='stored':raise ValueError('Seed snapshot unavailable')
    if not rows:(target/'lessons.jsonl').write_text('')
    return digest(rows)


def arm_row(iteration, eid, episode_dir, qa=None, error=False):
    root=Path(episode_dir);evaluation=None
    try:
        from pipelines.ffmpeg.evaluator import validate_eval_report
        evaluation=validate_eval_report(read_json(root/'eval-report.json'))
        if evaluation['episode_id']!=eid:raise ValueError()
        qa=read_json(root/'qa-report.json')
        evaluation['_experiment_config_hash']=config_hash(read_json(root/'narrative-spec.json'),evaluation)
    except Exception:evaluation=None
    valid=bool(evaluation and qa and qa.get('passed') is True and evaluation['gates']['passed'] is True and not error)
    cost=(evaluation or {}).get('model_cost')
    if cost is None:
        # Failed pre-evaluation calls still have real ledgers, not fabricated zero cost.
        from pipelines.ffmpeg.evaluator import model_cost
        cost,_=model_cost(root,None)
        if not root.exists():cost=dict(usd=None,known_usd=None,status='unavailable')
    scores=(evaluation or {}).get('scores') or {}
    # Do not promote failed/unavailable arms to measured quality results.
    row=dict(iteration=iteration,episode_id=eid,status='passed' if valid else 'failed' if error or (qa and qa.get('passed') is False) or (evaluation and not evaluation['gates']['passed']) else 'unavailable',
             J=scores.get('J') if valid else None,Q=scores.get('Q') if valid else None,
             model_usd=cost.get('usd'),known_model_usd=cost.get('known_usd'),cost_status=cost.get('status','unknown'),
             elapsed_seconds=(evaluation or {}).get('timing',{}).get('elapsed_seconds'),
             lessons_used=((evaluation or {}).get('memory') or {}).get('lessons_used',[]))
    return row,evaluation


def comparable_pair(a,b,ea,eb):
    """Rendered evidence, footage, Goal, rubric/model/config must agree."""
    if not ea or not eb or a['status']!='passed' or b['status']!='passed':return False
    if not ea.get('footage_hash') or ea['footage_hash']!=eb.get('footage_hash') or ea['goal']!=eb['goal']:return False
    return (ea.get('_experiment_config_hash') is not None and ea.get('_experiment_config_hash')==eb.get('_experiment_config_hash') and ea['scores']['method_version']==eb['scores']['method_version'] and ea['scores']['weights']==eb['scores']['weights']
            and ea['gates']['version']==eb['gates']['version'] and ea['gates']['thresholds']==eb['gates']['thresholds']
            and all(ea['judge'].get(k)==eb['judge'].get(k) for k in ('provider','model','rubric_version')))


def run_experiment(footage_dir, goal, *, experiment_id, iterations, episodes_root, evals_root,
                   seed_root=None, execute=False, paid=False, authorized_total=200, prior_spend=74,
                   recipe=None, runner=None, budget_root=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}',experiment_id):raise ValueError('Use a fresh lowercase experiment ID <=40 characters')
    if type(iterations) is not int or not 1<=iterations<=100:raise ValueError('Iterations must be 1–100')
    if paid and not execute:raise ValueError('Paid execution requires --execute')
    root=Path(evals_root).resolve();curves=root/'learning-curves'
    recipe=dict(recipe or {})
    allowed={'transcript_fixture','eval_labels','asr_provider','voiceover','target_seconds','aspect','overlays','sections','decider','planner_retry','keep_work'}
    if set(recipe)-allowed:raise ValueError('Unsupported controlled recipe setting')
    if not paid and (recipe.get('voiceover') or recipe.get('decider')=='decisions'):raise ValueError('Paid-only recipe requires explicit --paid')
    recipe.setdefault('asr_provider','local');recipe.setdefault('planner_retry',False)
    if not paid:recipe['asr_provider']='local'
    episode_root=Path(episodes_root).resolve()
    ids={arm:[f'{experiment_id}-{arm.replace("memory_", "")}-{i:03d}' for i in range(1,iterations+1)] for arm in ('memory_on','memory_off')}
    paths=[resolve_episode(episode_root,eid) for group in ids.values() for eid in group]
    if any(p.exists() for p in paths):raise ValueError('Episode IDs already exist; use a fresh experiment ID')
    curve=dict(schema_version='1.0',experiment_id=experiment_id,status='pending',created_at=datetime.now(timezone.utc).isoformat(),
               config_hash=digest(dict(goal=goal,recipe={k:str(v) if isinstance(v,Path) else v for k,v in recipe.items()},iterations=iterations)),
               arms={arm:[dict(iteration=i,episode_id=eid,status='unavailable',J=None,Q=None,model_usd=None,known_model_usd=None,cost_status='unavailable',elapsed_seconds=None,lessons_used=[]) for i,eid in enumerate(group,1)] for arm,group in ids.items()},
               deltas=[dict(iteration=i,comparable=False,J=None,Q=None,model_usd=None) for i in range(1,iterations+1)],
               limitations=['Recorded observations only; improvement and causation are not established.',
                            'Live HydraDB and Arizona quality proof are not established; no live-curve claim.',
                            'Both arms use isolated inference caches and fresh analysis/render/evaluation provenance; prior-judge repair is disabled.',
                            'Experiment memory is local-only to freeze the seed; ordinary Episodes may opt into HydraDB.'])
    with episode_lock(curves/experiment_id):
        path=curves/f'{experiment_id}.json'
        if path.exists():raise ValueError('Experiment already exists; use a fresh ID')
        scratch=curves/experiment_id
        seed_hash=snapshot_seed(seed_root,scratch/'seed')
        snapshot_seed(scratch/'seed',scratch/'memory_on')
        curve['config_hash']=digest([curve['config_hash'],seed_hash])
        def save():
            from jsonschema import Draft202012Validator, FormatChecker
            curve['limitations']=list(dict.fromkeys(curve['limitations']))[:100]
            schema=read_json(Path(__file__).resolve().parents[1]/'contracts/learning-curve.schema.json')
            Draft202012Validator(schema,format_checker=FormatChecker()).validate(curve)
            atomic_write_json(path,curve);atomic_write_json(curves/'latest.json',curve)
        save()
        if not execute:
            curve['limitations'].append('Evaluation pending: explicit --execute was not supplied; no Episode dispatched.')
            save();return curve
        budget_root=Path(budget_root or os.environ.get('FRAMEFOUNDRY_PROJECT_BUDGET_DIR') or Path(__file__).resolve().parents[1])
        try:
            if paid:
                curve['budget']=reserve_experiment(budget_root,[str(p) for p in paths],authorized_total=authorized_total,prior_spend=prior_spend)
                from pipelines.ffmpeg.llm_provider import LLMProvider
                LLMProvider.from_env()  # Validate configuration before any Episode starts.
        except Exception:
            curve['limitations'].append('Evaluation pending: authorization, headroom or provider configuration unavailable; no paid dispatch.')
            save();return curve
        if runner is None:
            from pipelines.ffmpeg.cli import run_episode
            runner=run_episode
        curve['status']='recorded';evaluations={}
        for i in range(iterations):
            for arm in ('memory_off','memory_on'):
                eid=ids[arm][i];start=time.monotonic();qa=None;error=False
                try:
                    with environment({'FRAMEFOUNDRY_CACHE_DIR':scratch/'cache'/arm,'FRAMEFOUNDRY_PROJECT_BUDGET_DIR':budget_root}):
                        qa=runner(footage_dir,goal,episodes_root=episode_root,episode_id=eid,evals_root=scratch/'evaluations'/arm,
                                  provider='env' if paid else None,memory=arm=='memory_on',memory_backend='local',
                                  memory_root=scratch/'memory_on' if arm=='memory_on' else None,reuse_history=False,**recipe)
                except Exception:error=True
                row,ev=arm_row(i+1,eid,episode_root/eid,qa,error)
                if row['elapsed_seconds'] is None:row['elapsed_seconds']=round(time.monotonic()-start,3)
                curve['arms'][arm][i]=row;evaluations[arm]=ev
                if ev:
                    from pipelines.ffmpeg.evaluator import append_scoreboard
                    saved={k:v for k,v in ev.items() if k!='_experiment_config_hash'}
                    try:append_scoreboard(saved,root)
                    except Exception:curve['limitations'].append('Shared scoreboard append unavailable; per-arm evaluation remains persisted.')
                if row['status']!='passed':curve['limitations'].append(f'{eid}: failed or unavailable evidence; no quality result fabricated.')
                save()
            a,b=(curve['arms'][arm][i] for arm in ('memory_on','memory_off'))
            comparable=comparable_pair(a,b,evaluations['memory_on'],evaluations['memory_off'])
            d=dict(iteration=i+1,comparable=comparable)
            for k in ('J','Q','model_usd'):d[k]=round(a[k]-b[k],6) if comparable and finite(a[k]) and finite(b[k]) else None
            curve['deltas'][i]=d;save()
        return curve


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--footage-dir',required=True,type=Path);p.add_argument('--goal',required=True)
    p.add_argument('--experiment',required=True);p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--episodes-root',type=Path,default=Path('episodes'));p.add_argument('--evals-root',type=Path,default=Path('evals'))
    p.add_argument('--seed-root',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--paid',action='store_true')
    p.add_argument('--authorized-total-usd',type=float,default=200);p.add_argument('--prior-spend-usd',type=float,default=74)
    p.add_argument('--transcript-fixture',type=Path);p.add_argument('--eval-labels',type=Path)
    p.add_argument('--asr-provider',choices=['local','auto','openai'],default='local');p.add_argument('--voiceover',action='store_true')
    p.add_argument('--duration-seconds',type=float);p.add_argument('--aspect',choices=['9:16','16:9'])
    p.add_argument('--overlays',action=argparse.BooleanOptionalAction,default=None);p.add_argument('--section',action='append',default=[])
    p.add_argument('--decider',choices=['llm','heuristic','decisions'],default='llm');p.add_argument('--planner-retry',action='store_true')
    a=p.parse_args(argv)
    recipe=dict(transcript_fixture=a.transcript_fixture,eval_labels=a.eval_labels,asr_provider=a.asr_provider,
                voiceover=a.voiceover,target_seconds=a.duration_seconds,aspect=a.aspect,overlays=a.overlays,sections=a.section,
                decider=a.decider,planner_retry=a.planner_retry)
    try:
        curve=run_experiment(a.footage_dir,a.goal,experiment_id=a.experiment,iterations=a.iterations,episodes_root=a.episodes_root,
            evals_root=a.evals_root,seed_root=a.seed_root,execute=a.execute,paid=a.paid,authorized_total=a.authorized_total_usd,prior_spend=a.prior_spend_usd,recipe=recipe)
    except ValueError as e:p.error(str(e))
    import json
    print(json.dumps(curve,indent=2,allow_nan=False))
    return 0 if curve['status']=='recorded' and all(r['status']=='passed' for rows in curve['arms'].values() for r in rows) else 1

if __name__=='__main__':raise SystemExit(main())
