"""Matched saved-plan editorial-policy A/B; no new Planner/ASR/TTS requests."""
import argparse
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipelines.ffmpeg.contracts import read_json,atomic_write_json,resolve_episode,StageError
from scripts.reoverlay import rerender

def row(root,qa):
    e=read_json(Path(root)/'eval-report.json');cost=e.get('model_cost') or {};scores=e['scores']
    return dict(episode_id=Path(root).name,status='passed' if qa.get('passed') and e['gates']['passed'] else 'failed',J=scores.get('J'),Q=scores.get('Q'),model_usd=cost.get('usd'),known_model_usd=cost.get('known_usd'),cost_status=cost.get('status','unknown'))
def compare(a,b,ae,be):
    judged=all(e.get('judge',{}).get('status')=='scored' for e in (ae,be))
    matched=bool(ae.get('footage_hash') and ae['footage_hash']==be.get('footage_hash') and ae.get('goal')==be.get('goal') and ae['scores'].get('method_version')==be['scores'].get('method_version') and ae.get('judge',{}).get('model')==be.get('judge',{}).get('model') and ae.get('judge',{}).get('provider')==be.get('judge',{}).get('provider'))
    known=all(r.get('cost_status') in ('measured','estimated','complete','zero') for r in (a,b)) and all(type(r.get(k)) in (int,float) and math.isfinite(r[k]) for r in (a,b) for k in ('J','Q','model_usd'))
    comparable=judged and matched and known and a['status']==b['status']=='passed'
    return dict(comparable=comparable,**{k:b[k]-a[k] if comparable else None for k in ('J','Q','model_usd')})
def run(source,episodes_root,evals_root,experiment_id,*,judge=False,music_bed=None,music_authorized=False,budget_limit=300):
    if not 0<float(budget_limit)<=300:raise StageError('BUDGET_LIMIT','planner','Policy A/B project credit limit must be positive and <=300USD')
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,56}',experiment_id):raise StageError('EXPERIMENT_ID_INVALID','planner','Use a fresh safe experiment identity')
    for suffix in ('base','prod'):
        if resolve_episode(episodes_root,experiment_id+'-'+suffix).exists():raise StageError('EPISODE_EXISTS','assembler','Use a fresh experiment identity')
    if (Path(evals_root)/'policy-comparisons'/(experiment_id+'.json')).exists():raise StageError('EXPERIMENT_EXISTS','planner','Recorded comparison identity cannot be overwritten')
    spec=read_json(Path(source)/'narrative-spec.json')
    if spec.get('editing_policy'):raise StageError('POLICY_BASELINE','planner','Use an original legacy saved Episode; an edited plan is not its own control')
    if music_bed is not None:raise StageError('POLICY_COMPARABILITY','planner','Editing-only A/B requires the same audio recipe; compare music in a separate experiment')
    # Reserve BOTH bounded evaluation requests before the first paid call.
    from pipelines.ffmpeg.budget import project_limit_usd
    os.environ['FRAMEFOUNDRY_PROJECT_API_BUDGET_USD']=str(min(project_limit_usd(),float(budget_limit)))
    if judge:
        from pipelines.ffmpeg.budget import reserve_project_episode
        from pipelines.ffmpeg.llm_provider import LLMProvider
        LLMProvider.from_env()
        budget=Path(os.environ.get('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(Path(__file__).resolve().parents[1])))
        for suffix in ('base','prod'):
            target=resolve_episode(episodes_root,experiment_id+'-'+suffix)
            if target.exists():raise StageError('EPISODE_EXISTS','assembler','Use a fresh experiment identity')
            reserve_project_episode(budget,str(target))
    arms={};evaluations={};limitations=[]
    for name,suffix,production in (('baseline','base',False),('production','prod',True)):
        try:
            root,qa=rerender(source,episodes_root,experiment_id+'-'+suffix,evals_root=evals_root,judge=judge,production=production)
            arms[name]=row(root,qa);evaluations[name]=read_json(root/'eval-report.json')
        except (StageError,OSError,ValueError):
            arms[name]=dict(episode_id=experiment_id+'-'+suffix,status='failed',J=None,Q=None,model_usd=None,known_model_usd=None,cost_status='unknown');limitations.append(name+' failed; retain Episode diagnostics')
    delta=compare(arms['baseline'],arms['production'],evaluations['baseline'],evaluations['production']) if len(evaluations)==2 else dict(comparable=False,J=None,Q=None,model_usd=None)
    if not delta['comparable']:limitations.append('No improvement claim: scores/cost/matched evaluation are unavailable or failed')
    result=dict(schema_version='1.0',status='recorded',experiment_id=experiment_id,created_at=datetime.now(timezone.utc).isoformat(),**arms,comparison=delta,limitations=limitations)
    directory=Path(evals_root)/'policy-comparisons';atomic_write_json(directory/(experiment_id+'.json'),result);atomic_write_json(directory/'latest.json',result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-episode',required=True);p.add_argument('--episodes-root',type=Path,default=Path(__file__).resolve().parents[1]/'episodes');p.add_argument('--evals-root',type=Path,default=Path(__file__).resolve().parents[1]/'evals');p.add_argument('--experiment',default='policy-'+datetime.now(timezone.utc).strftime('%H%M%S'));p.add_argument('--judge',action='store_true');p.add_argument('--budget-limit-usd',type=float,default=300)
    a=p.parse_args(argv)
    try:r=run(resolve_episode(a.episodes_root,a.from_episode),a.episodes_root,a.evals_root,a.experiment,judge=a.judge,budget_limit=a.budget_limit_usd)
    except (StageError,OSError,ValueError) as e:print(str(e),file=sys.stderr);return 2
    print(r);return 0 if r['baseline']['status']==r['production']['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
