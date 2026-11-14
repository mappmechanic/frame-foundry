"""Read-only local episode projections; no provider or evaluation dispatch."""
from pathlib import Path
from functools import lru_cache
import json,math
from .contracts import resolve_episode, StageError, validate_report
from .evaluator import validate_eval_report, model_cost, sha256_file

ARTIFACTS={'finished.mp4','contact-sheet.jpg','narrative-spec.json','qa-report.json','eval-report.json','timeline.otio'}


def artifact(root,eid,name):
    if name not in ARTIFACTS:raise ValueError('Unsupported artifact')
    directory=resolve_episode(root,eid)
    path=directory/name
    if (Path(root)/eid).is_symlink() or path.is_symlink() or path.resolve().parent!=directory:
        raise ValueError('Unsafe artifact path')
    if not path.is_file():raise FileNotFoundError(name)
    return path


def load(root,eid,name):
    directory=resolve_episode(root,eid);path=directory/name
    if (Path(root)/eid).is_symlink() or path.is_symlink() or path.resolve().parent!=directory:return None
    try:
        if path.stat().st_size>10*1024*1024:return None
        value=json.loads(path.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError()))
        return value if isinstance(value,dict) else None
    except (OSError,ValueError):return None


@lru_cache(maxsize=128)
def _hash(path,size,mtime,ctime,inode):
    return sha256_file(Path(path))


def detail(root,eid,job=None,active=False):
    directory=resolve_episode(root,eid)
    if (Path(root)/eid).is_symlink():raise ValueError('Unsafe Episode')
    job=job or {};limitations=[]
    state=load(root,eid,'state.json') or {}
    qa=load(root,eid,'qa-report.json')
    spec=load(root,eid,'narrative-spec.json') or {}
    evaluation=load(root,eid,'eval-report.json')
    try:
        if qa is not None:validate_report(qa)
        if qa is not None and qa.get('episode_id')!=eid:raise ValueError()
    except Exception:qa=None;limitations.append('QA report invalid or unavailable')
    status='available' if evaluation else 'unavailable'
    try:
        if evaluation:
            validate_eval_report(evaluation)
            if evaluation['episode_id']!=eid:raise ValueError()
            render=artifact(root,eid,'finished.mp4')
            stat=render.stat()
            if evaluation.get('render_hash')!=_hash(str(render),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino):
                status='stale';evaluation=None;limitations.append('Evaluation does not match current finished.mp4')
    except FileNotFoundError:
        # Failed evaluation is useful even when no deliverable exists.
        if evaluation and evaluation['gates']['passed']:
            evaluation=None;status='stale';limitations.append('Evaluated finished clip missing')
    except Exception:
        evaluation=None;status='unavailable';limitations.append('Evaluation invalid or unreadable')
    stage=state.get('stage') or job.get('stage','interrupted')
    if not active and job.get('stage') in ('failed','interrupted') and stage not in ('completed','failed','interrupted'):
        stage=job['stage']
    try:cost=evaluation['model_cost'] if evaluation else model_cost(directory,None)[0]
    except (AttributeError,ValueError,TypeError,OSError):
        cost=dict(usd=None,known_usd=None,status='unknown',quality_per_dollar=None)
        limitations.append('AI cost unavailable: usage ledger invalid')
    reservations=0.0
    for name in ('llm-usage.json','tts-usage.json'):
        ledger=load(root,eid,name)
        if ledger is None and (directory/name).exists():reservations=None;break
        calls=(ledger or {}).get('calls',[])
        if not isinstance(calls,list) or any(not isinstance(c,dict) for c in calls):
            reservations=None;limitations.append('Usage reservations unavailable: invalid ledger');break
        for call in calls:
            if call.get('cost_usd') is None and call.get('status')!='blocked':
                try:
                    value=float(call.get('reserved_usd',0))
                    if not math.isfinite(value) or value<0:raise ValueError()
                    reservations+=value
                except (TypeError,ValueError):reservations=None;break
        if reservations is None:break
    shots=[]
    try:
        artifact(root,eid,'timeline.otio')
        from .eval_gates import read_shots
        ordered,_=read_shots(directory)
        segments={s['segment_id']:s for s in spec.get('segments',[]) if isinstance(s,dict) and s.get('segment_id')}
        for shot in ordered:
            shots.append(dict(shot_id=shot['shot_id'],start_ms=shot['timeline_start_ms'],end_ms=shot['timeline_end_ms'],source_id=shot['source_id'],
                reason='; '.join(segments[sid].get('reason','') for sid in shot['selection_ids'] if sid in segments)))
    except Exception:
        if spec.get('segments'):limitations.append('Final shot timestamps unavailable; timeline missing or invalid')
    settings=job.get('settings') or dict(target_seconds=spec.get('target_duration_ms',45000)/1000,
                                        aspect=spec.get('aspect_ratio','9:16'),voiceover=bool(spec.get('voiceover')))
    video=None;sheet=None
    for name in ('finished.mp4','contact-sheet.jpg'):
        try:
            path=artifact(root,eid,name)
            if path.stat().st_size>0:
                url=f'/api/episodes/{eid}/artifacts/{name}'
                if name=='contact-sheet.jpg':sheet=url
                elif not active and qa and qa['passed'] and status!='stale':video=url
        except (ValueError,FileNotFoundError,OSError,StageError):pass
    err=job.get('error') or (qa or {}).get('error') or state.get('error')
    if isinstance(err,dict):err=err.get('message') or err.get('code')
    return dict(episode_id=eid,stage=stage,active=active,elapsed_seconds=(qa or {}).get('elapsed_seconds'),
        qa_passed=qa['passed'] if qa else None,Q=evaluation['scores']['Q'] if evaluation else None,
        D=evaluation['scores']['D'] if evaluation else None,model_cost=cost,reservations_usd=reservations,
        error=str(err) if err else None,created_at=job.get('created_at') or state.get('updated_at') or '',
        goal=job.get('goal') or (qa or {}).get('goal') or spec.get('goal',''),aspect_ratio=spec.get('aspect_ratio',settings['aspect']),
        coverage=(qa or {}).get('coverage') or spec.get('coverage',dict(status='unavailable',limitations=[])),shots=shots,
        voiceover_text=(spec.get('voiceover') or {}).get('text'),video_url=video,contact_sheet_url=sheet,
        gates=evaluation['gates']['checks'] if evaluation else [],judge=evaluation['judge'] if evaluation else dict(status='unavailable',scores=None,fix_instruction=None),
        limitations=limitations+(evaluation or {}).get('limitations',[]),evaluation_status=status,settings=settings,can_regenerate=False)
