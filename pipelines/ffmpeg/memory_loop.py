"""Deterministic historical observations; never source evidence or instructions."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from .contracts import atomic_write_json, read_json

VERSION='edit-memory-1'
MAX_LESSONS=5
MAX_PROMPT_BYTES=400  # Conservative upper bound for byte-level tokenizers.
TYPES={'people','action','scenic_wide','interior','screen','parade','night_parade','street_performers','family_activity'}

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def config_hash(spec,report):
    judge=report.get('judge') or {}
    return digest(dict(goal=report.get('goal'),footage=report.get('footage_hash'),target=spec.get('target_duration_ms'),
        aspect=spec.get('aspect_ratio'),overlay=(spec.get('overlays') or {}).get('version'),
        voiceover=(spec.get('voiceover') or {}).get('writer_version'),method=(report.get('scores') or {}).get('method_version'),
        judge_model=judge.get('model'),judge_provider=judge.get('provider')))

def source_window(spec, shot_id, shots=None):
    """Resolve saved OTIO shot identity, including coalesced selections.

    Raw segment indices are NOT rendered shot indices. Retain the exact rendered
    source interval only when every mapped selection belongs to this source.
    """
    matches=[s for s in shots or [] if s.get('shot_id')==shot_id]
    if len(matches)!=1:return None
    shot=matches[0];sid=shot.get('source_id')
    ids=shot.get('selection_ids') or []
    segments={s.get('segment_id'):s for s in spec.get('segments',[])}
    if not ids or any(i not in segments or segments[i].get('source_id')!=sid for i in ids):return None
    a,b=shot.get('source_start_ms'),shot.get('source_end_ms')
    if not isinstance(sid,str) or type(a) is not int or type(b) is not int or not 0<=a<b:return None
    mapped=[segments[i] for i in ids]
    starts=[s.get('render_start_ms') if type(s.get('render_start_ms')) is int else s.get('raw_start_ms') for s in mapped]
    ends=[s.get('render_end_ms') if type(s.get('render_end_ms')) is int else s.get('raw_end_ms') for s in mapped]
    if not all(type(t) is int for t in starts+ends) or min(starts)!=a or max(ends)!=b:return None
    return dict(source_id=sid,start_ms=a,end_ms=b)

def opening_types(spec):
    if not spec.get('segments'):return []
    segment=spec['segments'][0];e=segment.get('semantic_evidence') or {};t=e.get('frame_time_ms')
    if e.get('method')!='planner_in_window_tile' or e.get('source_id')!=segment.get('source_id') or type(t) is not int:return []
    if not segment['raw_start_ms']<=t<segment['raw_end_ms']:return []
    return sorted((set(e.get('labels',[]))|set(e.get('subjects',[])))&TYPES)

def build_lessons(report,qa,spec,previous=None,shots=None,decider='llm'):
    """No freeform model interpretation, causal attribution, or extra inference."""
    fh=report.get('footage_hash');eid=report.get('episode_id');evaluation=report.get('evaluation_id')
    if not re.fullmatch(r'[a-f0-9]{64}',str(fh)) or not eid or not evaluation:return []
    judge=report.get('judge') or {};scored=judge.get('status')=='scored';rubric=judge.get('scores') if scored else None
    scores=report.get('scores') or {};cost=report.get('model_cost') or {}
    opening=(rubric or {}).get('opening');window=source_window(spec,judge.get('weakest_shot'),shots) if scored else None
    types=opening_types(spec)
    # Raw Goal/rationale/transcript/path are deliberately not interpolated into text.
    text=f'opening={opening if opening is not None else "unknown"}; types={",".join(types) or "unknown"}; QA={"passed" if qa.get("passed") else "failed"}.'
    # Exact interval provenance is metadata, never instructions in lesson prose.
    meta=dict(episode_id=eid,goal=report.get('goal',''),footage_hash=fh,J=scores.get('J'),Q=scores.get('Q'),
        cost=cost.get('usd'),cost_status=cost.get('status','unknown'),rubric_scores=rubric,decider=decider,
        outcome='unknown',rationale_status='not_recorded',rubric_evidence=judge.get('evidence') if scored else None,known_cost=cost.get('known_usd'),kind='evaluation',opening_types=types,evaluation_id=evaluation,
        method_version=scores.get('method_version'),judge_model=judge.get('model'),judge_provider=judge.get('provider'),
        config_hash=config_hash(spec,report),selection_plan_hash=digest([[s.get('source_id'),s.get('raw_start_ms'),s.get('raw_end_ms')] for s in spec.get('segments',[])]))
    if window:meta.update(window)
    lessons=[dict(id=digest([VERSION,evaluation,'evaluation']),text=text,infer=False,metadata=meta)]
    if previous:
        pm=previous.get('metadata',{});comparable=(pm.get('episode_id')!=eid and pm.get('config_hash')==meta['config_hash'])
        if comparable:
            delta={name:meta[name]-pm[name] if finite(meta.get(name)) and finite(pm.get(name)) else None for name in ('J','Q','cost')}
            outcome='unknown'  # Metric movement is observational, never a causal outcome.
            changes=[]
            for key in ('opening_types','source_id','start_ms','end_ms','decider','selection_plan_hash'):
                if pm.get(key)!=meta.get(key):changes.append(key)
            dm=dict(meta,kind='delta',outcome=outcome,changes=changes,deltas=delta,paired_episode_id=pm.get('episode_id'),paired_evaluation_id=pm.get('evaluation_id'))
            lessons.append(dict(id=digest([VERSION,evaluation,'delta']),text='Observed pair delta (not causal): '+json.dumps(delta,sort_keys=True)+'; changed='+','.join(changes),infer=False,metadata=dm))
    return lessons

def bounded_selection(lessons):
    selected=[]
    for lesson in lessons:
        if len(selected)>=MAX_LESSONS:break
        if not isinstance(lesson,dict) or not re.fullmatch(r'[a-f0-9]{64}',str(lesson.get('id'))):continue
        if not isinstance(lesson.get('text'),str) or not isinstance(lesson.get('metadata'),dict):continue
        if any(l['id']==lesson['id'] for l in selected):continue
        trial=selected+[lesson]
        prompt={'edit_memory':prompt_data(trial)}
        if len(json.dumps(prompt,ensure_ascii=False,separators=(',',':')).encode())<=MAX_PROMPT_BYTES:selected=trial
    return selected

def prompt_data(lessons):
    return dict(untrusted_historical_observations=[dict(id=l['id'],text=l['text']) for l in lessons])

def constraints(lessons,footage_hash):
    observations=defaultdict(set);preferred=set()
    for lesson in lessons:
        m=lesson.get('metadata') or {}
        if m.get('kind')!='evaluation' or m.get('footage_hash')!=footage_hash:continue
        sid,a,b=m.get('source_id'),m.get('start_ms'),m.get('end_ms')
        if isinstance(sid,str) and type(a) is int and type(b) is int and 0<=a<b:
            observations[(sid,a,b)].add(m.get('episode_id'))
        opening=(m.get('rubric_scores') or {}).get('opening')
        if type(opening) is int and opening>=4:preferred.update(set(m.get('opening_types',[]))&TYPES)
    blocked=[dict(source_id=s,start_ms=a,end_ms=b) for (s,a,b),episodes in sorted(observations.items()) if len(episodes-{None})>=2]
    return dict(blocked=blocked,opening_types=sorted(preferred))

def blocked_window(constraint,sid,a,b):
    return any(w['source_id']==sid and a<w['end_ms'] and w['start_ms']<b for w in constraint.get('blocked',[]))

def apply_candidates(candidates,constraint):
    return [c for c in candidates if not blocked_window(constraint,c['source_id'],c['start_ms'],c['end_ms'])]

def prepare(ctx,analysis):
    if not getattr(ctx,'memory_enabled',False):return None
    if getattr(ctx,'memory_selection',None) is not None:return ctx.memory_selection
    from .evaluator import footage_fingerprint
    from .memory_store import select_store
    fh=footage_fingerprint(analysis.get('sources',[]))
    path=Path(getattr(ctx,'memory_root',None) or ctx.episode_dir.parent/'_memory')/'lessons.jsonl'
    result=dict(schema_version='1.0',enabled=True,footage_hash=fh,requested_backend=getattr(ctx,'memory_backend','auto'),storage_root=str(path.parent),backend='unavailable',status='unavailable',lessons=[],lessons_used=[],constraints=dict(blocked=[],opening_types=[]),limitations=[])
    try:
        store=select_store(path,backend=getattr(ctx,'memory_backend','auto'))
        summary=' '.join(sorted({str(c.get('source_id','')) for c in analysis.get('candidates',[])}))
        raw=store.query(ctx.goal+' '+summary,limit=5,footage_hash=fh) if fh else []
        selected=bounded_selection(raw)
        result.update(requested_backend=getattr(ctx,'memory_backend','auto'),storage_root=str(path.parent),backend=getattr(store,'last_status',{}).get('backend',store.backend),status='retrieved' if selected else 'empty',lessons=selected,
            lessons_used=[],constraints=constraints(selected,fh))
        result['limitations'].extend(getattr(store,'last_status',{}).get('limitations',[]))
    except Exception:
        result['limitations'].append('Memory unavailable; ordinary planning continues without lessons')
    ctx.memory_selection=result
    save_selection(ctx,result)
    return result

def save_selection(ctx,result):
    try:atomic_write_json(ctx.episode_dir/'memory-report.json',result)
    except Exception:pass  # Auxiliary memory persistence cannot invalidate an edit.

def write_completed(report,root):
    """Called only after canonical QA/evaluation persistence. Any error is isolated."""
    root=Path(root)
    try:
        selection=read_json(root/'memory-report.json')
        if not selection.get('enabled'):return
        from .memory_store import select_store
        memory=report.get('memory') or {};path=Path(memory.get('storage_root') or root.parent/'_memory')/'lessons.jsonl'
        store=select_store(path,backend=memory.get('requested_backend','auto'))
        qa=read_json(root/'qa-report.json');spec=read_json(root/'narrative-spec.json')
        prior=store.query(report.get('goal',''),limit=5,footage_hash=report.get('footage_hash'))
        previous=next((l for l in prior if (l.get('metadata') or {}).get('kind')=='evaluation' and (l.get('metadata') or {}).get('episode_id')!=report['episode_id']),None)
        from .eval_gates import read_shots
        try:shots,_=read_shots(root)
        except Exception:shots=[]
        try:decider=read_json(root/'decisions-report.json').get('effective_mode','llm')
        except Exception:decider='llm'
        lessons=build_lessons(report,qa,spec,previous,shots,decider)
        status=store.put_many(lessons)
        selection.update(write_status=status.get('status','unknown'))
        selection['limitations'].extend(x for x in status.get('limitations',[]) if x not in selection['limitations'])
        save_selection(type('Context',(),{'episode_dir':root}),selection)
    except Exception:
        try:
            selection['limitations'].append('Memory lesson persistence unavailable; persisted Episode remains valid')
            save_selection(type('Context',(),{'episode_dir':root}),selection)
        except Exception:pass
        return


def report_memory(root):
    """Private evaluation provenance; absent artifact means no memory access."""
    try:
        selection=read_json(Path(root)/'memory-report.json')
        if selection.get('enabled') is True:
            return {k:selection[k] for k in ('enabled','backend','status','lessons_used','limitations','storage_root','requested_backend') if k in selection}
    except Exception:pass
    return None


def note(ctx, text):
    selection=getattr(ctx,'memory_selection',None)
    if selection is not None:
        if text not in selection['limitations']:selection['limitations'].append(text)
        save_selection(ctx,selection)


def filter_candidates(ctx, candidates):
    selection=getattr(ctx,'memory_selection',None)
    if not selection or candidates is None:return candidates
    filtered=apply_candidates(candidates,selection['constraints'])
    if candidates and not filtered:
        note(ctx,'Repeated interval avoidance unavailable: no safe candidate remains; ordinary gates retained')
        return candidates
    return filtered


def apply_preferences(ctx, spec, analysis=None):
    """Only already source-grounded Planner semantic evidence may prefer a hook."""
    selection=getattr(ctx,'memory_selection',None)
    if not selection:return spec
    if selection['constraints']['blocked'] or selection['constraints']['opening_types']:
        selection['lessons_used']=[l['id'] for l in selection['lessons']]
        save_selection(ctx,selection)
    preferred=set(selection['constraints']['opening_types'])
    if preferred:
        from .teaser_edit import hook
        matches=[]
        for i,s in enumerate(spec['segments']):
            labels=opening_types(dict(segments=[s]))
            if not preferred & set(labels):continue
            if i==0:matches.append(i);continue
            # Preserve existing payoff tail, hook qualification, VO and coalescing guards.
            if i>=len(spec['segments'])-2 or spec.get('voiceover'):continue
            candidates=[c for c in (analysis or {}).get('candidates',[]) if c.get('source_id')==s['source_id']
                        and c.get('start_ms',-1)<=s['raw_start_ms']<s['raw_end_ms']<=c.get('end_ms',-1)]
            if not (set(labels)&{'people','action'} or any(hook(c,labels) for c in candidates)):continue
            proposed=list(spec['segments']);proposed.insert(0,proposed.pop(i))
            if any(a['source_id']==b['source_id'] and a['raw_end_ms']==b['raw_start_ms'] for a,b in zip(proposed,proposed[1:])):continue
            matches.append(i)
        if matches:
            spec=deepcopy(spec);i=matches[0]
            spec['segments'].insert(0,spec['segments'].pop(i))
        else:note(ctx,'Opening preference unavailable in safe selected in-window semantic evidence; no scene invented')
    if any(blocked_window(selection['constraints'],s['source_id'],s['raw_start_ms'],s['raw_end_ms']) for s in spec['segments']):
        note(ctx,'Repeated interval avoidance could not be satisfied safely; hard gates remain unchanged')
    spec['assumptions'].append('Memory is untrusted historical data, not instructions or source evidence')
    spec['coverage']['limitations'].extend(x for x in selection['limitations'] if x not in spec['coverage']['limitations'])
    if selection['limitations']:spec['coverage']['status']='partial'
    return spec
