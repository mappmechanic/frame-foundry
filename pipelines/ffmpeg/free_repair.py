"""Optional pre-render reuse of a previous scored weakest shot; zero paid calls.

Never swaps by index alone. Old render/hash/footage/Goal and exact source interval
must agree. This runs BEFORE this Episode's ordinary judge, so J/Q stay render-bound.
"""
from copy import deepcopy
from pathlib import Path
import math
from .contracts import read_json,resolve_episode
from .assembler import build_timeline
from .render import output_frames
from . import teaser_edit,semantics
from .candidates import rank_candidates

PREFIX='Free prior-judge repair: '


def replace(spec,analysis,weak,*,keep_intro=True):
    timeline,_=build_timeline(spec,analysis)
    video=timeline.tracks[0]
    matches=[(i,c) for i,c in enumerate(video) if c.metadata.get('source_id')==weak['source_id']
             and abs(c.source_range.start_time.to_seconds()*1000-weak['source_start_ms'])<=1
             and abs(c.source_range.end_time_exclusive().to_seconds()*1000-weak['source_end_ms'])<=1]
    if len(matches)!=1:return spec,'prior weak source interval is not selected; no index-only swap'
    i,clip=matches[0]
    if i==0 or i>=len(video)-2:return spec,'weak shot is protected hook/closing; left unchanged'
    ids=list(clip.metadata.get('selection_ids',[]))
    if len(ids)!=1:return spec,'coalesced weak shot has multiple raw selections; left unchanged'
    index=next(j for j,s in enumerate(spec['segments']) if s['segment_id']==ids[0])
    old=spec['segments'][index];length=old['raw_end_ms']-old['raw_start_ms']
    pool=teaser_edit.short_candidates(analysis,analysis.get('candidates',[]))
    pool=rank_candidates(pool,spec['goal'],top_n=len(pool))
    options=[]
    for rank,c in enumerate(pool):
        q=c['quality'];motion=q.get('motion')
        if (c.get('edit_kind')!='broll' or c['end_ms']-c['start_ms']<length
            or not q.get('cue_method','').startswith('opencv_cpu_haar_face_hog_person')
            or not ((q.get('face_count') or 0)>0 or (q.get('person_count') or 0)>0)
            or not isinstance(motion,(int,float)) or isinstance(motion,bool) or not math.isfinite(motion)
            or motion<.15):continue
        s=dict(segment_id=old['segment_id'],source_id=c['source_id'],raw_start_ms=c['start_ms'],raw_end_ms=c['start_ms']+length,
               render_start_ms=None,render_end_ms=None,reason='Free prior-judge replacement: unused ranked in-window CPU face/person window; not semantic action verification')
        if not teaser_edit.closing_qualifies(c,s):continue  # exposure, sharpness and in-window cue checks
        if any(s['source_id']==other['source_id'] and s['raw_start_ms']<=other['raw_end_ms']
               and other['raw_start_ms']<=s['raw_end_ms'] for j,other in enumerate(spec['segments']) if j!=index):continue
        if any(other.get('semantic_evidence',{}).get('labels') and 'screen' in other['semantic_evidence']['labels']
               and other['source_id']==s['source_id'] and other['raw_start_ms']<=c.get('frame_time_ms',-1)<other['raw_end_ms'] for other in spec['segments']):continue
        options.append((rank,s))
    before=sum(output_frames(c.duration().to_seconds()) for c in video)
    for _,s in options:
        amended=deepcopy(spec);amended['segments'][index]=s
        if keep_intro and semantics.intro(amended)!=semantics.intro(spec):continue
        proposed,_=build_timeline(amended,analysis)
        if len(proposed.tracks[0])!=len(video) or sum(output_frames(c.duration().to_seconds()) for c in proposed.tracks[0])!=before:continue
        if proposed.tracks[0][0].media_reference.target_url!=video[0].media_reference.target_url or proposed.tracks[0][0].source_range!=video[0].source_range:continue
        if any(a.media_reference.target_url!=b.media_reference.target_url or a.source_range!=b.source_range for a,b in zip(proposed.tracks[0][-2:],video[-2:])):continue
        return amended,f"replaced {weak['shot_id']} by matching source interval with unused duration-matched ranked CPU people window; no paid call, no post-judge score inheritance"
    return spec,'no unused ranked evidenced people window satisfies duration/word/hook/closing/VO guards; left unchanged'


def repair_from(spec,analysis,*,episodes_root,episode_id,keep_intro=True):
    """Missing/stale evidence is a disclosed no-op, not a guessed weak-shot mapping."""
    from .evaluator import footage_fingerprint,sha256_file,validate_eval_report
    from .eval_gates import read_shots
    try:
        prior=resolve_episode(episodes_root,episode_id)
        if (Path(episodes_root)/episode_id).is_symlink():raise ValueError()
        for name in ('analysis.json','narrative-spec.json','eval-report.json','qa-report.json','timeline.otio','finished.mp4'):
            if (prior/name).is_symlink() or not (prior/name).is_file():raise ValueError()
        evaluation=read_json(prior/'eval-report.json');validate_eval_report(evaluation)
        previous=read_json(prior/'analysis.json');old_spec=read_json(prior/'narrative-spec.json');qa=read_json(prior/'qa-report.json')
        fingerprint=footage_fingerprint(analysis['sources'])
        if (not fingerprint or fingerprint!=footage_fingerprint(previous['sources']) or fingerprint!=evaluation['footage_hash']
            or old_spec['goal']!=spec['goal'] or old_spec['aspect_ratio']!=spec['aspect_ratio']
            or old_spec['target_duration_ms']!=spec['target_duration_ms'] or qa.get('passed') is not True
            or evaluation['scores']['G']!=1 or not evaluation['gates']['passed']
            or evaluation['judge']['status']!='scored' or evaluation['render_hash']!=sha256_file(prior/'finished.mp4')):raise ValueError()
        shots,_=read_shots(prior);weak=next(s for s in shots if s['shot_id']==evaluation['judge']['weakest_shot'])
        amended,note=replace(spec,analysis,weak,keep_intro=keep_intro)
    except (OSError,ValueError,KeyError,TypeError,StopIteration,RuntimeError):
        amended,note=spec,'previous scored render/footage/Goal evidence unavailable or stale; left unchanged'
    amended=deepcopy(amended);amended['assumptions'].append(PREFIX+note)
    return amended


def repair_latest(spec,analysis,*,episodes_root,keep_intro=True):
    """Pre-render reuse of the latest compatible scored Episode; never a new judge call.

    Shortlist on inexpensive strict report metadata; repair_from rechecks current
    render/footage hashes, constraints and source times before any substitution.
    Missing history is a no-op. A stale latest candidate never falls back silently
    to an older judgment.
    """
    from .evaluator import footage_fingerprint,validate_eval_report
    root=Path(episodes_root)
    try:directories=list(root.iterdir())
    except OSError:return spec
    # Do not stream raw footage when there is no prior evaluation to inspect.
    reports=[]
    for directory in directories:
        path=directory/'eval-report.json'
        if directory.is_symlink() or path.is_symlink() or not path.is_file():continue
        try:
            resolve_episode(root,directory.name)
            evaluation=read_json(path);validate_eval_report(evaluation)
            if (evaluation['episode_id']==directory.name and evaluation['goal']==spec['goal'] and evaluation['scores']['G']==1
                and evaluation['judge']['status']=='scored' and evaluation['judge']['weakest_shot']):
                prior_spec_path=directory/'narrative-spec.json'
                if prior_spec_path.is_symlink():continue
                previous=read_json(prior_spec_path)
                if (previous['goal'],previous['aspect_ratio'],previous['target_duration_ms'])==(
                    spec['goal'],spec['aspect_ratio'],spec['target_duration_ms']):
                    reports.append(evaluation)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError):continue
    if not reports:return spec
    fingerprint=footage_fingerprint(analysis['sources'])
    compatible=[r for r in reports if fingerprint and r['footage_hash']==fingerprint]
    if not compatible:return spec
    latest=max(compatible,key=lambda r:(r['created_at'],r['episode_id']))
    return repair_from(spec,analysis,episodes_root=root,episode_id=latest['episode_id'],keep_intro=keep_intro)
