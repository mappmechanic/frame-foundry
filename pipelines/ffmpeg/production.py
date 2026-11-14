"""Original, opt-in deterministic editorial rules. Source evidence outranks targets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from .contracts import StageError

CONFIG=Path(__file__).resolve().parents[2]/'contracts/reel-policy.json'
def config():return json.loads(CONFIG.read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def plan_hash(spec):return digest([[s['source_id'],s['raw_start_ms'],s['raw_end_ms']] for s in spec['segments']])
def enabled(ctx,spec=None):return bool(getattr(ctx,'production',False) or (spec or {}).get('editing_policy',{}).get('mode')=='production-v1')
def words_in(segment,analysis):
    return sorted([w for w in analysis.get('words',[]) if w['source_id']==segment['source_id'] and w['start_ms']<segment['raw_end_ms'] and segment['raw_start_ms']<w['end_ms']],key=lambda w:w['start_ms'])
def hook_score(segment,root):
    e=segment.get('semantic_evidence') or {};t=e.get('frame_time_ms')
    if e.get('method')!='planner_in_window_tile' or e.get('source_id')!=segment['source_id'] or type(t) is not int or not segment['raw_start_ms']<=t<segment['raw_end_ms']:return 0
    try:
        p=(Path(root)/e['frame_path']).resolve()
        if not p.is_relative_to(Path(root).resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=e['frame_sha256']:return 0
    except (OSError,KeyError):return 0
    if 'screen' in e.get('labels',[]):return 0
    if t-segment['raw_start_ms']>config()['hook_by_ms']:return 0
    return 3*bool(set(e.get('subjects',[]))&{'family_activity','parade','amusement_ride','hiking'})+2*('action' in e.get('labels',[]))+('people' in e.get('labels',[]))
def _quiet(bins,start,end):
    rows=[b for b in bins if b['start_ms']<end and start<b['end_ms']]
    return bool(rows) and rows[0]['start_ms']<=start and rows[-1]['end_ms']>=end and all(b.get('active') is False for b in rows) and all(a['end_ms']>=b['start_ms'] for a,b in zip(rows,rows[1:]))
def _bins(ctx,source,cache,limitations):
    sid=source['source_id']
    if sid in cache:return cache[sid]
    cache[sid]=[]
    try:
        from .audio_analysis import analyze_audio
        p=(Path(ctx.episode_dir)/source['proxy_path']).resolve()
        if not p.is_relative_to(Path(ctx.episode_dir).resolve()):raise ValueError()
        audio=analyze_audio(p);offset=source.get('proxy_time_offset_ms') or 0
        cache[sid]=[dict(b,start_ms=b['start_ms']+offset,end_ms=b['end_ms']+offset) for b in audio['bins']]
    except (OSError,ValueError,KeyError,TypeError,StageError):limitations.append('Measured silence unavailable for a retained source; speech and pacing exception preserved')
    return cache[sid]
def apply(ctx,spec,analysis):
    if not enabled(ctx,spec):return spec
    old=spec.get('editing_policy') or {}
    if old.get('output_plan_hash')==plan_hash(spec):return deepcopy(spec)
    cfg=config();amended=deepcopy(spec);changes=[];limitations=[];segments=amended['segments'];sources={s['source_id']:s for s in analysis['sources']};cache={}
    # Strip only repeated COMPLETE exact utterances from the same source, within30s.
    drop=set()
    for i,s in enumerate(segments):
        ws=words_in(s,analysis);text=' '.join(w['text'] for w in ws).strip()
        complete=bool(ws and re.search(r'[.!?]$',text) and s['raw_start_ms']<=ws[0]['start_ms'] and ws[-1]['end_ms']<=s['raw_end_ms'])
        if not complete:continue
        for j,later in enumerate(segments[i+1:],i+1):
            lw=words_in(later,analysis)
            if (later['source_id']==s['source_id'] and 0<later['raw_start_ms']-s['raw_start_ms']<=cfg['retake_window_ms'] and lw and later['raw_start_ms']<=lw[0]['start_ms'] and lw[-1]['end_ms']<=later['raw_end_ms'] and ' '.join(w['text'] for w in lw).strip().casefold()==text.casefold()):drop.add(i);break
    if drop:changes.append('Kept final complete exact retake within the same-source30s window')
    kept=[]
    for i,s in enumerate(segments):
        if i in drop:continue
        ws=words_in(s,analysis);parts=[(s['raw_start_ms'],s['raw_end_ms'])]
        if len(ws)>1:
            bins=_bins(ctx,sources[s['source_id']],cache,limitations);quiet=[]
            for left,right in zip(ws,ws[1:]):
                a,b=left['end_ms']+cfg['speech_post_ms'],right['start_ms']-cfg['speech_pre_ms']
                if b-a>cfg['dead_air_ms'] and _quiet(bins,a,b):quiet.append((a,b))
            if quiet:
                parts=[];cursor=s['raw_start_ms']
                for a,b in quiet:
                    if a-cursor>=cfg['shot_min_ms'] and s['raw_end_ms']-b>=cfg['shot_min_ms']:parts.append((cursor,a));cursor=b;changes.append('Removed measured inter-word dead air outside80/120ms protected speech')
                parts.append((cursor,s['raw_end_ms']))
        elif not ws and s['raw_end_ms']-s['raw_start_ms']>cfg['shot_max_ms']:
            # No-word does not prove no speech. Only explicitly visual-only/scenic tiles qualify.
            e=s.get('semantic_evidence') or {}
            if analysis.get('transcription_status')=='visual_only_unavailable':
                n=(s['raw_end_ms']-s['raw_start_ms']+cfg['shot_max_ms']-1)//cfg['shot_max_ms'];edges=[s['raw_start_ms']+(s['raw_end_ms']-s['raw_start_ms'])*j//n for j in range(n+1)];parts=list(zip(edges,edges[1:]));changes.append('Split an evidenced visual shot into bounded pacing windows')
            else:limitations.append('No-word visual interval is not verified silent speech; long shot retained')
        for a,b in parts:
            piece=deepcopy(s);piece.update(raw_start_ms=a,raw_end_ms=b,render_start_ms=None,render_end_ms=None)
            e=piece.get('semantic_evidence') or {}
            if e and not a<=e.get('frame_time_ms',-1)<b:piece.pop('semantic_evidence',None)
            if b-a<cfg['shot_min_ms']:limitations.append('Shot shorter than target because the retained source window is limited')
            if b-a>cfg['shot_max_ms']:limitations.append('Shot target exceeded to preserve continuous protected speech/source evidence')
            kept.append(piece)
    if len(kept)>64:raise StageError('POLICY_LIMIT','assembler','Production policy exceeds64 protected selections')
    scores=[hook_score(s,ctx.episode_dir) for s in kept];index=max(range(len(kept)),key=lambda i:scores[i])
    hook='verified' if scores[index]>0 else 'unavailable'
    if index and scores[index]>scores[0] and not re.search(r'\bchronological\b',spec['goal'],re.I):kept.insert(0,kept.pop(index));changes.append('Promoted an already retained source-bound hook to opening time0')
    if hook=='unavailable':limitations.append('Hook evidence unavailable; no inferred claim of an engaging opening')
    hook='verified' if hook_score(kept[0],ctx.episode_dir)>0 else 'unavailable'
    if hook=='unavailable' and 'Hook evidence unavailable; no inferred claim of an engaging opening' not in limitations:limitations.append('Hook not safely promoted under goal/source constraints')
    for i,s in enumerate(kept):s['segment_id']=f'production-{i+1:03d}'
    amended['segments']=kept
    amended['editing_policy']=dict(mode='production-v1',config_hash=digest(cfg),input_plan_hash=plan_hash(spec),output_plan_hash=plan_hash(amended),hook_status=hook,changes=list(dict.fromkeys(changes)),limitations=list(dict.fromkeys(limitations)),audio=None,visual=[])
    if drop or any('Removed measured' in c for c in changes):amended['coverage']['status']='partial';amended['coverage']['limitations'].append('Production policy removed exact retakes or measured dead air; retained speech protection remains authoritative')
    return amended

def dialogue_intervals(timeline,analysis,spec):
    from .render import output_frames
    rows=[];cursor=0
    for clip in timeline.tracks[0]:
        a=clip.source_range.start_time.to_seconds()*1000;duration=clip.source_range.duration.to_seconds()*1000;sid=clip.metadata['source_id'];length=output_frames(duration/1000)/30
        for w in analysis.get('words',[]):
            if w['source_id']==sid and a<=w['start_ms'] and w['end_ms']<=a+duration:
                rows.append((max(cursor,cursor+(w['start_ms']-80-a)/1000),min(cursor+length,cursor+(w['end_ms']+120-a)/1000)))
        cursor+=length
    if spec.get('voiceover'):rows.append((0,min(cursor,spec['voiceover']['duration_ms']/1000)))
    return rows

def verify(spec,analysis,*,root=None):
    p=spec.get('editing_policy')
    if not p:return []
    rows=[]
    def check(name,ok,reason=None):rows.append(dict(name=name,status='pass' if ok else 'fail',reason=reason,required=True))
    check('policy_configuration',p.get('config_hash')==digest(config()) and p.get('output_plan_hash')==plan_hash(spec),'Policy configuration/source plan binding')
    check('policy_hook',p.get('hook_status') in ('verified','unavailable') and (p.get('hook_status')!='verified' or root is None or hook_score(spec['segments'][0],root)>0),'Missing hook evidence is disclosed, not inferred')
    check('policy_pacing',all(s['raw_end_ms']>s['raw_start_ms'] for s in spec['segments']) and len(spec['segments'])<=64,'Source-preserving shot targets/recorded exceptions')
    audio=p.get('audio')
    check('policy_audio',isinstance(audio,dict) and audio.get('status') in ('absent','mixed'),'Optional music must have a measured receipt or explicit absence')
    cards=(spec.get('overlays') or {}).get('cards',[]);w,h=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
    safe=all(c['y']>=.10*h and c['y']+c['height']<=.78*h and c['x']+c['width']<=.85*w for c in cards) if h>w else all(c['x']>=.10*w and c['x']+c['width']<=.90*w and c['y']>=.10*h and c['y']+c['height']<=.90*h for c in cards)
    check('policy_visual_safe_zones',safe,'Production exclusion/title-safe geometry')
    visual=p.get('visual',[])
    check('policy_subject_collision',all(not r.get('collision',False) for r in visual),'Verified sampled subject collision blocks delivery')
    return rows
