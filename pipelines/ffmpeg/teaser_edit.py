"""Short, source-backed edits; only cut outside protected speech components."""
from copy import deepcopy
import hashlib
import math
from itertools import permutations
from .render import output_frames


def enabled(analysis):
    return 'visual_cues' in analysis


def regions(source, words):
    duration=source['duration_ms']; merged=[]
    for w in sorted(words,key=lambda w:w['start_ms']):
        a,b=max(0,w['start_ms']-80),min(duration,w['end_ms']+120)
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    # Group complete short components into a phrase only while <=2.8s.
    phrases=[]
    for a,b in merged:
        if phrases and b-phrases[-1][0]<=2800:phrases[-1][1]=b
        else:phrases.append([a,b])
    merged=phrases
    result=[];cursor=0
    for a,b in merged:
        if a>cursor:result.append((cursor,a,'broll'))
        if b-a<=3000:result.append((a,b,'speech'))
        cursor=b
    if cursor<duration:result.append((cursor,duration,'broll'))
    return result


def short_candidates(analysis, base):
    result=[]
    for source in analysis['sources']:
        sid=source['source_id'];words=[w for w in analysis.get('words',[]) if w['source_id']==sid]
        templates=[c for c in base if c['source_id']==sid]
        for lo,hi,kind in regions(source,words):
            if hi-lo<500:continue
            count=max(1,(hi-lo+2999)//3000) if kind=='broll' else 1
            edges=[lo+(hi-lo)*i//count for i in range(count+1)]
            for a,b in zip(edges,edges[1:]):
                middle=(a+b)/2
                c=deepcopy(min(templates,key=lambda c:abs((c['start_ms']+c['end_ms'])/2-middle))) if templates else dict(source_id=sid,quality={},audio={})
                c.update(start_ms=a,end_ms=b,edit_kind=kind,
                    candidate_id='candidate-'+hashlib.sha256(f'{sid}:{a}:{b}:short4'.encode()).hexdigest()[:20])
                # Preserve measurement provenance before replacing the preview frame.
                base_time=c.get('quality_frame_time_ms',c.get('frame_time_ms'))
                base_in_window=bool(c.get('frame_in_window') and base_time is not None and a<=base_time<b)
                c['quality_frame_time_ms']=base_time if base_in_window else None
                if not base_in_window:
                    for metric in ('sharpness','exposure'):
                        c['quality'][metric]=None
                frames=[f for f in source.get('keyframes',[]) if a<=f['source_time_ms']<b]
                frame=min(frames,key=lambda f:abs(f['source_time_ms']-middle)) if frames else None
                c.update(frame_path=frame['path'] if frame else None,frame_time_ms=frame['source_time_ms'] if frame else None,
                         frame_in_window=bool(frame),canonical_frame_path=frame['path'] if frame else None)
                cues=[q for q in analysis['visual_cues'].get(sid,[]) if a<=q['time_ms']<b and not q['method'].startswith('unavailable')]
                q=min(cues,key=lambda q:abs(q['time_ms']-middle)) if cues else {}
                motion=q.get('motion')
                if motion is None:
                    scenes=[s for s in analysis.get('scenes',[]) if s['source_id']==sid and s['start_ms']<=middle<s['end_ms']]
                    motion=max((s.get('motion',0) for s in scenes),default=0)
                # Scene brightness is a sampled-frame mean, not an interval mean.
                sampled={f['path']:f['source_time_ms'] for f in frames}
                brightness={sampled[s['thumbnail_path']]:s['brightness'] for s in analysis.get('scenes',[])
                            if s['source_id']==sid and s.get('thumbnail_path') in sampled
                            and _finite(s.get('brightness'))}
                c['brightness_samples']=[dict(time_ms=t,brightness=v) for t,v in brightness.items()]
                c['quality']['brightness']=sum(brightness.values())/len(brightness) if brightness else None
                c['cue_time_ms']=q.get('time_ms')
                c['quality'].update(face_count=q.get('face_count'),person_count=q.get('person_count'),motion=motion,
                                    cue_method=q.get('method','scene_sparse_motion;face_person_unavailable'))
                c['transcript_snippet']=' '.join(w['text'] for w in words if a<=w['start_ms'] and w['end_ms']<=b)[:160]
                c['provenance']=[dict(source_id=sid,start_ms=a,end_ms=b)]
                result.append(c)
    return result


def hook(candidate,labels=()):
    q=candidate.get('quality',{})
    return (q.get('face_count') or 0)>0 or (q.get('person_count') or 0)>0 or (q.get('motion') or 0)>=.20 or (candidate.get('frame_in_window') and bool(set(labels)&{'people','action'}))


def score(candidate,labels=()):
    q=candidate.get('quality',{});labels=set(labels)
    base=q.get('visual_score',0)
    return base+20*min(2,(q.get('face_count') or 0)+(q.get('person_count') or 0))+30*(q.get('motion') or 0)+20*bool(labels&{'people','action','scenic_wide'})-25*bool(labels&{'interior','screen'} and not labels&{'people','action'})



# Versioned conservative sparse-frame guards, not calibrated aesthetic scores.
CLOSING_POLICY = 'closing-v2'
CLOSING_MIN_EXPOSURE = 15.0  # candidate luma/clipping score, 0..100
CLOSING_MIN_SHARPNESS = 8.0  # candidate positive Laplacian mean * 4, 0..100
CLOSING_MIN_BRIGHTNESS = .12  # refinery sampled luma mean, 0..1


def _inside(time, segment):
    return isinstance(time, (int, float)) and segment['raw_start_ms'] <= time < segment['raw_end_ms']


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def closing_qualifies(candidate, segment, labels=()):
    """Only evidence inside the actual edit qualifies; motion is never sufficient."""
    q = candidate.get('quality', {})
    quality_time = candidate.get('quality_frame_time_ms', candidate.get('frame_time_ms'))
    if not candidate.get('frame_in_window') or not _inside(quality_time, segment):
        return False
    if not all(_finite(q.get(k)) and q[k] >= minimum for k, minimum in
               (('exposure', CLOSING_MIN_EXPOSURE), ('sharpness', CLOSING_MIN_SHARPNESS))):
        return False
    samples = candidate.get('brightness_samples')
    values = [v['brightness'] for v in samples or [] if _inside(v['time_ms'], segment)]
    brightness = (sum(values)/len(values) if values else None) if samples is not None else q.get('brightness')
    if _finite(brightness) and brightness < CLOSING_MIN_BRIGHTNESS:
        return False
    local_people = (_inside(candidate.get('cue_time_ms'), segment)
                    and not q.get('cue_method', 'unavailable').startswith('unavailable')
                    and ((q.get('face_count') or 0) > 0 or (q.get('person_count') or 0) > 0))
    semantic = (_inside(candidate.get('frame_time_ms'), segment)
                and bool(set(labels) & {'people', 'scenic_wide'}))
    return bool(local_people or semantic)


def _touches(first, second):
    a, b = first[1], second[1]
    return a['source_id'] == b['source_id'] and a['raw_end_ms'] == b['raw_start_ms']


def _closing_order(entries, count):
    """Keep the hook fixed; stable bounded search separates touching source edits."""
    if len(entries) <= count:
        return None
    choices = [e for e in entries[1:] if closing_qualifies(e[0], e[1], e[2])]
    budget = 5000
    for tail in permutations(choices, count):
        if any(_touches(a, b) for a, b in zip(tail, tail[1:])):
            continue
        remaining = [e for e in entries[1:] if not any(e is t for t in tail)]
        def arrange(prefix, pending):
            nonlocal budget
            budget -= 1
            if budget < 0:
                return None
            if not pending:
                return prefix + list(tail) if not _touches(prefix[-1], tail[0]) else None
            for i, entry in enumerate(pending):
                if not _touches(prefix[-1], entry):
                    found = arrange(prefix + [entry], pending[:i] + pending[i+1:])
                    if found is not None:
                        return found
            return None
        found = arrange([entries[0]], remaining)
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def enforce_closing(entries, pool, semantic, blocked):
    """Reorder selected evidence first, then replace weak tail-sized edits only."""
    if len(entries) < 2:
        return entries
    for count in (2, 1):
        found = _closing_order(entries, count)
        if found is not None:
            return found
        trial = list(entries)
        # Up to two weak non-hook shots; exact duration/frame count stays unchanged.
        for index in reversed(range(1, len(trial))):
            old = trial[index]
            if closing_qualifies(old[0], old[1], old[2]):
                continue
            duration = old[1]['raw_end_ms'] - old[1]['raw_start_ms']
            alternatives = []
            for c in pool:
                if c['candidate_id'] in blocked:
                    continue
                end = c['start_ms'] + duration
                if end > c['end_ms'] or (c.get('edit_kind') != 'broll' and end != c['end_ms']):
                    continue
                segment = dict(source_id=c['source_id'], raw_start_ms=c['start_ms'], raw_end_ms=end,
                               reason='Evidence-backed closing payoff; in-window people/faces or scenic wide; closing-v2 numeric guards')
                labels = semantic.get(c['candidate_id'], ())
                if not closing_qualifies(c, segment, labels):
                    continue
                # Unused means nonoverlapping with ALL selected edits, even the replaced shot.
                if any(c['source_id'] == e[1]['source_id'] and c['start_ms'] < e[1]['raw_end_ms']
                       and e[1]['raw_start_ms'] < end for e in entries + trial):
                    continue
                alternatives.append((c, segment, labels))
            for replacement in sorted(alternatives, key=lambda e: -score(e[0], e[2])):
                proposed = trial[:index] + [replacement] + trial[index+1:]
                found = _closing_order(proposed, count)
                if found is not None:
                    return found
            if alternatives:
                trial[index] = max(alternatives, key=lambda e: score(e[0], e[2]))
                if sum(closing_qualifies(*e) for e in trial[1:]) >= count:
                    break
    return entries


def complete(analysis,intent,response,pool):
    """Reconcile compact model decisions with a full local editable-window pool."""
    result=deepcopy(response);result['segments']=[];entries=[];used=[]
    target_frames=output_frames(intent.target_duration_ms/1000);frames=0
    def free(c):
        return not any(c['source_id']==s['source_id'] and c['start_ms']<s['raw_end_ms'] and s['raw_start_ms']<c['end_ms'] for s in used)
    def add(c,reason,labels=()):
        nonlocal frames
        left=target_frames-frames
        if left<=0 or not free(c):return False
        length=output_frames((c['end_ms']-c['start_ms'])/1000)
        end=c['end_ms']
        if length>left:
            if c.get('edit_kind')!='broll':return False
            end=c['start_ms']+round(left*1000/30);length=left
        segment=dict(source_id=c['source_id'],raw_start_ms=c['start_ms'],raw_end_ms=end,reason=reason)
        used.append(segment);entries.append((c,segment,labels));frames+=length
        return True
    # Local labels apply only to candidates actually mapped to the model's supplied windows.
    semantic={}
    blocked=set()
    model=[]
    for s in response['segments']:
        options=[c for c in pool if c['source_id']==s['source_id'] and c['start_ms']<=s['raw_start_ms']<s['raw_end_ms']<=c['end_ms']]
        if not options:continue
        c=options[0];labels=s.get('semantic_labels',[]) if c.get('frame_in_window') and (c.get('frame_time_ms') is None or _inside(c.get('frame_time_ms'),s)) else []
        semantic[c['candidate_id']]=labels
        if set(labels)&{'interior','screen'} and not set(labels)&{'people','action'} and not ((c['quality'].get('face_count') or 0)>0 or (c['quality'].get('person_count') or 0)>0):
            blocked.add(c['candidate_id'])
            continue
        model.append((c,s['reason'],labels))
    qualified=[(c,r,l) for c,r,l in model if hook(c,l)]
    if qualified:
        first=max(qualified,key=lambda item:score(item[0],item[2]));add(*first)
    else:
        alternatives=[c for c in pool if hook(c) and c['candidate_id'] not in blocked]
        if alternatives:add(max(alternatives,key=score),'Locally enforced people/person or high-motion opening; detector/sparse evidence, not semantic certainty')
    for c,reason,labels in model:add(c,reason,labels)
    # Interleave sources so adjacent touching windows cannot coalesce into long shots.
    pending=sorted([c for c in pool if c['candidate_id'] not in blocked],key=lambda c:(-score(c,semantic.get(c['candidate_id'],[])),c['source_id'],c['start_ms']))
    while frames<target_frames and pending:
        seen={e[0]['source_id'] for e in entries};previous=entries[-1][0]['source_id'] if entries else None
        options=[c for c in pending if free(c)]
        if not options:break
        choices=sorted(options,key=lambda c:(c['source_id']==previous,c['source_id'] in seen,-score(c,semantic.get(c['candidate_id'],[]))))
        c=choices[0];pending.remove(c)
        add(c,'Deterministic short word-safe duration fill; may be beyond the eight LLM-inspected windows')
    # Preserve model order except the explicit hook; separate adjacent same-source touching cuts.
    ordered=[];pending=list(entries)
    while pending:
        index=0
        if ordered:
            last=ordered[-1][1]
            safe=[i for i,e in enumerate(pending) if e[1]['source_id']!=last['source_id'] or e[1]['raw_start_ms']!=last['raw_end_ms']]
            index=(safe or [0])[0]
            if len(pending)==2:
                for i in safe:
                    first,last_entry=pending[i][1],pending[1-i][1]
                    if first['source_id']!=last_entry['source_id'] or first['raw_end_ms']!=last_entry['raw_start_ms']:
                        index=i;break
        ordered.append(pending.pop(index))
    result['segments']=[s for _,s,_ in ordered]
    if not result['segments']:
        from .contracts import StageError
        raise StageError('NO_USABLE_MOMENTS','planner','No short protected-speech or b-roll windows available')
    raw=sum(s['raw_end_ms']-s['raw_start_ms'] for s in result['segments'])
    # Canonical JSON's raw budget must agree with the frame-grid duration to <=1 frame.
    if raw>intent.target_duration_ms:
        delta=raw-intent.target_duration_ms
        for c,s,_ in reversed(ordered):
            if c.get('edit_kind')=='broll' and s['raw_end_ms']-s['raw_start_ms']>delta:
                s['raw_end_ms']-=delta;break
    # Normalize the raw budget BEFORE payoff selection so a final trim cannot
    # silently remove the qualifying frame or detector cue.
    ordered=enforce_closing(ordered,pool,semantic,blocked)
    result['segments']=[s for _,s,_ in ordered]
    total=sum(s['duration_ms'] for s in analysis['sources'])
    usable=sum(c['end_ms']-c['start_ms'] for c in pool)
    result['assumptions'].append(f'Source duration {total}ms; short speech/b-roll window coverage {usable}ms; selected encoded duration {frames*1000/30:.0f}ms. Eight-window model view is not total footage.')
    result['assumptions'].append('Short word-safe reconciliation/duration completion may reorder the hook and add locally ranked footage beyond model evidence; speech guards remain 80ms/120ms')
    limitations=[l for l in result['coverage']['limitations'] if not any(t in l.lower() for t in ('duration','footage is less','insufficient footage','shorter','shortfall'))]
    if frames<target_frames:limitations.append(f'Short protected-speech/b-roll coverage cannot fill target: {target_frames-frames} output frames short; raw source duration {total}ms is not the usable-selection duration')
    if not hook(ordered[0][0],ordered[0][2]):limitations.append('Hook constraint unavailable: no measured people/person, evidence-backed action, or high-motion window found')
    closing=[closing_qualifies(c,s,l) for c,s,l in ordered[-2:]]
    result['assumptions'].append(f'{CLOSING_POLICY}: final two closing qualifiers={closing}; exposure>={CLOSING_MIN_EXPOSURE:g}, sharpness>={CLOSING_MIN_SHARPNESS:g}; sampled brightness>={CLOSING_MIN_BRIGHTNESS:g} when available. Missing/out-of-window quality or subject evidence cannot qualify; sparse cues are not exhaustive inspection.')
    if not closing[-1]:
        limitations.append('Closing constraint unavailable: final shot lacks safe in-window people/faces/scenic evidence with measured exposure and sharpness; no invented scenic labels or high-motion-only qualification')
    elif len(closing)<2 or not closing[-2]:
        limitations.append('Closing constraint partial: only final shot qualifies; two safe noncoalescing closing choices unavailable without changing hook/duration/word protection')
    result['coverage']=dict(status='partial' if limitations else 'full',limitations=limitations)
    return result
