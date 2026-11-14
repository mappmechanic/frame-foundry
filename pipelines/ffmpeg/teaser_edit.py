"""Short, source-backed edits; only cut outside protected speech components."""
from copy import deepcopy
import hashlib
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
        c=options[0];labels=s.get('semantic_labels',[]) if c.get('frame_in_window') else []
        semantic[c['candidate_id']]=labels
        if set(labels)&{'interior','screen'} and not set(labels)&{'people','action'} and not hook(c):
            blocked.add(c['candidate_id'])
            continue
        model.append((c,s['reason'],labels))
    qualified=[(c,r,l) for c,r,l in model if hook(c,l)]
    if qualified:
        first=max(qualified,key=lambda item:score(item[0],item[2]));add(*first)
    else:
        alternatives=[c for c in pool if hook(c)]
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
    total=sum(s['duration_ms'] for s in analysis['sources'])
    usable=sum(c['end_ms']-c['start_ms'] for c in pool)
    result['assumptions'].append(f'Source duration {total}ms; short speech/b-roll window coverage {usable}ms; selected encoded duration {frames*1000/30:.0f}ms. Eight-window model view is not total footage.')
    result['assumptions'].append('Short word-safe reconciliation/duration completion may reorder the hook and add locally ranked footage beyond model evidence; speech guards remain 80ms/120ms')
    limitations=[l for l in result['coverage']['limitations'] if not any(t in l.lower() for t in ('duration','footage is less','insufficient footage','shorter','shortfall'))]
    if frames<target_frames:limitations.append(f'Short protected-speech/b-roll coverage cannot fill target: {target_frames-frames} output frames short; raw source duration {total}ms is not the usable-selection duration')
    if not hook(ordered[0][0],ordered[0][2]):limitations.append('Hook constraint unavailable: no measured people/person, evidence-backed action, or high-motion window found')
    result['coverage']=dict(status='partial' if limitations else 'full',limitations=limitations)
    return result
