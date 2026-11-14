"""Closed descriptive vocabulary and frame-bound semantic provenance; no place names."""
from pathlib import Path
import hashlib

SUBJECTS={
    'night_parade':'night parade', 'parade':'parade', 'market':'market',
    'street_performers':'street performers', 'food_stalls':'food stalls',
    'family_walk':'family walks', 'family_activity':'family activities',
    'scenic_view':'scenic views', 'beach':'beach scenes', 'hiking':'hiking',
    'amusement_ride':'amusement rides', 'boat_ride':'boat rides',
    'train_ride':'train rides', 'fireworks':'fireworks', 'wildlife':'wildlife',
    'sunset':'sunsets', 'city_street':'street scenes',
}
LABELS=['people','action','scenic_wide','interior','screen']


def normalize_tags(data):
    """Canonicalize only closed descriptive aliases; never infer from prose/Goal."""
    from copy import deepcopy
    result=deepcopy(data)
    for segment in result.get('segments',[]):
        tags=segment.get('subject_tags')
        if not isinstance(tags,list) or not all(isinstance(t,str) for t in tags):continue
        labels=list(segment.get('semantic_labels',[]))
        canonical=[]
        for tag in tags:
            value=tag.strip().lower().replace(' ','_').replace('-','_')
            if value in LABELS:
                labels.append(value)
            else:
                # Unknown tags remain invalid, including named places.
                canonical.append(value)
        segment['subject_tags']=list(dict.fromkeys(canonical))
        if labels:segment['semantic_labels']=list(dict.fromkeys(labels))
    return result


def attach(spec,data,rows):
    """Only retained final selections inherit labels from an actually supplied tile."""
    for segment in spec['segments']:
        for decision in data['segments']:
            if decision['source_id']!=segment['source_id']:continue
            if not (decision.get('subject_tags') or decision.get('semantic_labels')):continue
            for row in rows:
                t=row.get('frame_time_ms')
                if (row['source_id']==segment['source_id'] and row.get('frame_in_window') and 'tile_index' in row
                    and type(t) is int and decision['raw_start_ms']<=t<decision['raw_end_ms']
                    and segment['raw_start_ms']<=t<segment['raw_end_ms']
                    and row['start_ms']<=decision['raw_start_ms']<decision['raw_end_ms']<=row['end_ms']
                    and row.get('frame_path') and row.get('frame_sha256')):
                    segment['semantic_evidence']=dict(method='planner_in_window_tile',candidate_id=row['candidate_id'],
                        source_id=row['source_id'],source_start_ms=row['start_ms'],source_end_ms=row['end_ms'],
                        frame_time_ms=t,frame_path=row['frame_path'],frame_sha256=row['frame_sha256'],view=row['view'],
                        subjects=list(decision.get('subject_tags',[])),labels=list(decision.get('semantic_labels',[])))
                    break
            if 'semantic_evidence' in segment:break
    if any(s.get('semantic_evidence') for s in spec['segments']):
        spec['assumptions'].append('Subject tags are Planner inferences from bounded, crop-matched tiles; frame/hash provenance is not independent semantic recognition.')
    return spec


def subjects_in(spec):
    subjects=set();labels=set()
    for s in spec.get('segments',[]):
        e=s.get('semantic_evidence',{});t=e.get('frame_time_ms')
        if (e.get('method')!='planner_in_window_tile' or e.get('source_id')!=s['source_id']
            or type(t) is not int or not s['raw_start_ms']<=t<s['raw_end_ms']):continue
        # A screen depicting an event is not evidence that the trip visited it.
        if 'screen' in e.get('labels',[]):continue
        subjects.update(t for t in e.get('subjects',[]) if t in SUBJECTS)
        labels.update(t for t in e.get('labels',[]) if t in LABELS)
    if 'night_parade' in subjects:subjects.discard('parade')
    return subjects,labels


def intro_v1(spec):
    subjects,labels=subjects_in(spec)
    nouns=[SUBJECTS[s] for s in SUBJECTS if s in subjects]
    if not nouns:
        nouns=['people sharing moments'] if 'people' in labels else []
        if 'scenic_wide' in labels:nouns.append('scenic views')
        if not nouns and 'action' in labels:nouns=['moments in motion']
    if not nouns:return "A few selected moments.\nLet's take a look."
    chosen=[]
    for noun in nouns:
        trial=chosen+[noun]
        joined=', '.join(trial[:-1])+' and '+trial[-1] if len(trial)>1 else trial[0]
        text=joined[0].upper()+joined[1:]+'.\nA few highlights, together.'
        if len(text)<90:chosen=trial
        if len(chosen)==3:break
    joined=', '.join(chosen[:-1])+' and '+chosen[-1] if len(chosen)>1 else chosen[0]
    return joined[0].upper()+joined[1:]+'.\nA few highlights, together.'


def validate_evidence(spec,analysis=None,root=None):
    validate_place(spec,analysis,root)
    sources={s['source_id']:s for s in spec['sources']}
    for s in spec['segments']:
        e=s.get('semantic_evidence')
        if not e:continue
        t=e['frame_time_ms']
        if (e['source_id']!=s['source_id'] or not s['raw_start_ms']<=t<s['raw_end_ms']
            or not 0<=e['source_start_ms']<=t<e['source_end_ms']<=sources[s['source_id']]['duration_ms']
            or not any(f['path']==e['frame_path'] and f['source_time_ms']==t for f in sources[s['source_id']]['keyframes'])):
            raise ValueError('Semantic subject lacks retained source-frame evidence')
        if e['view']!=('center_crop_9_16' if spec['aspect_ratio']=='9:16' else 'fit_16_9'):
            raise ValueError('Semantic evidence view differs from output framing')
        if root is not None:
            root=Path(root).resolve();path=(root/e['frame_path']).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=e['frame_sha256']:
                raise ValueError('Semantic evidence frame hash/path invalid')


DISTINCTIVE_ORDER=('night_parade','parade','street_performers','market','fireworks','wildlife',
                  'amusement_ride','boat_ride','train_ride','food_stalls','hiking','beach','sunset')


def intro_v2(spec):
    """Historical writer: name retained distinctive evidence before generic family/scenic labels."""
    subjects,_=subjects_in(spec)
    distinctive=[SUBJECTS[s] for s in DISTINCTIVE_ORDER if s in subjects]
    if not distinctive:
        # Avoid redundant "family walks and family activities" when both exist.
        if 'family_walk' in subjects and 'family_activity' in subjects:
            from copy import deepcopy
            spec=deepcopy(spec)
            for segment in spec['segments']:
                evidence=segment.get('semantic_evidence')
                if evidence:evidence['subjects']=[s for s in evidence['subjects'] if s!='family_activity']
        return intro_v1(spec)
    chosen=[]
    for noun in distinctive:
        trial=chosen+[noun]
        joined=', '.join(trial[:-1])+' and '+trial[-1] if len(trial)>1 else trial[0]
        text=joined.capitalize()+'.\nThe moments that stood out.'
        if len(text)<90:chosen=trial
        if len(chosen)==3:break
    joined=', '.join(chosen[:-1])+' and '+chosen[-1] if len(chosen)>1 else chosen[0]
    return joined.capitalize()+'.\nThe moments that stood out.'


def intro_v3(spec):
    """Historical writer: one complete sentence, at most two retained evidenced subjects; no fragments."""
    subjects,labels=subjects_in(spec)
    if 'family_walk' in subjects:subjects.discard('family_activity')
    tags=[s for s in DISTINCTIVE_ORDER if s in subjects][:2]
    if not tags:tags=[s for s in SUBJECTS if s in subjects][:2]
    nouns=[('the ' if s in ('night_parade','parade','market') else '')+SUBJECTS[s] for s in tags]
    if not nouns:
        nouns=(['people'] if 'people' in labels else [])+(['scenic views'] if 'scenic_wide' in labels else [])
        if not nouns and 'action' in labels:nouns=['moments in motion']
    if not nouns:return 'Here are a few selected moments.'
    if len(nouns)==2:
        text=f'From {nouns[0]} to {nouns[1]}, these moments stood out.'
        if len(text)<90:return text
    return f'These selected moments highlight {nouns[0]}.'


PLACE_ASSERTIONS=(("we're","in","arizona"),("we","are","in","arizona"),("welcome","to","arizona"))


def _retained_place(spec,words):
    """Conservative closed place support: explicit retained ASR assertion, not Goal."""
    import re
    for segment in spec.get('segments',[]):
        if 'screen' in segment.get('semantic_evidence',{}).get('labels',[]):continue
        source_words=sorted([w for w in words if w['source_id']==segment['source_id']],key=lambda w:w['start_ms'])
        tokens=[re.sub(r"[^a-z']",'',w['text'].lower()) for w in source_words]
        for i in range(len(tokens)):
            for pattern in PLACE_ASSERTIONS:
                selected=source_words[i:i+len(pattern)]
                if tuple(tokens[i:i+len(pattern)])!=pattern:continue
                if i and not source_words[i-1]['text'].rstrip().endswith(('.', '!', '?')):continue
                if set(tokens[max(0,i-4):i]) & {'not','never','no','if','wish','maybe','perhaps','say','says','said','tv','screen','pretend',
                    "don't",'dont',"isn't",'isnt',"wasn't",'wasnt',"can't",'cant',"won't",'wont','quote','imagine','would','could','should','hope'}:continue
                if not (segment['raw_start_ms']<=selected[0]['start_ms'] and selected[-1]['end_ms']<=segment['raw_end_ms']):continue
                if any(b['start_ms']-a['end_ms']>800 for a,b in zip(selected,selected[1:])):continue
                return dict(name='Arizona',method='retained_transcript_assertion',source_id=segment['source_id'],
                    start_ms=selected[0]['start_ms'],end_ms=selected[-1]['end_ms'],assertion=' '.join(pattern))
    return None


def attach_place(spec,root):
    from .contracts import read_json
    path=Path(root)/'transcript.json'
    words=read_json(path).get('words',[]) if path.is_file() else []
    spec.pop('narration_place',None)
    place=_retained_place(spec,words)
    if place:spec['narration_place']=place


def validate_place(spec,analysis=None,root=None):
    place=spec.get('narration_place')
    if not place:return
    if analysis is None and root is None:
        if not any(s['source_id']==place['source_id'] and s['raw_start_ms']<=place['start_ms']<place['end_ms']<=s['raw_end_ms'] for s in spec['segments']):
            raise ValueError('Place assertion is outside selected source range')
        return  # Full transcript binding is checked with analysis/root before TTS and by evaluation.
    from .contracts import read_json
    words=read_json(Path(root)/'transcript.json')['words'] if root is not None else (analysis or {}).get('words',[])
    if place!=_retained_place(spec,words):raise ValueError('Place lacks retained positive transcript assertion')


def intro(spec):
    """Compact, warm teaser hook; two subjects only when both survive as evidence."""
    subjects,labels=subjects_in(spec)
    if 'family_walk' in subjects:subjects.discard('family_activity')
    tags=[s for s in DISTINCTIVE_ORDER if s in subjects][:2]
    if not tags:tags=[s for s in SUBJECTS if s in subjects][:2]
    nouns=[SUBJECTS[s] for s in tags]
    if not nouns:
        nouns=(['people'] if 'people' in labels else [])+(['scenic views'] if 'scenic_wide' in labels else [])
    place=spec.get('narration_place')
    located=bool(place and place.get('name')=='Arizona' and place.get('method')=='retained_transcript_assertion')
    if not nouns:
        return 'Come explore Arizona with us!' if located else 'Come along for these moments!'
    joined=' and '.join(nouns[:2])
    if located:
        text=f"Come explore Arizona's {joined}!"
        return text if len(text.split())<=8 else text.removeprefix('Come ')
    article='the ' if tags and tags[0] in ('night_parade','parade','market') else ''
    return f'Come explore {article}{joined}'+(' with us!' if len(nouns)==1 else '!')
