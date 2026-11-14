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


def intro(spec):
    """Name retained distinctive evidence before generic family/scenic labels."""
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
