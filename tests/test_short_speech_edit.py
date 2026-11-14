"""Source-backed pacing and hook invariants; fake labels are not artistic proof."""
from copy import deepcopy
import json
from pipelines.ffmpeg.teaser_edit import regions,short_candidates,complete,hook
from pipelines.ffmpeg.goal_parser import parse_goal
from pipelines.ffmpeg.assembler import build_timeline
from pipelines.ffmpeg.planner import plan_teaser
from pipelines.ffmpeg.contracts import validate_narrative
from tests.test_visual_duration import analysis_for,context,FakeProvider,response_for


def analysis():
    a=analysis_for('45 second family-trip teaser',count=4,duration=20000)
    a['visual_cues']={s['source_id']:[] for s in a['sources']}
    a['words']=[dict(source_id=s['source_id'],text='long continuous speech',start_ms=1000,end_ms=7000) for s in a['sources']]
    a['transcription_status']='openai_asr'
    base=[dict(candidate_id='base-'+s['source_id'],source_id=s['source_id'],start_ms=0,end_ms=20000,quality=dict(visual_score=50,motion=.4),audio={}) for s in a['sources']]
    for scene in a['scenes']:scene['motion']=.4
    a['candidates']=short_candidates(a,base)
    return a


def test_regions_reject_long_connected_speech_but_include_all_gaps():
    a=analysis();s=a['sources'][0]
    r=regions(s,[w for w in a['words'] if w['source_id']==s['source_id']])
    assert r==[(0,920,'broll'),(7120,20000,'broll')]
    assert all(c['end_ms']-c['start_ms']<=3000 for c in a['candidates'])
    assert all(c['end_ms']<=920 or c['start_ms']>=7120 for c in a['candidates'])


def test_asr_pool_fills_45_with_short_safe_shots_and_honest_evidence(tmp_path):
    a=analysis();provider=FakeProvider(response_for(a['candidates'][:4],6000))
    before=deepcopy(a);spec=plan_teaser(context(tmp_path,a),a,provider=provider)
    timeline,amended=build_timeline(spec,a)
    assert abs(timeline.duration().to_seconds()-45)<.08
    lengths=[clip.duration().to_seconds() for clip in timeline.tracks[0]]
    assert max(lengths)<=3.01 and 2<=sum(lengths)/len(lengths)<=3
    assert len(provider.calls)==1
    payload=json.loads(provider.calls[0]['prompt']);e=payload['duration_evidence']
    assert e['source_duration_ms']==80000 and e['usable_short_windows_ms']>45000
    assert len(payload['candidates'])==8
    assert 'beyond' in ' '.join(spec['assumptions'])
    assert not any('shorter' in l for l in spec['coverage']['limitations'])
    assert a==before
    validate_narrative(spec,a)


def test_hard_hook_promotes_local_people_when_model_chooses_static_first():
    a=analysis();pool=a['candidates']
    for c in pool:c['quality']['motion']=0
    person=pool[-1];person['quality']['face_count']=1
    response=dict(segments=[dict(source_id=pool[0]['source_id'],raw_start_ms=pool[0]['start_ms'],raw_end_ms=pool[0]['end_ms'],reason='static first')],coverage=dict(status='full',limitations=[]),assumptions=[])
    result=complete(a,parse_goal(a['goal']),response,pool)
    assert result['segments'][0]['source_id']==person['source_id']
    assert result['segments'][0]['raw_start_ms']==person['start_ms']
    assert hook(person)


def test_no_qualifying_hook_is_explicit_partial_not_fake_detection():
    a=analysis()
    for c in a['candidates']:c['quality']['motion']=0
    response=dict(segments=[],coverage=dict(status='full',limitations=[]),assumptions=[])
    result=complete(a,parse_goal(a['goal']),response,a['candidates'])
    assert result['coverage']['status']=='partial'
    assert any('Hook constraint unavailable' in l for l in result['coverage']['limitations'])


def test_known_screen_is_not_promoted_merely_for_camera_motion():
    a=analysis();screen=a['candidates'][0];screen['frame_in_window']=True
    screen['quality']['motion']=1.0
    response=dict(segments=[dict(source_id=screen['source_id'],raw_start_ms=screen['start_ms'],raw_end_ms=screen['end_ms'],reason='screen',semantic_labels=['screen'])],coverage=dict(status='full',limitations=[]),assumptions=[])
    result=complete(a,parse_goal(a['goal']),response,a['candidates'])
    assert not any(s['source_id']==screen['source_id'] and s['raw_start_ms']==screen['start_ms'] for s in result['segments'])
