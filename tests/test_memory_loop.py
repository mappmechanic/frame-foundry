"""Local deterministic lesson contracts; no live clients or paid transports."""
import json
from pathlib import Path
import pytest
from pipelines.ffmpeg import memory_loop as m
from pipelines.ffmpeg.contracts import EpisodeContext, atomic_write_json, read_json
from pipelines.ffmpeg.memory_store import LocalJsonlStore

FH='a'*64

def evidence(eid='one', evaluation='evaluation-one', J=.75, Q=.8):
    spec=dict(target_duration_ms=10000,aspect_ratio='9:16',segments=[
        dict(segment_id='a',source_id='s',raw_start_ms=0,raw_end_ms=1000),
        dict(segment_id='b',source_id='s',raw_start_ms=1000,raw_end_ms=2000),
        dict(segment_id='c',source_id='s',raw_start_ms=6000,raw_end_ms=7000)])
    spec['segments'][0]['semantic_evidence']=dict(method='planner_in_window_tile',source_id='s',frame_time_ms=500,labels=['people'],subjects=[])
    report=dict(episode_id=eid,evaluation_id=evaluation,goal='family trip',footage_hash=FH,scores=dict(J=J,Q=Q,method_version='scores'),
                judge=dict(status='scored' if J is not None else 'unavailable',scores=dict(opening=4),weakest_shot='shot-02',model='mock',provider='mock',evidence={'opening':['shot-01']}),model_cost=dict(usd=.01,known_usd=.01,status='estimated'))
    shots=[dict(shot_id='shot-01',source_id='s',source_start_ms=0,source_end_ms=2000,selection_ids=['a','b']),
           dict(shot_id='shot-02',source_id='s',source_start_ms=6000,source_end_ms=7000,selection_ids=['c'])]
    return report,dict(passed=True),spec,shots

def lesson(eid='one', evaluation='evaluation-one'):
    report,qa,spec,shots=evidence(eid,evaluation)
    return m.build_lessons(report,qa,spec,shots=shots)[0]

def test_actual_otio_identity_not_segment_index_and_coalescing():
    r,q,s,shots=evidence()
    assert m.source_window(s,'shot-02',shots)==dict(source_id='s',start_ms=6000,end_ms=7000)
    assert m.source_window(s,'shot-01',shots)==dict(source_id='s',start_ms=0,end_ms=2000)
    assert m.source_window(s,'shot-02') is None
    shots[1]['selection_ids']=['missing']
    assert m.source_window(s,'shot-02',shots) is None

def test_observational_pair_unknown_scores_and_idempotent(tmp_path):
    prior=lesson();r,q,s,shots=evidence('two','evaluation-two',J=None,Q=None)
    rows=m.build_lessons(r,q,s,prior,shots)
    assert len(rows)==2 and rows[1]['metadata']['outcome']=='unknown'
    assert rows[1]['metadata']['deltas']['Q'] is None
    assert 'source_id' not in rows[0]['metadata']
    assert rows[0]['metadata']['J'] is None and rows[0]['metadata']['Q'] is None
    store=LocalJsonlStore(tmp_path/'lessons.jsonl')
    assert store.put_many(rows)['status']=='stored'
    store.put_many(rows)
    assert len(store.query('family trip',footage_hash=FH))==2
    assert 'not causal' in rows[1]['text']
    r,q,s,shots=evidence('two','evaluation-two',Q=.9)
    delta=m.build_lessons(r,q,s,prior,shots)[1]
    assert delta['metadata']['outcome']=='unknown' and delta['metadata']['deltas']['Q']>0

def test_bounded_serialized_untrusted_data_and_repeated_exact_footage():
    rows=[lesson('one','one'),lesson('two','two')]
    chosen=m.bounded_selection(rows*4)
    assert 0<len(chosen)<=5
    assert len(json.dumps({'edit_memory':m.prompt_data(chosen)},ensure_ascii=False,separators=(',',':')).encode())<=400
    c=m.constraints(rows,FH)
    assert c['blocked']==[dict(source_id='s',start_ms=6000,end_ms=7000)]
    assert c['opening_types']==['people']
    assert not m.constraints(rows,'b'*64)['blocked']
    assert not m.constraints([rows[0],rows[0]],FH)['blocked']
    assert m.blocked_window(c,'s',6500,6600)
    assert not m.blocked_window(c,'s',2000,3000)
    assert not m.blocked_window(c,'other',6500,6600)
    assert m.bounded_selection([dict(rows[0],text='🌵'*400)])==[]

def test_memory_off_never_selects_store(tmp_path,monkeypatch):
    monkeypatch.setattr('pipelines.ffmpeg.memory_store.select_store',lambda *a,**k:pytest.fail('off arm accessed memory'))
    ctx=EpisodeContext('off',tmp_path,'family trip')
    assert m.prepare(ctx,{}) is None
    m.write_completed({},tmp_path)
    assert not (tmp_path/'memory-report.json').exists()

def test_memory_unavailable_is_recorded_not_fatal(tmp_path,monkeypatch):
    monkeypatch.setattr('pipelines.ffmpeg.evaluator.footage_fingerprint',lambda x:FH)
    monkeypatch.setattr('pipelines.ffmpeg.memory_store.select_store',lambda *a,**k:(_ for _ in ()).throw(TimeoutError()))
    ctx=EpisodeContext('on',tmp_path/'on','family trip',memory_enabled=True,memory_root=tmp_path/'private')
    result=m.prepare(ctx,dict(sources=[],candidates=[]))
    assert result['status']=='unavailable' and result['lessons_used']==[]
    assert result['storage_root']==str(tmp_path/'private') and result['limitations']
    assert read_json(ctx.episode_dir/'memory-report.json')['enabled']

def test_preferences_only_evidenced_types(tmp_path):
    r,q,s,shots=evidence()
    s.update(coverage=dict(status='full',limitations=[]),assumptions=[])
    ctx=EpisodeContext('on',tmp_path,'family trip',memory_enabled=True)
    ctx.memory_selection=dict(lessons=[],constraints=dict(blocked=[],opening_types=['night_parade']),limitations=[])
    result=m.apply_preferences(ctx,s)
    assert result['segments']==s['segments'] and result['coverage']['status']=='partial'
    assert 'no scene invented' in ' '.join(result['coverage']['limitations'])


def test_delta_canonical_per_evaluation_and_actual_decider_changes():
    r,q,s,shots=evidence('three','same-evaluation')
    first=m.build_lessons(r,q,s,lesson('one','one'),shots,decider='heuristic')[1]
    second=m.build_lessons(r,q,s,lesson('two','two'),shots,decider='heuristic')[1]
    assert first['id']==second['id']
    assert first['metadata']['paired_episode_id']=='one'
    assert 'decider' in first['metadata']['changes']
    assert first['metadata']['outcome']=='unknown'


def test_shot_mapping_rejects_same_source_but_wrong_saved_interval():
    r,q,s,shots=evidence()
    shots[1]['source_start_ms']=1000
    assert m.source_window(s,'shot-02',shots) is None


def test_safe_opening_preference_cannot_move_closing_or_invent_labels(tmp_path):
    r,q,s,shots=evidence()
    s['segments'].append(dict(segment_id='d',source_id='s',raw_start_ms=8000,raw_end_ms=9000))
    s['segments'][0].pop('semantic_evidence')
    s['segments'][1]['semantic_evidence']=dict(method='planner_in_window_tile',source_id='s',frame_time_ms=1500,labels=['people'],subjects=[])
    s.update(coverage=dict(status='full',limitations=[]),assumptions=[])
    ctx=EpisodeContext('on',tmp_path,'family trip',memory_enabled=True)
    ctx.memory_selection=dict(lessons=[],constraints=dict(blocked=[],opening_types=['people']),limitations=[])
    # Moving b before a would create an a->c noncoalescing chain; current a,b touch
    # but new b,a don't, so this safe preference preserves the exact last two.
    result=m.apply_preferences(ctx,s)
    assert result['segments'][0]['segment_id']=='b'
    assert result['segments'][-2:]==s['segments'][-2:]
    assert {x['segment_id']:(x['raw_start_ms'],x['raw_end_ms']) for x in result['segments']}=={x['segment_id']:(x['raw_start_ms'],x['raw_end_ms']) for x in s['segments']}
