"""Real offline media gates with mocked Planner/Judge and paired runner."""
from pathlib import Path
import json
import pytest
from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.contracts import read_json
from pipelines.ffmpeg.eval_gates import generate_distinct_footage
from pipelines.ffmpeg.eval_judge import CRITERIA
from pipelines.ffmpeg.memory_store import LocalJsonlStore
from scripts.improve import run_experiment
from tests.test_low_score_retry import offline_provider

@pytest.fixture(scope='module')
def media(tmp_path_factory):
    root=tmp_path_factory.mktemp('memory-distinct')
    return root,generate_distinct_footage(root)


def test_full_mocked_episode_pass_lesson_retrieval_single_planner_call(media,tmp_path):
    provider,calls=offline_provider({c:4 for c in CRITERIA})
    memory=tmp_path/'memory';episodes=tmp_path/'episodes'
    common=dict(episodes_root=episodes,transcript_fixture=media[1],provider=provider,memory=True,memory_backend='local',memory_root=memory,planner_retry=False,reuse_history=False)
    first=run_episode(media[0],'Create a 10 second family-trip teaser',episode_id='first',evals_root=tmp_path/'eval-first',**common)
    assert first['passed'] and [r for r,_ in calls]==['planner','judge']
    ev=read_json(episodes/'first/eval-report.json')
    assert ev['gates']['passed'] and ev['scores']['J'] is not None
    lessons=LocalJsonlStore(memory/'lessons.jsonl').query('family trip',footage_hash=ev['footage_hash'])
    assert len(lessons)==1 and lessons[0]['metadata']['episode_id']=='first'
    assert lessons[0]['metadata']['source_id'] and lessons[0]['metadata']['start_ms']<lessons[0]['metadata']['end_ms']
    calls.clear()
    second=run_episode(media[0],'Create a 10 second family-trip teaser',episode_id='second',evals_root=tmp_path/'eval-second',**common)
    assert second['passed'] and [r for r,_ in calls]==['planner','judge']
    payload=calls[0][1]['edit_memory']
    assert len(json.dumps({'edit_memory':payload},ensure_ascii=False,separators=(',',':')).encode())<=400
    used=read_json(episodes/'second/eval-report.json')['memory']['lessons_used']
    assert used==[lessons[0]['id']]
    assert -15<=second['integrated_lufs']<=-13 and second['true_peak_dbtp']<=-1
    assert (episodes/'second/contact-sheet.jpg').stat().st_size


def test_actual_paired_mocked_run_fresh_provenance_no_arm_leak(media,tmp_path,monkeypatch):
    provider,calls=offline_provider({c:4 for c in CRITERIA})
    def mocked(footage,goal,**kw):
        assert kw['provider'] is None
        kw['provider']=provider  # Only mock transport; no real dispatch.
        return run_episode(footage,goal,**kw)
    c=run_experiment(media[0],'Create a 10 second family-trip teaser',experiment_id='paired',iterations=2,
        episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals',execute=True,runner=mocked,
        recipe={'transcript_fixture':media[1]},budget_root=tmp_path/'budget')
    assert c['status']=='recorded'
    assert all(r['status']=='passed' for rows in c['arms'].values() for r in rows)
    assert all(not r['lessons_used'] for r in c['arms']['memory_off'])
    assert c['arms']['memory_on'][0]['lessons_used']==[]
    assert c['arms']['memory_on'][1]['lessons_used']
    assert all(d['comparable'] and d['Q'] is not None for d in c['deltas'])
    assert all(r['model_usd'] is not None for rows in c['arms'].values() for r in rows)
    for arm,rows in c['arms'].items():
        for r in rows:
            root=tmp_path/'episodes'/r['episode_id']
            assert read_json(root/'analysis.json')['episode_id']==r['episode_id']
            assert read_json(root/'narrative-spec.json')['episode_id']==r['episode_id']
            assert read_json(root/'eval-report.json')['episode_id']==r['episode_id']
            assert (root/'memory-report.json').exists()==(arm=='memory_on')
    assert [role for role,_ in calls].count('planner')==3  # Separate off/on caches; on changes with learned payload.
    assert [role for role,_ in calls].count('judge')==2    # Valid identical-render cache within each arm only.
    assert read_json(tmp_path/'evals/learning-curves/latest.json')==c
