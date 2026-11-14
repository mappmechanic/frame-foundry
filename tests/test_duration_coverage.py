"""Live az-0550 regression: eight-window evidence is not 263s footage availability."""
from copy import deepcopy
import json
import pytest
from pipelines.ffmpeg import teaser_edit
from pipelines.ffmpeg.goal_parser import parse_goal
from pipelines.ffmpeg.planner import plan_teaser
from tests.test_short_speech_edit import analysis
from tests.test_visual_duration import context,FakeProvider,response_for

@pytest.mark.parametrize('claim',[
    'The total available footage is only about 15 seconds',
    'Only 15s of usable footage available',
    'The eight clips provide just 15 seconds of material',
    'Footage is less than the requested duration',
    'Selections are shorter than the requested target',
])
def test_live_shortlist_duration_claim_replaced_with_local_totals(claim):
    a=analysis();pool=a['candidates'];original=deepcopy(a)
    result=teaser_edit.complete(a,parse_goal(a['goal']),dict(segments=[],assumptions=[claim,'No invented locations'],coverage=dict(status='partial',limitations=[claim,'Birthday cake not verified','ASR unavailable for one source'])),pool)
    assert claim not in result['coverage']['limitations'] and claim not in result['assumptions']
    assert 'Birthday cake not verified' in result['coverage']['limitations']
    assert 'ASR unavailable for one source' in result['coverage']['limitations']
    evidence=teaser_edit.duration_evidence(a,pool,result['segments'])
    assert evidence['source_duration_ms']==80000 and abs(evidence['selected_encoded_duration_ms']-45000)<=150
    assert str(evidence['usable_short_windows_ms']) in ' '.join(result['assumptions'])
    assert a==original


def test_usable_is_source_bounded_union_not_candidate_sum():
    a=dict(sources=[dict(source_id='a',duration_ms=10000),dict(source_id='b',duration_ms=5000)])
    candidates=[dict(source_id='a',start_ms=0,end_ms=8000),dict(source_id='a',start_ms=3000,end_ms=12000),dict(source_id='a',start_ms=0,end_ms=8000),dict(source_id='b',start_ms=1000,end_ms=2000)]
    result=teaser_edit.duration_evidence(a,candidates)
    assert result['source_duration_ms']==15000 and result['usable_short_windows_ms']==11000
    assert result['selected_encoded_duration_ms']==0


def test_real_shortfall_is_recomputed_not_hidden():
    a=analysis();a['goal']='90 second family-trip teaser'
    result=teaser_edit.complete(a,parse_goal(a['goal']),dict(segments=[],assumptions=[],coverage=dict(status='partial',limitations=['Only 15 seconds footage available'])),a['candidates'])
    assert any('cannot fill target' in s and '80000ms' in s for s in result['coverage']['limitations'])
    assert not any('15 seconds' in s for s in result['coverage']['limitations'])


def test_planner_prompt_uses_full_pool_and_reconciles_model_prose(tmp_path):
    a=analysis()
    class SuppliedEvidenceProvider(FakeProvider):
        def complete_json(self,**kwargs):
            rows=json.loads(kwargs['prompt'])['candidates']
            self.data=response_for(rows,6000)
            self.data['coverage']=dict(status='partial',limitations=['total available footage is only about 15 seconds','Missing birthday cake'])
            return super().complete_json(**kwargs)
    provider=SuppliedEvidenceProvider(None);spec=plan_teaser(context(tmp_path,a),a,provider=provider)
    assert spec['planner_mode']=='llm'
    payload=json.loads(provider.calls[0]['prompt']);e=payload['duration_evidence']
    assert len(payload['candidates'])==8
    assert e['source_count']==4 and e['source_duration_ms']==80000
    assert e['usable_short_windows_ms']>e['shortlist_duration_ms']
    assert 'Missing birthday cake' in spec['coverage']['limitations']
    assert not any('15 seconds' in s for s in spec['coverage']['limitations'])


def test_cli_reports_263_seconds_not_the_shortlist(tmp_path,monkeypatch,capsys):
    from pipelines.ffmpeg import cli
    root=tmp_path/'e';root.mkdir()
    sources=[dict(source_id=f's-{i}',duration_ms=10000 if i<25 else 13282) for i in range(26)]
    a=dict(sources=sources,visual_cues={s['source_id']:[] for s in sources},words=[],candidates=[])
    segments=[dict(source_id=s['source_id'],raw_start_ms=0,raw_end_ms=2500) for s in sources[:18]]
    (root/'analysis.json').write_text(json.dumps(a));(root/'narrative-spec.json').write_text(json.dumps(dict(segments=segments)))
    (root/'eval-report.json').write_text(json.dumps(dict(scores=dict(G=1,J=.792,Q=.808,D=.835),model_cost=dict(quality_per_dollar=35.6))))
    report=dict(episode_id='e',passed=True,result='partial',error=None,output_path='finished.mp4',ai_usage=[],cost={},goal_judge={},actual_duration_ms=45000,coverage=dict(status='partial',limitations=['Birthday cake not verified']))
    monkeypatch.setattr(cli,'run_episode',lambda *a,**kw:report)
    assert cli._run_main(['--footage-dir',str(tmp_path),'--goal','teaser','--episodes-root',str(tmp_path)])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary['duration_evidence']['source_duration_ms']==263282
    assert summary['duration_evidence']['source_count']==26
    assert summary['duration_evidence']['selected_encoded_duration_ms']==45000
    assert '263.282s raw across 26 clips' in summary['coverage_note']
    assert '45.000s selected output' in summary['coverage_note']
