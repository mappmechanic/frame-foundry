"""Offline end-to-end proofs with explicit synthetic transcripts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import copy,json,shutil,subprocess
import pytest
from PIL import Image
from pipelines.ffmpeg.contracts import *
from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.refinery import run_refinery
from pipelines.ffmpeg.proxy import create_proxy,probe
from pipelines.ffmpeg.budget import assert_can_spend,reserve_project_episode
from pipelines.ffmpeg.llm_provider import LLMProvider,ProviderError
from pipelines.ffmpeg.qa_validator import validate_candidate,main as revalidate
from pipelines.ffmpeg.goal_judge import unavailable_judge
from tests.fixtures.generate_footage import generate_footage

@pytest.fixture(scope='module')
def input_set(tmp_path_factory):
    path=tmp_path_factory.mktemp('sources');generate_footage(path);return path

@pytest.fixture(scope='module')
def output_set(tmp_path_factory,input_set):
    root=tmp_path_factory.mktemp('episodes')
    report=run_episode(input_set,'Create a 10 second family-trip teaser',episodes_root=root,episode_id='offline',transcript_fixture=input_set/'transcript-fixture.json',provider=None)
    assert report['passed'];return root/'offline',report

def test_artifacts_hard_gate_and_accounting(output_set):
    root,report=output_set
    for name in ['finished.mp4','transcript.json','narrative-spec.json','timeline.otio','contact-sheet.jpg','qa-report.json']:
        assert (root/name).exists() and (root/name).stat().st_size
    assert -15<=report['integrated_lufs']<=-13 and report['true_peak_dbtp']<=-1
    assert report['fixture_mode'] and report['ai_usage']==[] and report['goal_judge']['score'] is None
    assert report['cost']['usd'] is None
    assert revalidate(['--episode-dir',str(root)])==0

def test_report_schema_rejects_forged_success(output_set):
    _,report=output_set;bad=copy.deepcopy(report);bad['integrated_lufs']=-30
    with pytest.raises(StageError):validate_report(bad)
    bad=copy.deepcopy(report);bad['goal_judge']['status']='scored';bad['goal_judge']['score']=None
    with pytest.raises(StageError):validate_report(bad)
    bad=copy.deepcopy(report);bad['invented']=True
    with pytest.raises(StageError):validate_report(bad)

def test_ambiguous_fallback_episode_identity(input_set,tmp_path):
    report=run_episode(input_set,'teaser',episodes_root=tmp_path,episode_id='one',transcript_fixture=input_set/'transcript-fixture.json',provider=None)
    assert not report['passed'] and report['error']['code']=='EVAL_DURATION_FAILED'
    assert (tmp_path/'one'/'diagnostic.mp4').exists()
    assert read_json(tmp_path/'one'/'eval-report.json')['scores']['Q']==0
    with pytest.raises(StageError):run_episode(input_set,'teaser',episodes_root=tmp_path,episode_id='one',provider=None)
    with pytest.raises(StageError):resolve_episode(tmp_path,'../../escape')
    with pytest.raises(ValueError):atomic_write_json(tmp_path/'nonfinite.json',dict(x=float('nan')))

def test_early_failure_report_missing_models(input_set,tmp_path):
    report=run_episode(input_set,'teaser',episodes_root=tmp_path,episode_id='missing',provider=None)
    assert not report['passed'] and report['error']['code']=='DEPENDENCY_MISSING'
    assert report['output_path'] is None and report['ai_usage']==[];validate_report(report)

def test_silent_audio_is_not_passed_or_scored(tmp_path,monkeypatch):
    inputs=tmp_path/'inputs';generate_footage(inputs,profile='silent')
    report=run_episode(inputs,'5 second teaser',episodes_root=tmp_path/'episodes',episode_id='silent',transcript_fixture=inputs/'transcript-fixture.json',provider=None)
    assert not report['passed'] and report['integrated_lufs'] is None
    assert len(report['attempts'])==1 and report['goal_judge']['score'] is None

def test_low_judge_score_flags_partial_but_never_overrides_audio(output_set,monkeypatch):
    root,previous=output_set;spec=read_json(root/'narrative-spec.json')
    import opentimelineio as otio
    timeline=otio.adapters.read_from_file(str(root/'timeline.otio'));track=next(t for t in timeline.tracks if t.name=='V1');cursor=0;cuts=[]
    for clip in track:
        if cursor:cuts.append(round(cursor))
        cursor+=clip.source_range.duration.to_seconds()*1000
    result=AssemblyResult(spec['episode_id'],root/'finished.mp4',root/'timeline.otio',root/'narrative-spec.json',cuts,round(cursor));ctx=EpisodeContext(spec['episode_id'],root,spec['goal'])
    monkeypatch.setattr('pipelines.ffmpeg.qa_validator.judge_goal',lambda *a,**kw:dict(status='scored',score=20,rationale='Limited coverage',evidence=['output frame'],provider='mock',model='mock'))
    report=validate_candidate(ctx,result,spec,judge_provider=object())
    assert report['passed'] and report['result']=='partial' and report['goal_judge']['score']==20
    monkeypatch.setattr('pipelines.ffmpeg.qa_validator.measure_audio',lambda p:dict(integrated_lufs=-40,true_peak_dbtp=-5))
    report=validate_candidate(ctx,result,spec,judge_provider=object())
    assert not report['passed'] and report['goal_judge']['score'] is None

def test_mocked_agent_run_retains_planner_and_judge_tokens(tmp_path,input_set,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(tmp_path/'budget'))
    response_count=[0]
    def transport(url,headers,body,timeout):
        payload=json.loads(body);role='judge' if 'Judge whether' in payload['system'] else 'planner'
        if role=='planner':
            prompt=json.loads(next(c['text'] for c in payload['messages'][0]['content'] if c['type']=='text'))
            data=dict(interpretation='A trip highlight teaser',assumptions=['Sampled frames only'],segments=[dict(source_id=c['source_id'],raw_start_ms=c['start_ms'],raw_end_ms=c['end_ms'],reason='Supplied candidate image') for c in prompt['candidates'][:2]],coverage=dict(status='partial',limitations=['Only compact candidate evidence inspected']))
        else:data=dict(score=75,rationale='Goal mostly covered',evidence=['rendered frame 1'])
        response_count[0]+=1
        return 200,json.dumps(dict(stop_reason='end_turn',content=[dict(type='text',text=json.dumps(data))],usage=dict(input_tokens=1000,output_tokens=100))).encode()
    from decimal import Decimal
    provider=LLMProvider('anthropic','dummy-not-a-real-key','claude-sonnet-4-5',(Decimal(3),Decimal(15)),transport=transport)
    report=run_episode(input_set,'8 second teaser',episodes_root=tmp_path,episode_id='mock-agent',transcript_fixture=input_set/'transcript-fixture.json',provider=provider,planner_retry=False)
    assert report['passed'] and response_count[0]==1
    # Synthetic sources repeat the same image: strict eval gate skips judgment.
    assert read_json(tmp_path/'mock-agent'/'eval-report.json')['scores']['G']==0
    assert read_json(tmp_path/'mock-agent'/'narrative-spec.json')['planner_mode']=='llm'
    assert [c['role'] for c in report['ai_usage']]==['planner']
    assert all(c['input_tokens']==1000 for c in report['ai_usage'])
    assert report['cost']['known_ai_usd']>0 and report['cost']['status']=='unknown'
    assert 'dummy-not-a-real-key' not in json.dumps(report)

def test_project_and_episode_budget_are_closed(tmp_path,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_SPENT_USD','475')
    with pytest.raises(ProviderError):assert_can_spend(None,paid_enabled=True,episode_spent_usd=0,project_spent_usd=0)
    with pytest.raises(ProviderError):assert_can_spend(1,paid_enabled=False,episode_spent_usd=0,project_spent_usd=0)
    with pytest.raises(ProviderError):assert_can_spend(1,paid_enabled=True,episode_spent_usd=5,project_spent_usd=10)
    for i in range(100):reserve_project_episode(tmp_path,f'episode-{i}')
    with pytest.raises(ProviderError):reserve_project_episode(tmp_path,'over-budget')
    reserve_project_episode(tmp_path,'episode-0')

def test_proxy_rotation_and_source_immutability(input_set,tmp_path):
    original=input_set/'01-trip.mp4';rotated=tmp_path/'rotated.mov'
    # New FFmpeg ignores the old output rotate tag; override the input display
    # matrix instead. Keep the real rotation/source-immutability assertions.
    help_result=subprocess.run(['ffmpeg','-hide_banner','-h','full'],capture_output=True,text=True,check=True)
    if '-display_rotation' in help_result.stdout+help_result.stderr:
        command=['ffmpeg','-y','-v','error','-noautorotate','-display_rotation:v:0','90','-i',str(original),'-map','0','-c','copy',str(rotated)]
    else:
        command=['ffmpeg','-y','-v','error','-i',str(original),'-c','copy','-metadata:s:v:0','rotate=90',str(rotated)]
    subprocess.run(command,check=True)
    import hashlib
    before=hashlib.sha256(rotated.read_bytes()).digest();out=tmp_path/'proxy.mp4';mapping=create_proxy(rotated,out)
    assert abs(mapping['rotation_degrees'])==90
    assert probe(out)['video']['height']>probe(out)['video']['width']
    assert max(probe(out)['video']['width'],probe(out)['video']['height'])<=640
    assert hashlib.sha256(rotated.read_bytes()).digest()==before

def test_twenty_six_4k_hevc_clips_create_time_safe_proxies_and_agent_payload(tmp_path):
    inputs=tmp_path/'inputs';generate_footage(inputs,profile='iphone26')
    root=tmp_path/'episode';root.mkdir();ctx=EpisodeContext('hevc26',root,'Create a 10 second trip teaser',fixture_mode=True)
    analysis=run_refinery(ctx,inputs,transcriber=lambda source:[])
    assert len(analysis['sources'])==26 and all(s['width']==3840 and s['height']==2160 for s in analysis['sources'])
    assert all(s['proxy_duration_ms']==1000 for s in analysis['sources'])
    images=[root/f['path'] for s in analysis['sources'] for f in s['keyframes']]
    assert len(images)==78
    for path in images:
        with Image.open(path) as image:assert max(image.size)<=640
    from decimal import Decimal
    from pipelines.ffmpeg.planner import plan_teaser
    calls=[]
    def transport(url,headers,body,timeout):
        calls.append(json.loads(body));data=dict(interpretation='Trip highlights',assumptions=['Sampled frames'],segments=[dict(source_id='source-0001',raw_start_ms=0,raw_end_ms=1000,reason='First source frame')],coverage=dict(status='partial',limitations=['Only one second selected']))
        return 200,json.dumps(dict(stop_reason='end_turn',content=[dict(type='text',text=json.dumps(data))],usage=dict(input_tokens=1000,output_tokens=100))).encode()
    provider=LLMProvider('anthropic','fake','claude-sonnet-4-5',(Decimal(3),Decimal(15)),transport=transport)
    spec=plan_teaser(ctx,analysis,provider=provider)
    assert spec['planner_mode']=='llm' and len(calls)==1

def test_corrupt_usage_preserves_failure_instead_of_fabricated_accounting(tmp_path):
    root=tmp_path/'episode';root.mkdir();(root/'llm-usage.json').write_text('not-json')
    ctx=EpisodeContext('corrupt',root,'teaser');report=failure_report(ctx,StageError('TEST','planner','Test failure'))
    assert report['error']['code']=='USAGE_LEDGER_INVALID' and report['cost']['usd'] is None
    assert not report['passed'];validate_report(report)

def test_schema_cannot_pass_with_missing_hard_checks(output_set):
    root,report=output_set;bad=copy.deepcopy(report);bad['checks']=[]
    with pytest.raises(StageError):validate_report(bad)

def test_corrupt_fixture_mapping_fails_before_fake_transcription(input_set,tmp_path):
    fixture=read_json(input_set/'transcript-fixture.json');fixture['sources'][0]['filename']='wrong.mp4'
    path=tmp_path/'bad-fixture.json';atomic_write_json(path,fixture)
    report=run_episode(input_set,'10 second teaser',episodes_root=tmp_path/'episodes',episode_id='badfixture',transcript_fixture=path,provider=None)
    assert not report['passed'] and report['error']['code']=='FIXTURE_INVALID'

def test_two_repair_attempts_are_bounded(input_set,tmp_path,monkeypatch):
    from pipelines.ffmpeg import cli
    original=cli.validate_candidate;counter=[0]
    def fail_then_pass(*args,**kwargs):
        report=original(*args,**kwargs);counter[0]+=1
        if counter[0]<=2:
            report.update(passed=False,result='failed',output_path=None,error=StageError('QA_FAILED','validator','Injected loudness failure').as_dict(),integrated_lufs=-20)
            for check in report['checks']:
                if check['name']=='loudness':check.update(status='fail',reason='Injected loudness failure')
        return report
    monkeypatch.setattr(cli,'validate_candidate',fail_then_pass)
    report=cli.run_episode(input_set,'10 second teaser',episodes_root=tmp_path,episode_id='repair',transcript_fixture=input_set/'transcript-fixture.json',provider=None)
    assert report['passed'] and len(report['attempts'])==3 and counter[0]==3
    assert [a['passed'] for a in report['attempts']]==[False,False,True]
