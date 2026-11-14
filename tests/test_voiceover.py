"""No ElevenLabs calls: API binary shape mocked, real FFmpeg checks the mix."""
from pathlib import Path
from decimal import Decimal
import json,subprocess
import pytest
from pipelines.ffmpeg import voiceover as vo, llm_provider as account
from pipelines.ffmpeg.contracts import EpisodeContext,StageError,read_json,usage_summary
from tests.test_render import inputs,loudness
from pipelines.ffmpeg.assembler import assemble

@pytest.fixture
def audio(tmp_path):
    path=tmp_path/'tts.mp3'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','sine=frequency=770:sample_rate=44100:duration=1.2','-c:a','libmp3lame',str(path)],check=True,capture_output=True)
    return path.read_bytes()


def prepare(tmp_path,audio,monkeypatch):
    monkeypatch.setenv('ELEVENLABS_API_KEY','offline-fake-secret')
    spec,analysis=inputs(tmp_path,'9:16');calls=[]
    def send(url,headers,body,timeout):
        calls.append((url,headers,body));return 200,audio
    intro=vo.prepare_intro(EpisodeContext('episode',tmp_path,'teaser'),spec,transport=send)
    return spec,analysis,intro,calls


def test_intro_cache_and_cost_are_shared(tmp_path,audio,monkeypatch):
    spec,_,intro,calls=prepare(tmp_path,audio,monkeypatch)
    assert len(calls)==1 and intro['start_ms']==0 and '\n' in intro['text']
    assert json.loads(calls[0][2])['model_id']=='eleven_flash_v2_5'
    row=read_json(tmp_path/'tts-usage.json')['calls'][0]
    assert row['status']=='success' and row['cost_status']=='estimated' and row['cost_usd']>0
    assert usage_summary(tmp_path)[1]['known_ai_usd']==row['cost_usd']
    assert 'offline-fake-secret' not in (tmp_path/'tts-usage.json').read_text()
    spec['voiceover']=intro
    assert vo.prepare_intro(EpisodeContext('episode',tmp_path,'teaser'),spec,transport=lambda *a:pytest.fail('No repeated dispatch'))['cached']
    other=tmp_path/'other';other.mkdir()
    assert vo.prepare_intro(EpisodeContext('other',other,'teaser'),dict(spec,voiceover=None),transport=lambda *a:pytest.fail('Cache hit'))['cached']
    assert not (other/'tts-usage.json').exists()
    monkeypatch.setenv(account.BUDGET_ENV,'0.00001')
    provider=account.LLMProvider('openai','fake','gpt-4.1',(Decimal(2),Decimal(8)))
    with pytest.raises(account.ProviderBudgetError):provider.complete_json(system='x',prompt='x',images=[],role='planner',episode_dir=tmp_path)


def test_budget_blocks_tts_before_dispatch(tmp_path,audio,monkeypatch):
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake');monkeypatch.setenv(account.BUDGET_ENV,'0.00001')
    spec,_=inputs(tmp_path,'9:16')
    with pytest.raises(account.ProviderBudgetError):vo.prepare_intro(EpisodeContext('episode',tmp_path,'teaser'),spec,transport=lambda *a:pytest.fail('No dispatch'))
    assert read_json(tmp_path/'tts-usage.json')['calls'][0]['status']=='blocked'


def test_failed_transport_keeps_unknown_reservation(tmp_path,monkeypatch):
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake')
    spec,_=inputs(tmp_path,'9:16')
    def failed(*a):raise RuntimeError('secret must not leak')
    with pytest.raises(StageError):vo.prepare_intro(EpisodeContext('episode',tmp_path,'teaser'),spec,transport=failed)
    row=read_json(tmp_path/'tts-usage.json')['calls'][0]
    assert row['status']=='failed' and row['cost_usd'] is None and row['reserved_usd']>0


def test_real_render_vo_stays_in_budget_and_normalizes(tmp_path,audio,monkeypatch):
    spec,analysis,intro,_=prepare(tmp_path,audio,monkeypatch);spec['voiceover']=intro
    result=assemble(EpisodeContext('episode',tmp_path,'teaser'),spec,analysis)
    stats=loudness(result.candidate_path)
    assert -15<=float(stats['input_i'])<=-13 and float(stats['input_tp'])<=-1
    import opentimelineio as otio
    timeline=otio.adapters.read_from_file(str(result.timeline_path))
    assert dict(timeline.metadata['voiceover'])==intro
    assert result.duration_ms==pytest.approx(2000,abs=80)


def test_saved_voice_hash_cannot_change_between_prepare_and_mix(tmp_path,audio,monkeypatch):
    spec,analysis,intro,_=prepare(tmp_path,audio,monkeypatch);spec['voiceover']=intro
    (tmp_path/intro['path']).write_bytes(b'tampered')
    with pytest.raises(StageError) as error:assemble(EpisodeContext('episode',tmp_path,'teaser'),spec,analysis)
    assert error.value.code=='VOICEOVER_INVALID'


def test_missing_key_preflight_does_not_start_refinery(tmp_path,monkeypatch):
    from pipelines.ffmpeg import cli
    monkeypatch.setattr(cli,'run_refinery',lambda *a,**k:pytest.fail('No paid source work'))
    report=cli.run_episode(tmp_path,'teaser',episodes_root=tmp_path/'episodes',episode_id='missing',provider=None,voiceover=True)
    assert report['error']['code']=='VOICEOVER_CONFIG' and not report['ai_usage']


def test_end_to_end_vo_cost_and_independent_gate(tmp_path,audio,monkeypatch):
    from tests.fixtures.generate_footage import generate_footage
    from pipelines.ffmpeg.cli import run_episode
    footage=tmp_path/'footage';generate_footage(footage)
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake');calls=[]
    def send(*args):calls.append(args);return 200,audio
    monkeypatch.setattr(account,'_transport',send)
    report=run_episode(footage,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='vo',provider=None,transcript_fixture=footage/'transcript-fixture.json',voiceover=True)
    assert report['passed'] and len(calls)==1 and report['cost']['known_ai_usd']>0
    root=tmp_path/'episodes'/'vo';evaluation=read_json(root/'eval-report.json')
    gate=next(c for c in evaluation['gates']['checks'] if c['name']=='voiceover_integrity')
    assert gate['status']=='pass'
    assert evaluation['model_cost']['tts_usd']==report['cost']['known_ai_usd']
    from pipelines.ffmpeg.eval_gates import _voiceover_check
    spec=read_json(root/'narrative-spec.json');spec['voiceover']['path']='../../outside.mp3'
    checks={};_voiceover_check(root,spec,{'duration_ms':10000},checks)
    assert checks['voiceover_integrity']['status']=='fail'


@pytest.mark.parametrize('g,expected',[(1,'PASS'),(0,'QA_PASS_EVAL_INCOMPLETE')])
def test_cli_partial_summary_explains_coverage(tmp_path,monkeypatch,capsys,g,expected):
    from pipelines.ffmpeg import cli
    report=dict(episode_id='e',passed=True,result='partial',error=None,output_path='finished.mp4',ai_usage=[],cost={},goal_judge={},coverage=dict(status='partial',limitations=['Missing requested moment']))
    monkeypatch.setattr(cli,'run_episode',lambda *a,**kw:report)
    episode=tmp_path/'e';episode.mkdir();(episode/'eval-report.json').write_text(json.dumps(dict(scores=dict(Q=None,J=None,D=.8,G=g),model_cost=dict(quality_per_dollar=None))))
    assert cli._run_main(['--footage-dir',str(tmp_path),'--goal','teaser','--episodes-root',str(tmp_path)])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary['status']==expected and 'Missing requested moment' in summary['coverage_note']


def test_voice_audio_cannot_be_truncated_by_short_timeline(tmp_path,audio,monkeypatch):
    _,_,intro,_=prepare(tmp_path,audio,monkeypatch)
    with pytest.raises(StageError):vo.mix_intro(tmp_path/'unused.mov',intro,tmp_path,.5)


def test_repair_reuses_episode_audio_if_global_cache_write_failed(tmp_path,audio,monkeypatch):
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake')
    cache=tmp_path/'cache-blocker';cache.write_text('not a directory')
    monkeypatch.setenv('FRAMEFOUNDRY_CACHE_DIR',str(cache))
    spec,_=inputs(tmp_path,'9:16');ctx=EpisodeContext('episode',tmp_path,'teaser')
    first=vo.prepare_intro(ctx,spec,transport=lambda *a:(200,audio))
    # Newly planned spec has no VO metadata, but the durable success proves identity.
    second=vo.prepare_intro(ctx,spec,transport=lambda *a:pytest.fail('No retry'))
    assert second['cached'] and first['sha256']==second['sha256']


@pytest.mark.parametrize('name,value',[('ELEVENLABS_VOICE_ID','../../evil'),('ELEVENLABS_MODEL_ID','unpriced'),('FRAMEFOUNDRY_TTS_USD_PER_1000_CHARS','NaN')])
def test_invalid_config_fails_closed(monkeypatch,name,value):
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake');monkeypatch.setenv(name,value)
    with pytest.raises(StageError):vo.configuration()


def test_independent_vo_failure_cannot_advertise_delivery(tmp_path,audio,monkeypatch):
    from tests.fixtures.generate_footage import generate_footage
    from pipelines.ffmpeg.cli import run_episode
    from pipelines.ffmpeg import evaluator
    footage=tmp_path/'footage';generate_footage(footage)
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake')
    monkeypatch.setattr(account,'_transport',lambda *args:(200,audio))
    original=evaluator.evaluate_episode
    def tamper(root,**kwargs):
        spec=read_json(root/'narrative-spec.json')
        spec['voiceover']['sha256']='0'*64
        (root/'narrative-spec.json').write_text(json.dumps(spec))
        return original(root,**kwargs)
    monkeypatch.setattr(evaluator,'evaluate_episode',tamper)
    report=run_episode(footage,'10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='tampered',provider=None,transcript_fixture=footage/'transcript-fixture.json',voiceover=True)
    assert not report['passed'] and report['output_path'] is None
    assert report['error']['code']=='EVAL_VOICEOVER_FAILED'
    assert (tmp_path/'episodes'/'tampered'/'diagnostic.mp4').is_file()


def test_original_generic_intro_still_evaluates_as_legacy(tmp_path,audio,monkeypatch):
    from pipelines.ffmpeg import eval_gates
    from pipelines.ffmpeg.assembler import build_timeline
    import opentimelineio as otio
    spec,analysis,intro,_=prepare(tmp_path,audio,monkeypatch)
    intro.pop('grounding_version');intro['text']='A few moments from our family trip.\nShared memories, together.'
    spec['voiceover']=intro
    timeline,_=build_timeline(spec,analysis)
    otio.adapters.write_to_file(timeline,str(tmp_path/'timeline.otio'))
    checks={};eval_gates._voiceover_check(tmp_path,spec,dict(duration_ms=2000),checks)
    assert checks['voiceover_integrity']['status']=='pass'
    spec['voiceover']['grounding_version']='subjects-v1'
    timeline,_=build_timeline(spec,analysis);otio.adapters.write_to_file(timeline,str(tmp_path/'timeline.otio'))
    checks={};eval_gates._voiceover_check(tmp_path,spec,dict(duration_ms=2000),checks)
    assert checks['voiceover_integrity']['status']=='fail'


def test_historical_subjects_v1_still_valid_after_distinctive_writer(tmp_path,audio,monkeypatch):
    from pipelines.ffmpeg import eval_gates,semantics
    from pipelines.ffmpeg.assembler import build_timeline
    import opentimelineio as otio
    spec,analysis,intro,_=prepare(tmp_path,audio,monkeypatch)
    intro['grounding_version']='subjects-v1';intro['text']=semantics.intro_v1(spec)
    spec['voiceover']=intro
    timeline,_=build_timeline(spec,analysis)
    otio.adapters.write_to_file(timeline,str(tmp_path/'timeline.otio'))
    checks={};eval_gates._voiceover_check(tmp_path,spec,dict(duration_ms=2000),checks)
    assert checks['voiceover_integrity']['status']=='pass'
