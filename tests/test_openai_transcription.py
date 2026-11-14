"""OpenAI REST is always mocked. Real temporary audio verifies clocks, not ASR accuracy."""
from pathlib import Path
from decimal import Decimal
import io
import os
import json
import wave
import pytest
from pipelines.ffmpeg import llm_provider as accounting
from pipelines.ffmpeg import transcription as asr
from pipelines.ffmpeg.contracts import EpisodeContext, StageError, read_json, usage_summary
from pipelines.ffmpeg.eval_gates import generate_distinct_footage


@pytest.fixture(scope='module')
def media(tmp_path_factory):
    root=tmp_path_factory.mktemp('asr-audio')
    generate_distinct_footage(root)
    paths=sorted(root.glob('*.mp4'))
    assert paths
    return root,paths[0]


@pytest.fixture
def source(media):
    from pipelines.ffmpeg.proxy import probe
    return dict(source_id='original',path=str(media[1]),duration_ms=probe(media[1])['duration_ms'],has_audio=True)


def response(source,**changes):
    data=dict(text='hello family',duration=source['duration_ms']/1000,words=[dict(word='hello',start=.2,end=.6),dict(word='family',start=.9,end=1.4)])
    data.update(changes)
    return data


def provider(source,calls,*,data=None,status=200):
    def send(url,headers,body,timeout):
        calls.append((url,headers,body))
        return status,json.dumps(data if data is not None else response(source)).encode()
    return asr.OpenAITranscriber('offline-fake-key',transport=send)


def test_multipart_words_cost_cache_and_source_remap(source,tmp_path):
    calls=[];p=provider(source,calls);ep=tmp_path/'one';ep.mkdir()
    words=p.words(source,episode_dir=ep)
    assert len(words)==2 and words[0]['start_ms']==200 and words[-1]['end_ms']==1400
    assert len(calls)==1 and calls[0][0]=='https://api.openai.com/v1/audio/transcriptions'
    body=calls[0][2]
    assert b'name="timestamp_granularities[]"' in body and b'word\r\n' in body
    assert b'verbose_json' in body and b'whisper-1' in body and b'RIFF' in body
    row=read_json(ep/'llm-usage.json')['calls'][0]
    assert row['role']=='transcription' and row['status']=='success'
    assert row['input_tokens'] is None and row['output_tokens'] is None
    assert row['cost_usd']>0 and row['cost_status']=='estimated'
    assert accounting._charged_tokens(row)==0
    assert 'offline-fake-key' not in (ep/'llm-usage.json').read_text()
    assert usage_summary(ep)[1]['known_ai_usd']==row['cost_usd']
    ep2=tmp_path/'two';ep2.mkdir();other=dict(source,source_id='renamed')
    cached=p.words(other,episode_dir=ep2)
    assert len(calls)==1 and all(w['source_id']=='renamed' for w in cached)
    assert p.sources['renamed']['cached'] and not (ep2/'llm-usage.json').exists()


@pytest.mark.parametrize('bad',[dict(words=None),dict(words=[]),dict(words=[dict(word='x',start=1,end=.99)]),dict(words=[dict(word='x',start=-1,end=1)]),dict(words=[dict(word='x',start=1,end=999)]),dict(words=[None])])
def test_missing_or_invalid_words_never_become_verified_empty(source,tmp_path,bad):
    p=provider(source,[],data=response(source,**bad))
    with pytest.raises(StageError) as error:p.words(source,episode_dir=tmp_path)
    assert error.value.code=='ASR_FAILED'
    row=read_json(tmp_path/'llm-usage.json')['calls'][0]
    assert row['status']=='failed' and row['cost_usd'] is not None


def test_real_empty_transcript_requires_explicit_response(source,tmp_path):
    p=provider(source,[],data=response(source,text='',words=[]))
    assert p.words(source,episode_dir=tmp_path)==[]
    assert p.sources[source['source_id']]['word_count']==0


def test_budget_blocks_before_any_transcription_dispatch(source,tmp_path,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.000001')
    calls=[];p=provider(source,calls)
    with pytest.raises(accounting.ProviderBudgetError):p.words(source,episode_dir=tmp_path)
    assert not calls
    row=read_json(tmp_path/'llm-usage.json')['calls'][0]
    assert row['status']=='blocked' and row['reserved_usd']==0


def test_transport_failure_retains_unknown_charge_reservation(source,tmp_path):
    def fail(*args):raise RuntimeError('offline-fake-key provider body must not leak')
    p=asr.OpenAITranscriber('offline-fake-key',transport=fail)
    with pytest.raises(StageError):p.words(source,episode_dir=tmp_path)
    row=read_json(tmp_path/'llm-usage.json')['calls'][0]
    assert row['cost_usd'] is None and row['reserved_usd']>0
    assert accounting._charged(row)==Decimal(str(row['reserved_usd']))
    assert 'offline-fake-key' not in json.dumps(row)


def test_transcription_spend_reduces_planner_allowance(source,tmp_path,monkeypatch):
    calls=[];p=provider(source,calls);p.words(source,episode_dir=tmp_path)
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.001')
    planner=accounting.LLMProvider('openai','fake','gpt-4.1',(Decimal(2),Decimal(8)),transport=lambda *a:pytest.fail('No dispatch after budget exhaustion'))
    with pytest.raises(accounting.ProviderBudgetError):planner.complete_json(system='plan',prompt='goal',images=[],role='planner',episode_dir=tmp_path,max_output_tokens=1024)


def test_router_prefers_local_and_never_falls_back_after_invalid_analysis(source,tmp_path,monkeypatch):
    import pipelines.ffmpeg.refinery as refinery
    ctx=EpisodeContext('router',tmp_path,'goal');router=asr.TranscriptionRouter(ctx)
    monkeypatch.setattr(refinery,'transcribe_words',lambda *a,**k:[])
    monkeypatch.setattr(asr.OpenAITranscriber,'from_env',lambda:pytest.fail('Local is preferred'))
    assert router.words(source)==[] and router.status=='local_asr'
    def invalid(*a,**kw):raise StageError('WORD_RANGE_INVALID','refinery','bad source words')
    monkeypatch.setattr(refinery,'transcribe_words',invalid)
    with pytest.raises(StageError):router.words(source)


def test_router_missing_dependency_falls_back_to_actual_provider_shape(source,tmp_path,monkeypatch):
    import pipelines.ffmpeg.refinery as refinery
    def missing(*a,**kw):raise StageError('DEPENDENCY_MISSING','refinery','no assets')
    monkeypatch.setattr(refinery,'transcribe_words',missing)
    p=provider(source,[]);monkeypatch.setattr(asr.OpenAITranscriber,'from_env',lambda:p)
    router=asr.TranscriptionRouter(EpisodeContext('router',tmp_path,'goal'))
    assert len(router.words(source))==2 and router.status=='openai_asr'
    assert router.sources[source['source_id']]['provider']=='openai'


def test_wav_preserves_source_duration_and_delayed_audio(tmp_path):
    import subprocess
    path=tmp_path/'delayed.mov'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=blue:s=160x90:r=30:d=3','-itsoffset','1','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=2','-c:v','libx264','-threads','1','-c:a','pcm_s16le',str(path)],check=True,capture_output=True)
    prepared=asr.prepare_audio(dict(path=str(path),duration_ms=3000,source_id='delay'),tmp_path)
    import array
    with wave.open(str(prepared)) as audio:
        assert audio.getframerate()==16000 and audio.getnframes()==48000
        values=array.array('h',audio.readframes(audio.getnframes()))
    assert max(abs(x) for x in values[:14000])==0
    assert max(abs(x) for x in values[17000:])>100


def test_real_audio_mock_asr_eligible_for_word_gate_and_single_judge(media,tmp_path,monkeypatch):
    from pipelines.ffmpeg.cli import run_episode
    from tests.test_run_evals import provider_transport
    # API-shaped response based on the actual WAV upload. No real ASR or billing.
    def send(url,headers,body,timeout):
        start=body.index(b'RIFF')
        with wave.open(io.BytesIO(body[start:])) as wav:duration=wav.getnframes()/wav.getframerate()
        return 200,json.dumps(dict(text='hello family',duration=duration,words=[dict(word='hello',start=.2,end=.6),dict(word='family',start=.9,end=1.4)])).encode()
    monkeypatch.setenv('OPENAI_API_KEY','offline-fake-key')
    monkeypatch.setattr(accounting,'_transport',send)
    planner,calls=provider_transport()
    qa=run_episode(media[0],'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='asr-e2e',provider=planner,evals_root=tmp_path/'evals')
    root=tmp_path/'episodes'/'asr-e2e';e=read_json(root/'eval-report.json')
    assert qa['passed'],qa
    assert read_json(root/'analysis.json')['transcription_status']=='openai_asr'
    assert e['scores']['G']==1 and e['scores']['J']==.75 and e['scores']['Q']>0
    assert e['model_cost']['quality_per_dollar']>0 and e['tokens']['complete']
    assert [r for r,_ in calls]==['planner','judge']
    assert len([r for r in qa['ai_usage'] if r['role']=='transcription'])==len(read_json(root/'analysis.json')['transcription_sources'])
    assert e['model_cost']['known_usd']==pytest.approx(sum(r['cost_usd'] for r in qa['ai_usage']))


def test_duration_billed_cost_breakdown_preserves_unknown_failure(tmp_path):
    from pipelines.ffmpeg.evaluator import model_cost
    row=dict(call_id='asr',role='transcription',provider='openai',model='whisper-1',status='success',input_tokens=None,output_tokens=None,cost_usd=.0264,cost_status='estimated',reserved_usd=.03,error=None)
    (tmp_path/'llm-usage.json').write_text(json.dumps(dict(version=2,calls=[row])))
    cost,tokens=model_cost(tmp_path,.5)
    assert cost['asr_usd']==.0264 and cost['llm_usd']==0
    assert cost['usd']==.0264 and cost['quality_per_dollar']==pytest.approx(.5/.0264,abs=1e-6)
    assert tokens==dict(input=0,output=0,complete=True)
    row.update(status='failed',cost_usd=None)
    (tmp_path/'llm-usage.json').write_text(json.dumps(dict(version=2,calls=[row])))
    cost,_=model_cost(tmp_path,.5)
    assert cost['asr_usd'] is None and cost['usd'] is None and cost['quality_per_dollar'] is None


def test_az_0305_reported_zero_length_yeah():
    # Exact reported response fields and offending word; the other word timings
    # were not supplied in the bug report, so do not invent them as live evidence.
    data={'duration':8.05,'text':"It's good, yeah, I like it.",
          'words':[{'word':'yeah','start':7.36,'end':7.36}]}
    words=asr._words(data,dict(source_id='source-0003',duration_ms=8050))
    assert words==[dict(source_id='source-0003',text='yeah',start_ms=7360,end_ms=7380,confidence=None)]


@pytest.mark.parametrize('rows,expected',[
    ([('yeah',7.36,7.36),('I',7.37,7.5)], [('yeah',7360,7370),('I',7370,7500)]),
    ([('yeah',7.36,7.36),('I',7.36,7.5)], [('yeah I',7360,7500)]),
    ([('yeah',7.36,7.4),('I',7.4,7.5)], [('yeah',7360,7400),('I',7400,7500)]),
    ([('yeah',7.36,7.4),('I',7.39,7.5)], [('yeah',7360,7390),('I',7390,7500)]),
    ([('yeah',7.36,7.5),('I',7.4,7.45)], [('yeah',7360,7400),('I',7400,7500)]),
    ([('yeah',7.36,7.5),('I',7.35,7.5)], [('yeah I',7360,7500)]),
    ([('yeah',8.04,8.04)], [('yeah',8040,8050)]),
    ([('yeah',8.05,8.05)], [('yeah',8030,8050)]),
    ([('yeah',8.03,8.07)], [('yeah',8030,8050)]),
    ([('yeah',7.3601,7.3602)], [('yeah',7360,7380)]),
])
def test_quantized_neighbors_and_source_edges(rows,expected):
    data=dict(text=' '.join(w for w,_,_ in rows),words=[dict(word=w,start=a,end=b) for w,a,b in rows])
    words=asr._words(data,dict(source_id='source-0003',duration_ms=8050))
    assert [(w['text'],w['start_ms'],w['end_ms']) for w in words]==expected
    assert all(0<=w['start_ms']<w['end_ms']<=8050 for w in words)
    assert all(a['end_ms']<=b['start_ms'] for a,b in zip(words,words[1:]))
    assert asr._words(dict(text=data['text'],words=[dict(word=w['text'],start=w['start_ms']/1000,end=w['end_ms']/1000) for w in words]),dict(source_id='source-0003',duration_ms=8050))==words


@pytest.mark.parametrize('start,end', [(-.001,.01),(8.06,8.06),(8,8.071),(7.36,7.359),(float('nan'),8),(True,1)])
def test_normalization_still_rejects_invalid_ranges(start,end):
    with pytest.raises(ValueError):
        asr._words(dict(text='yeah',words=[dict(word='yeah',start=start,end=end)]),dict(source_id='source-0003',duration_ms=8050))


def test_invalid_remote_source_degrades_without_cache_and_preserves_charge(source,tmp_path):
    p=provider(source,[],data=response(source,words=[dict(word='bad',start=1,end=.5)]))
    router=asr.TranscriptionRouter(EpisodeContext('degraded',tmp_path,'goal'),mode='openai');router.remote=p
    assert router.words(source)==[]
    audit=router.sources[source['source_id']]
    assert audit['status']=='degraded' and audit['word_count']==0 and not audit['cached']
    assert 'unverified' in audit['limitation'] and 'original audio retained' in audit['limitation']
    row=read_json(tmp_path/'llm-usage.json')['calls'][0]
    assert row['status']=='failed' and row['cost_usd']>0
    assert not list((Path(os.environ['FRAMEFOUNDRY_CACHE_DIR'])/'transcription').glob('*.json'))


def test_router_does_not_degrade_budget_failure(source,tmp_path,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.000001')
    router=asr.TranscriptionRouter(EpisodeContext('budget',tmp_path,'goal'),mode='openai')
    router.remote=provider(source,[])
    with pytest.raises(accounting.ProviderBudgetError):router.words(source)
    assert not router.sources


def test_one_invalid_source_continues_to_partial_render_without_false_word_gate(media,tmp_path,monkeypatch):
    from pipelines.ffmpeg.cli import run_episode
    from tests.test_run_evals import provider_transport
    api_calls=[]
    def send(url,headers,body,timeout):
        start=body.index(b'RIFF')
        with wave.open(io.BytesIO(body[start:])) as wav:duration=wav.getnframes()/wav.getframerate()
        api_calls.append(duration)
        words=[dict(word='hello',start=.2,end=.6),dict(word='family',start=.9,end=1.4)]
        if len(api_calls)==3:words=[dict(word='invalid',start=1,end=.5)]
        return 200,json.dumps(dict(text='hello family',duration=duration,words=words)).encode()
    monkeypatch.setenv('OPENAI_API_KEY','offline-fake-key')
    monkeypatch.setattr(accounting,'_transport',send)
    planner,calls=provider_transport()
    qa=run_episode(media[0],'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='asr-partial',provider=planner,evals_root=tmp_path/'evals',asr_provider='openai')
    root=tmp_path/'episodes'/'asr-partial';e=read_json(root/'eval-report.json');analysis=read_json(root/'analysis.json')
    assert len(api_calls)==3
    assert qa['passed'] and qa['result']=='partial' and qa['error'] is None,qa
    assert (root/'finished.mp4').is_file() and (root/'contact-sheet.jpg').is_file()
    degraded=[sid for sid,audit in analysis['transcription_sources'].items() if audit.get('status')=='degraded']
    assert len(degraded)==1 and analysis['sources'][2]['has_audio']
    assert all(w['source_id']!=degraded[0] for w in analysis['words'])
    assert any('ASR_FAILED' in limitation and degraded[0] in limitation for limitation in qa['coverage']['limitations'])
    gate=next(c for c in e['gates']['checks'] if c['name']=='word_cut_integrity')
    assert gate['status']=='not_run' and 'degraded' in gate['reason']
    assert e['scores']['G']==0 and e['scores']['Q']==0 and e['scores']['J'] is None
    assert [role for role,_ in calls]==['planner']
    assert len([row for row in qa['ai_usage'] if row['role']=='transcription'])==3
    assert e['model_cost']['known_usd']==pytest.approx(sum(row['cost_usd'] for row in qa['ai_usage']))


def test_zero_length_success_is_cached_and_remapped(source,tmp_path):
    calls=[]
    p=provider(source,calls,data=response(source,text='yeah',words=[dict(word='yeah',start=.9,end=.9)]))
    ep=tmp_path/'one';ep.mkdir()
    words=p.words(source,episode_dir=ep)
    assert words[0]['start_ms']==900 and words[0]['end_ms']==920
    assert read_json(ep/'llm-usage.json')['calls'][0]['status']=='success'
    other=dict(source,source_id='new-id');second=tmp_path/'two';second.mkdir()
    cached=p.words(other,episode_dir=second)
    assert len(calls)==1 and cached[0]['source_id']=='new-id'
    assert cached[0]['start_ms']==900 and cached[0]['end_ms']==920
    assert p.sources['new-id']['cached'] and not (second/'llm-usage.json').exists()
