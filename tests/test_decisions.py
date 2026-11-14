import json
from types import SimpleNamespace
from decimal import Decimal
import pytest
from pipelines.ffmpeg import decisions as d
from pipelines.ffmpeg.contracts import EpisodeContext,usage_summary
from pipelines.ffmpeg.llm_provider import ProviderBudgetError


def rows(n=1):
    return [dict(candidate_id=f'candidate-{i}',quality={'visual_score':50},frame_in_window=True,tile_index=0) for i in range(n)]


def response(questions):
    answers=[]
    for q in questions:
        a=dict(name=q['name'],type=q['type'])
        if q['type']=='predicate': a['probability']=.7
        elif q['type']=='score': a.update(score=1,confidence=.5,probabilities=[dict(value=i,label=str(i),probability=1 if i==1 else 0) for i in range(3)])
        else: a.update(choice='other',confidence=1,probabilities=[dict(value=c['value'],probability=1 if c['value']=='other' else 0) for c in q['choices']])
        answers.append(a)
    return dict(model=d.MODEL,answers=answers,usage=dict(input_tokens=100,input_tokens_details=dict(cached_tokens=20,cache_write_tokens=30),output_tokens=0))


class Client:
    def __init__(self,error=None,mutate=None):
        self.decisions=self;self.calls=[];self.error=error;self.mutate=mutate
    def create(self,**kw):
        self.calls.append(kw)
        if self.error: raise self.error
        r=response(kw['questions'])
        if self.mutate:self.mutate(r)
        return r


def ctx(tmp_path,mode='decisions'):
    return EpisodeContext('offline',tmp_path,'Sedona trip',decider_mode=mode)


def test_batch_limit_and_mixed_types():
    q=d.batch(rows(66),['Halloween in Uptown Sedona','Hilton Sedona Resort at Bell Rock'])
    assert len(q)==396 and {x['type'] for x in q}=={'predicate','choice','score'}
    with pytest.raises(ValueError):d.batch(rows(67))
    with pytest.raises(ValueError):d.batch(rows(),['other'])


@pytest.mark.parametrize('mutate',[
 lambda r:r['answers'].pop(),lambda r:r['answers'].append(r['answers'][0]),
 lambda r:r['answers'][0].update(probability=float('nan')),
 lambda r:r['answers'][0].update(name='unknown'),
 lambda r:r['answers'][1].update(score=3),
 lambda r:r['answers'][0].update(type='refusal'),
 lambda r:r['answers'][0].update(probability=True),
 lambda r:r['answers'][-1].update(choice='invented'),
])
def test_invalid_answers_all_or_nothing(tmp_path,mutate):
    report=d.selection_hints(ctx(tmp_path),rows(),[],['section'],Client(mutate=mutate))
    assert report['effective_mode']=='heuristic' and report['answers'] is None
    call=json.loads((tmp_path/'llm-usage.json').read_text())['calls'][0]
    assert call['status']=='failed' and call['reserved_usd']>0


def test_request_shape_accounting_cache_latency(tmp_path):
    c=Client(); report=d.selection_hints(ctx(tmp_path),rows(),[],['section'],c)
    kw=c.calls[0]
    assert set(kw)=={'model','input','questions','prompt_cache_key'}
    assert len(kw['prompt_cache_key'])==64
    assert report['effective_mode']=='decisions'
    call=report['usage'];assert call['cost_status']=='estimated'
    assert Decimal(str(call['cost_usd']))==Decimal('0.00000895')
    assert call['latency_seconds']>=0 and call['cached_input_tokens']==20 and call['cache_write_tokens']==30
    clean,_=usage_summary(tmp_path);assert clean[0]['decision_stage']=='selection'
    c2=Client();r2=d.selection_hints(ctx(tmp_path),rows(),[],['section'],c2)
    assert r2['usage']['prompt_cache_key']==kw['prompt_cache_key']
    with pytest.raises(ProviderBudgetError):d.selection_hints(ctx(tmp_path),rows(),[],['section'],Client())


def test_failure_unknown_reservation_no_retry(tmp_path):
    c=Client(error=TimeoutError('SECRET'))
    report=d.selection_hints(ctx(tmp_path),rows(),[],client=c)
    assert len(c.calls)==1 and report['effective_mode']=='heuristic'
    call=json.loads((tmp_path/'llm-usage.json').read_text())['calls'][0]
    assert call['cost_usd'] is None and call['reserved_usd']>0
    assert 'SECRET' not in (tmp_path/'llm-usage.json').read_text()


def test_budget_denial_no_dispatch(tmp_path,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_TOKEN_BUDGET','1');c=Client()
    with pytest.raises(ProviderBudgetError):d.selection_hints(ctx(tmp_path),rows(),[],client=c)
    assert not c.calls


def test_missing_usage_unknown_not_zero(tmp_path):
    r=d.selection_hints(ctx(tmp_path),rows(),[],client=Client(mutate=lambda r:r.pop('usage')))
    assert r['usage']['cost_usd'] is None and r['usage']['output_tokens'] is None


def test_no_key_recorded_fallback(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    r=d.selection_hints(ctx(tmp_path),rows(),[])
    assert r['requested_mode']=='decisions' and r['effective_mode']=='heuristic' and r['fallback_reason']


def test_heuristic_no_client_dispatch(tmp_path):
    c=Client();r=d.selection_hints(ctx(tmp_path,'heuristic'),rows(),[],client=c)
    assert r['answers'] and not c.calls


def test_emitted_sidecar_and_ledger_strict_artifact_schemas(tmp_path):
    from jsonschema import Draft202012Validator
    from pathlib import Path
    d.selection_hints(ctx(tmp_path),rows(),[],['section'],Client())
    schema=json.loads((Path(__file__).resolve().parents[1]/'schemas/episode-artifacts.schema.json').read_text())
    for filename,definition in [('decisions-report.json','decisions_report'),('llm-usage.json','llm_usage')]:
        Draft202012Validator(schema['$defs'][definition]).validate(json.loads((tmp_path/filename).read_text()))


def test_sections_require_retained_frame_evidence():
    assert len(d.batch([dict(candidate_id='candidate-0')],['section']))==5


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(model='wrong-model'),
    lambda r:r['usage']['input_tokens_details'].update(cached_tokens=99,cache_write_tokens=99),
    lambda r:r['answers'][1]['probabilities'][0].update(label='invented'),
])
def test_wrong_model_cache_overlap_and_score_labels_fallback(tmp_path,mutate):
    r=d.selection_hints(ctx(tmp_path),rows(),[],client=Client(mutate=mutate))
    assert r['effective_mode']=='heuristic' and r['answers'] is None


def test_usage_overrun_is_terminal_not_hidden_fallback(tmp_path):
    with pytest.raises(ProviderBudgetError):
        d.selection_hints(ctx(tmp_path),rows(),[],client=Client(mutate=lambda r:r['usage'].update(input_tokens=999999)))
    call=json.loads((tmp_path/'llm-usage.json').read_text())['calls'][0]
    assert call['overrun'] is True


def test_missing_cache_count_unknown(tmp_path):
    r=d.selection_hints(ctx(tmp_path),rows(),[],client=Client(mutate=lambda r:r['usage']['input_tokens_details'].pop('cache_write_tokens')))
    assert r['usage']['cache_write_tokens'] is None and r['usage']['cost_usd'] is None


def test_missing_sdk_with_configured_key_fallback(tmp_path,monkeypatch):
    import sys
    monkeypatch.setenv('OPENAI_API_KEY','offline-not-real')
    monkeypatch.setitem(sys.modules,'openai',None)
    assert d.selection_hints(ctx(tmp_path),rows(),[])['effective_mode']=='heuristic'


def test_cache_identity_changes_with_goal_and_episode(tmp_path):
    a=ctx(tmp_path/'a');b=ctx(tmp_path/'b');b.episode_id='different';b.goal='Other Goal'
    first=d.selection_hints(a,rows(),[],client=Client())
    second=d.selection_hints(b,rows(),[],client=Client())
    assert first['usage']['prompt_cache_key']!=second['usage']['prompt_cache_key']


def test_private_frame_paths_not_dispatched(tmp_path):
    r=rows();r[0]['frame_path']='/private/path';c=Client()
    d.selection_hints(ctx(tmp_path),r,[],client=c)
    assert '/private/path' not in json.dumps(c.calls)
