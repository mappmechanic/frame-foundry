"""Real offline full-run wiring; provider REST transports are mocked, not artistic proof."""
import json
from decimal import Decimal
from pathlib import Path
import pytest
from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.contracts import read_json
from pipelines.ffmpeg.eval_gates import generate_distinct_footage
from pipelines.ffmpeg.eval_judge import CRITERIA
from pipelines.ffmpeg.llm_provider import LLMProvider


@pytest.fixture(scope='module')
def footage(tmp_path_factory):
    root=tmp_path_factory.mktemp('run-eval-sources')
    fixture=generate_distinct_footage(root)
    return root,fixture


def provider_transport(*,judge_fails=False,short_initial=False):
    calls=[]
    def send(url,headers,body,timeout):
        request=json.loads(body)
        prompt=json.loads(next(c['text'] for c in request['messages'][0]['content'] if c['type']=='text'))
        if 'candidates' in prompt:
            remaining=8000 if short_initial and not calls else 10000
            segments=[]
            for c in prompt['candidates']:
                n=min(c['end_ms']-c['start_ms'],remaining)
                if n:
                    segments.append(dict(source_id=c['source_id'],raw_start_ms=c['start_ms'],raw_end_ms=c['start_ms']+n,reason='Planner-secret-reason: locally sampled candidate'))
                    remaining-=n
            if judge_fails and calls:segments=list(reversed(segments))
            data=dict(interpretation='Observed highlights',assumptions=['Fixture test only'],segments=segments,coverage=dict(status='partial' if short_initial and not calls else 'full',limitations=['Short initial edit'] if short_initial and not calls else []))
            calls.append(('planner',prompt))
        else:
            ids=[s['shot_id'] for s in prompt['edit']['shots']]
            data=dict(scores={c:2 if judge_fails else 4 for c in CRITERIA},evidence={c:ids[:1] for c in CRITERIA},weakest_shot=ids[-1],fix_instruction='Reverse shot order for a stronger opening',verdict='fail' if judge_fails else 'pass',unsupported_goal=False,limitations=[])
            calls.append(('judge',prompt))
        return 200,json.dumps(dict(stop_reason='end_turn',content=[dict(type='text',text=json.dumps(data))],usage=dict(input_tokens=1000,output_tokens=100))).encode()
    provider=LLMProvider('anthropic','offline-not-a-real-key','claude-sonnet-4-5',(Decimal(3),Decimal(15)),transport=send)
    return provider,calls


def test_run_automatically_evaluates_clean_cut_and_charges_single_judge(footage,tmp_path):
    path,fixture=footage;p,calls=provider_transport()
    qa=run_episode(path,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='clean',transcript_fixture=fixture,provider=p,evals_root=tmp_path/'evals')
    assert qa['passed'],qa
    ep=tmp_path/'episodes'/'clean';e=read_json(ep/'eval-report.json')
    assert e['scores']['G']==1 and e['scores']['J']==.75 and e['scores']['Q']>0
    assert [c[0] for c in calls]==['planner','judge']
    assert len(qa['ai_usage'])==2 and e['tokens']==dict(input=2000,output=200,complete=True)
    assert e['model_cost']['status']=='estimated' and e['model_cost']['usd']>0
    assert e['budget']['status']=='ok'
    assert 'Planner-secret-reason' not in json.dumps(calls[1][1]) and 'transcript' not in json.dumps(calls[1][1])
    assert len((tmp_path/'evals'/'scoreboard.jsonl').read_text().splitlines())==1
    assert read_json(ep/'state.json')['stage']=='completed'


def test_judge_driven_changed_cut_does_not_reuse_score(footage,tmp_path):
    path,fixture=footage;p,calls=provider_transport(judge_fails=True)
    qa=run_episode(path,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='retry',transcript_fixture=fixture,provider=p,evals_root=tmp_path/'evals')
    assert qa['passed'],qa
    e=read_json(tmp_path/'episodes'/'retry'/'eval-report.json')
    assert [c[0] for c in calls]==['planner','judge','planner']
    assert calls[2][1]['fix_instruction']=='Reverse shot order for a stronger opening'
    assert e['scores']['G']==1 and e['scores']['J'] is None and e['scores']['Q'] is None
    assert len(qa['attempts'])==2


def test_gate_driven_retry_preserves_single_remaining_judge(footage,tmp_path):
    path,fixture=footage;p,calls=provider_transport(short_initial=True)
    qa=run_episode(path,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='gate-retry',transcript_fixture=fixture,provider=p,evals_root=tmp_path/'evals')
    assert qa['passed'],qa
    e=read_json(tmp_path/'episodes'/'gate-retry'/'eval-report.json')
    assert [c[0] for c in calls]==['planner','planner','judge']
    assert e['scores']['G']==1 and e['scores']['J']==.75 and e['first_try_passed'] is False


def test_budget_stop_persists_both_reports_without_dispatch(footage,tmp_path,monkeypatch):
    path,fixture=footage;p,calls=provider_transport()
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.000001')
    qa=run_episode(path,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='budget',transcript_fixture=fixture,provider=p,evals_root=tmp_path/'evals')
    assert not qa['passed'] and qa['error']['code']=='BUDGET_LIMIT' and not calls
    e=read_json(tmp_path/'episodes'/'budget'/'eval-report.json')
    assert e['scores']['Q']==0 and e['budget']['blocked_calls']==1
    assert len((tmp_path/'evals'/'scoreboard.jsonl').read_text().splitlines())==1


def test_interruption_still_persists_eval(footage,tmp_path,monkeypatch):
    import pipelines.ffmpeg.cli as cli
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(cli,'run_refinery',interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_episode(footage[0],'10 second teaser',episodes_root=tmp_path/'episodes',episode_id='interrupt',provider=None,evals_root=tmp_path/'evals')
    ep=tmp_path/'episodes'/'interrupt'
    assert read_json(ep/'eval-report.json')['scores']['Q']==0
    assert read_json(ep/'state.json')['stage']=='interrupted'
    assert read_json(ep/'qa-report.json')['error']['code']=='INTERRUPTED'


def test_compatible_planner_rerun_uses_zero_new_planner_tokens(footage,tmp_path):
    path,fixture=footage;p,calls=provider_transport()
    # Only gate-passing episodes can cache a judge; identity includes final render.
    for eid in ['one','two']:
        qa=run_episode(path,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id=eid,transcript_fixture=fixture,provider=p,evals_root=tmp_path/'evals')
        assert qa['passed'],qa
    assert [c[0] for c in calls]==['planner','judge']
    e=read_json(tmp_path/'episodes'/'two'/'eval-report.json')
    assert e['judge']['cache_hit'] and e['judge']['historical_usage']['input_tokens']==1000
    assert e['tokens']==dict(input=0,output=0,complete=True) and e['model_cost']['usd']==0
