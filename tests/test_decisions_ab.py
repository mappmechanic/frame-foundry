from copy import deepcopy
import pytest
from pipelines.ffmpeg.decisions_ab import compare,MATCH_FIELDS,arm
from pipelines.ffmpeg.contracts import atomic_write_json


def arms():
    config={k:'matched' for k in MATCH_FIELDS};config['judge_protocol']='gpt-4.1-current-six-criteria'
    a=dict(episode_id='baseline',Q=.8,J=.8,total_model_cost_usd=.1,cost_status='estimated',quality_per_dollar=8,
           elapsed_seconds=20,hard_qa_passed=True,requested_mode='llm',effective_mode='llm',fallback_reason=None,
           config=config,actual_footage_hash='matched',actual_goal='matched',decision_call_id=None)
    b=deepcopy(a);b.update(episode_id='experimental',Q=.78,total_model_cost_usd=.09,requested_mode='decisions',effective_mode='decisions',decision_call_id='recorded')
    return a,b


def test_threshold_exact_and_no_activation():
    a,b=arms();r=compare(a,b)
    assert r['status']=='qualified' and r['default_activated'] is False
    b['Q']=.779999;assert compare(a,b)['status']=='unqualified'


@pytest.mark.parametrize('field,value', [('total_model_cost_usd',.1),('total_model_cost_usd',None),('Q',None),('J',None),('cost_status','unknown'),('effective_mode','heuristic'),('hard_qa_passed',False),('elapsed_seconds',None),('decision_call_id',None),('fallback_reason','timeout')])
def test_incomplete_equal_cost_or_fallback_cannot_qualify(field,value):
    a,b=arms();b[field]=value
    assert compare(a,b)['status']=='unqualified'


def test_unmatched_and_missing_configs():
    a,b=arms();b['config']['cache_policy']='warm'
    assert compare(a,b)['status']=='unqualified'
    del b['config']['judge_protocol'];assert compare(a,b)['status']=='unqualified'


def test_absent_runs_pending(tmp_path):
    a=arm(tmp_path,{})
    assert a['Q'] is None and compare(a,a)['status']=='unqualified'


def test_arm_reads_canonical_eval_and_qa(tmp_path):
    atomic_write_json(tmp_path/'eval-report.json',dict(episode_id='a',scores=dict(G=1,Q=.8,J=.9),gates=dict(passed=True),model_cost=dict(status='estimated',usd=.1,quality_per_dollar=8)))
    atomic_write_json(tmp_path/'qa-report.json',dict(passed=True,elapsed_seconds=12))
    assert arm(tmp_path,{})['hard_qa_passed'] is True
