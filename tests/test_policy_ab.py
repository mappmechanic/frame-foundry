from copy import deepcopy
from scripts.policy_ab import compare,run
import pytest
from pipelines.ffmpeg.contracts import StageError,atomic_write_json

def sample():
    r=dict(status='passed',J=.8,Q=.8,model_usd=.01,cost_status='measured')
    e=dict(footage_hash='a'*64,goal='g',scores={'method_version':'v'},judge={'status':'scored','provider':'p','model':'m'})
    return r,e

def test_only_matched_measured_pairs_claim_delta():
    r,e=sample();b=dict(r,Q=.82)
    assert compare(r,b,e,e)['Q']==pytest.approx(.02)
    for key,value in [('footage_hash','b'*64),('goal','other')]:
        other=dict(e,**{key:value});assert compare(r,b,e,other)['comparable'] is False
    other=deepcopy(e);other['judge']['status']='unavailable';assert not compare(r,b,e,other)['comparable']
    assert not compare(r,dict(b,model_usd=None),e,e)['comparable']

def test_policy_baseline_and_limit_are_fail_closed(tmp_path):
    atomic_write_json(tmp_path/'narrative-spec.json',{'editing_policy':{'mode':'production-v1'}})
    with pytest.raises(StageError,match='original legacy'):run(tmp_path,tmp_path/'runs',tmp_path/'evals','ab')
    with pytest.raises(StageError,match='<=300'):run(tmp_path,tmp_path/'runs',tmp_path/'evals','ab',budget_limit=301)

def test_both_offline_arms_use_same_seed_and_no_planner(monkeypatch,tmp_path):
    from scripts import policy_ab
    atomic_write_json(tmp_path/'narrative-spec.json',{})
    calls=[]
    def replay(source,root,eid,**kwargs):
        calls.append((source,kwargs['production'],kwargs['judge']))
        d=root/eid;d.mkdir(parents=True);r,e=sample();e.update(scores=dict(J=.8,Q=.8,method_version='v'),gates={'passed':True},model_cost={'usd':.01,'known_usd':.01,'status':'measured'})
        atomic_write_json(d/'eval-report.json',e);return d,{'passed':True}
    monkeypatch.setattr(policy_ab,'rerender',replay)
    result=run(tmp_path,tmp_path/'runs',tmp_path/'evals','ab')
    assert calls==[(tmp_path,False,False),(tmp_path,True,False)]
    assert result['comparison']['comparable'] and result['comparison']['Q']==0
    assert (tmp_path/'evals/policy-comparisons/latest.json').exists()


def test_experiment_paths_and_overwrites_refused(tmp_path):
    atomic_write_json(tmp_path/'narrative-spec.json',{})
    with pytest.raises(StageError,match='safe'):run(tmp_path,tmp_path/'runs',tmp_path/'evals','../bad')
    (tmp_path/'runs/ab-base').mkdir(parents=True)
    with pytest.raises(StageError,match='fresh'):run(tmp_path,tmp_path/'runs',tmp_path/'evals','ab')
