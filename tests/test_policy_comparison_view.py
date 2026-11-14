"""Explicitly synthetic A/B fixtures; no paid runs or sponsor performance proof."""
from copy import deepcopy
import json
import os

import pytest

from pipelines.ffmpeg.learning_view import (
    MAX_BYTES, pending_policy_comparison, public_policy_projection, read_policy_comparison,
)


def synthetic_policy():
    row = dict(status='passed', J=.6, Q=.7, model_usd=.02,
               known_model_usd=.02, cost_status='measured')
    return dict(schema_version='1.0', status='recorded', experiment_id='synthetic-only',
                created_at='2026-01-01T00:00:00Z', baseline=dict(row, episode_id='synthetic-base'),
                production=dict(row, episode_id='synthetic-prod', J=.5, model_usd=.01, known_model_usd=.01),
                comparison=dict(comparable=True, J=900, Q=900, model_usd=900), limitations=['Synthetic only.'])


def save(root, value):
    path=root/'policy-comparisons/latest.json';path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value));return path


def project(value):
    return public_policy_projection(value, ['synthetic-base','synthetic-prod'])


def test_reader_and_derived_public_deltas(tmp_path):
    value=synthetic_policy();save(tmp_path,value)
    assert read_policy_comparison(tmp_path)['status']=='recorded'
    p=project(read_policy_comparison(tmp_path))
    assert p['comparison']==dict(comparable=True, J=pytest.approx(-.1), Q=0, model_usd=pytest.approx(-.01))
    assert p['baseline']['data_url']=='/runs/synthetic-base/run.json'
    assert set(p)=={'schema_version','status','baseline','production','comparison','source_episode_ids','limitations'}
    assert set(p['baseline'])=={'episode_id','status','J','Q','model_usd','known_model_usd','cost_status','data_url'}


@pytest.mark.parametrize('kind', ['missing','malformed','oversize','file-link','dir-link','root-link','fifo','directory'])
def test_bounded_no_follow_policy_read(tmp_path,kind):
    root=tmp_path/'evals';path=save(root,synthetic_policy())
    if kind=='missing':path.unlink()
    if kind=='malformed':path.write_text('{invalid')
    if kind=='oversize':path.write_bytes(b' '*(MAX_BYTES+1))
    if kind in ('file-link','dir-link','root-link'):
        target=path if kind=='file-link' else path.parent if kind=='dir-link' else root
        outside=tmp_path/'outside';target.rename(outside);target.symlink_to(outside,target_is_directory=kind!='file-link')
    if kind=='fifo':path.unlink();os.mkfifo(path)
    if kind=='directory':path.unlink();path.mkdir()
    assert read_policy_comparison(root)==pending_policy_comparison()


@pytest.mark.parametrize('kind', ['version','enum','enum-list','cost-enum','cost-list','date','same-id','traversal','boolean',
    'nan','infinity','negative','over-unit','missing-score','bad-comparable','delta-nan','missing-delta','limitations','arm-list'])
def test_invalid_policy_fails_closed(tmp_path,kind):
    v=synthetic_policy();r=v['production']
    if kind=='version':v['schema_version']='2.0'
    if kind=='enum':r['status']='win'
    if kind=='enum-list':r['status']=[]
    if kind=='cost-enum':r['cost_status']='free'
    if kind=='cost-list':r['cost_status']=[]
    if kind=='date':v['created_at']='SECRET'
    if kind=='same-id':r['episode_id']=v['baseline']['episode_id']
    if kind=='traversal':r['episode_id']='../SECRET'
    if kind=='boolean':r['Q']=True
    if kind=='nan':r['Q']=float('nan')
    if kind=='infinity':r['model_usd']=float('inf')
    if kind=='negative':r['model_usd']=-1
    if kind=='over-unit':r['J']=1.1
    if kind=='missing-score':del r['J']
    if kind=='bad-comparable':v['comparison']['comparable']=1
    if kind=='delta-nan':v['comparison']['Q']=float('nan')
    if kind=='missing-delta':del v['comparison']['Q']
    if kind=='limitations':v['limitations']={}
    if kind=='arm-list':v['baseline']=[]
    save(tmp_path,v)
    assert read_policy_comparison(tmp_path)==pending_policy_comparison()
    assert project(v)['status']=='pending'


@pytest.mark.parametrize('kind', ['failed','unknown-cost','incomplete-cost','unknown-score','unknown-known-cost','known-over-total','zero-inconsistent','unpublished'])
def test_whole_pair_withheld_without_complete_public_evidence(kind):
    v=synthetic_policy();r=v['production'];ids=['synthetic-base','synthetic-prod']
    if kind=='failed':r['status']='failed'
    if kind=='unknown-cost':r.update(cost_status='unknown',model_usd=None)
    if kind=='incomplete-cost':r['cost_status']='incomplete'
    if kind=='unknown-score':r['Q']=None
    if kind=='unknown-known-cost':r['known_model_usd']=None
    if kind=='known-over-total':r['known_model_usd']=.03
    if kind=='zero-inconsistent':r['cost_status']='zero'
    if kind=='unpublished':ids=['synthetic-base']
    p=public_policy_projection(v,ids)
    assert p['status']=='recorded' and p['baseline'] is None and p['production'] is None
    assert p['source_episode_ids']==[] and p['comparison']==dict(comparable=False,J=None,Q=None,model_usd=None)
    assert 'synthetic-' not in json.dumps(p)


def test_incomparable_zero_estimated_and_privacy():
    v=synthetic_policy();v.update(source='/SECRET',goal='SECRET',config={'key':'SECRET'})
    v['limitations']=['SECRET lesson text'];v['comparison']['comparable']=False
    for row in (v['baseline'],v['production']):
        row.update(model_usd=0,known_model_usd=0,cost_status='zero',lesson='SECRET')
    p=project(v)
    assert p['baseline']['model_usd']==0
    assert p['comparison']==dict(comparable=False,J=None,Q=None,model_usd=None)
    assert 'SECRET' not in json.dumps(p) and 'experiment_id' not in p
    v['comparison']['comparable']=True;v['production']['cost_status']='estimated'
    assert project(v)['comparison']['model_usd']==0
    assert project(dict(schema_version='1.0',status='pending',goal='SECRET'))['status']=='pending'
