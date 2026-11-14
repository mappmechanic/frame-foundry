from copy import deepcopy
import json
import pytest
from pipelines.ffmpeg.campaigns import DEFAULT, load, save_binding, project
from scripts.campaign import main

def test_default_three_planned_slots(tmp_path):
    value=project(load(tmp_path/'missing.json'))
    assert [v['target_seconds'] for v in value['variants']]==[45,15,15]
    assert all(v['status']=='planned' and v['run'] is None for v in value['variants'])

def test_explicit_binding_persists_and_init_preserves(tmp_path):
    path=tmp_path/'campaign.json'
    save_binding(path,'sedona-kids','az-real')
    save_binding(path)
    assert load(path)['variants'][0]['episode_id']=='az-real'
    save_binding(path,'sedona-kids',None)
    assert load(path)==DEFAULT

@pytest.mark.parametrize('eid',['../secret','/tmp/a','UPPER','a/b','',12])
def test_unsafe_binding_leaves_manifest_unchanged(tmp_path,eid):
    path=tmp_path/'campaign.json';save_binding(path)
    before=path.read_bytes()
    with pytest.raises(ValueError):save_binding(path,'sedona-kids',eid)
    assert path.read_bytes()==before

def test_allowlist_only_real_bound_run_and_failures():
    value=deepcopy(DEFAULT)
    value['variants'][0]['episode_id']='az-real'
    value['variants'][1]['episode_id']='az-failed'
    value['variants'][2]['episode_id']='az-missing'
    run=dict(id='az-real',scores=dict(Q=.8,D=1),cost=dict(usd=.3),data_url='/runs/az-real/run.json')
    result=project(value,[run],failed_ids=['az-failed'])
    assert [v['status'] for v in result['variants']]==['verified','failed','unavailable']
    assert result['variants'][0]['run']['scores']==run['scores']
    assert result['variants'][1]['run'] is None
    assert project(value,[])['variants'][0]['run'] is None
    assert 'status' not in value['variants'][0]

def test_symlink_rejected(tmp_path):
    target=tmp_path/'target.json';save_binding(target)
    link=tmp_path/'link.json';link.symlink_to(target)
    with pytest.raises(ValueError):load(link)
    with pytest.raises(ValueError):save_binding(link)

def test_cli_never_launches_processing(tmp_path,capsys):
    path=tmp_path/'manifest.json'
    assert main(['--manifest',str(path),'init'])==0
    assert main(['--manifest',str(path),'bind','halloween-uptown','az-real'])==0
    assert load(path)['variants'][1]['episode_id']=='az-real'
    assert main(['--manifest',str(path),'bind','bellrock-resort','../unsafe'])==1
