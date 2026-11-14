"""Synthetic offline v1 fixtures; never Arizona or sponsor performance proof."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from pipelines.ffmpeg.learning_view import MAX_BYTES, pending_curve, public_projection, read_curve
from pipelines.ffmpeg.scoreboard_html import render_html


def synthetic_row(eid='synthetic-on', iteration=1, **extra):
    return dict(iteration=iteration, episode_id=eid, status='passed', J=.6, Q=.7,
                model_usd=0, known_model_usd=0, cost_status='measured', elapsed_seconds=3,
                lessons_used=['synthetic-lesson'], **extra)


def synthetic_curve():
    return dict(schema_version='1.0', experiment_id='synthetic-experiment', status='recorded',
                created_at='2026-01-01T00:00:00Z', config_hash='synthetic-config',
                arms=dict(memory_on=[synthetic_row()], memory_off=[synthetic_row('synthetic-off')]),
                deltas=[dict(iteration=1, comparable=True, J=0, Q=0, model_usd=0)],
                limitations=['Synthetic test-only observation, not Arizona proof.'])


def save(root, curve):
    path=root/'learning-curves/latest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve))
    return path


def test_valid_and_pending_reader(tmp_path):
    curve=synthetic_curve();save(tmp_path,curve)
    assert read_curve(tmp_path)==curve
    save(tmp_path,dict(schema_version='1.0', status='pending', goal='SECRET'))
    assert read_curve(tmp_path)==pending_curve()


@pytest.mark.parametrize('kind', ['missing','malformed','oversize','wrong-version','nan','bool','bad-date',
                                  'bad-arm','duplicate-iteration','too-many-rows','unknown-status',
                                  'missing-metric','link-file','link-directory','link-root','fifo','directory'])
def test_fail_closed_reader(tmp_path, kind):
    root=tmp_path/'evals';root.mkdir();curve=synthetic_curve();path=save(root,curve)
    if kind=='missing':path.unlink()
    if kind=='malformed':path.write_text('{no')
    if kind=='oversize':path.write_bytes(b' '*(MAX_BYTES+1))
    if kind=='wrong-version':curve['schema_version']='2.0'
    if kind=='nan':curve['arms']['memory_on'][0]['J']=float('nan')
    if kind=='bool':curve['arms']['memory_on'][0]['model_usd']=False
    if kind=='bad-date':curve['created_at']='private/SECRET'
    if kind=='bad-arm':curve['arms']['memory_off']={}
    if kind=='duplicate-iteration':curve['arms']['memory_on']*=2
    if kind=='too-many-rows':curve['arms']['memory_on']*=201
    if kind=='unknown-status':curve['arms']['memory_on'][0]['status']='win'
    if kind=='missing-metric':del curve['arms']['memory_on'][0]['Q']
    if kind.startswith('link-'):
        target=path if kind=='link-file' else (path.parent if kind=='link-directory' else root)
        outside=tmp_path/('outside-'+kind);target.rename(outside);target.symlink_to(outside,target_is_directory=kind!='link-file')
    if kind=='fifo':path.unlink();os.mkfifo(path)
    if kind=='directory':path.unlink();path.mkdir()
    if kind in ('wrong-version','nan','bool','bad-date','bad-arm','duplicate-iteration','too-many-rows','unknown-status','missing-metric'):
        path.write_text(json.dumps(curve))
    assert read_curve(root)==pending_curve()


def test_public_projection_strips_every_private_field_and_unknown_id():
    curve=synthetic_curve();curve.update(goal='SECRET',credentials={'token':'SECRET'},source_path='/private/SECRET')
    curve['limitations']=['SECRET /private lesson text']
    row=curve['arms']['memory_off'][0];row.update(episode_id='private-secret',lesson_text='SECRET',goal='SECRET')
    curve['arms']['memory_on'][0]['lessons_used']=['SECRET']
    result=public_projection(curve,['synthetic-on'])
    serialized=json.dumps(result)
    for secret in ('SECRET','private-secret','goal','lesson','credential','source_path','config_hash','experiment_id'):
        assert secret not in serialized
    assert result['source_episode_ids']==['synthetic-on']
    withheld=result['arms']['memory_off'][0]
    assert withheld['episode_id'] is None and withheld['data_url'] is None
    for key in ('J','Q','model_usd','known_model_usd','elapsed_seconds'):
        assert withheld[key] is None
    assert result['deltas']==[dict(iteration=1,comparable=False,J=None,Q=None,model_usd=None)]


def test_failed_allowlisted_episode_is_not_public_evidence():
    curve=synthetic_curve();curve['arms']['memory_on'][0]['status']='failed'
    p=public_projection(curve,['synthetic-on','synthetic-off'])
    assert p['source_episode_ids']==['synthetic-off']
    assert p['arms']['memory_on'][0]['J'] is None
    assert p['deltas'][0]['J'] is None


def test_comparable_deltas_derive_from_actual_pair_not_untrusted_supplied_delta():
    curve=synthetic_curve();curve['deltas'][0].update(J=900,Q=900,model_usd=900)
    curve['arms']['memory_on'][0]['J']=.8
    p=public_projection(curve,['synthetic-on','synthetic-off'])
    assert p['deltas'][0]['J']==pytest.approx(.2)
    assert p['deltas'][0]['model_usd']==0
    assert p['arms']['memory_on'][0]['model_usd']==0
    curve['arms']['memory_on'][0].update(Q=None,model_usd=None,cost_status='unknown')
    p=public_projection(curve,['synthetic-on','synthetic-off'])
    assert p['deltas'][0]['Q'] is None and p['deltas'][0]['model_usd'] is None
    curve['arms']['memory_off']=[]
    assert public_projection(curve,['synthetic-on'])['deltas'][0]['comparable'] is False


def test_pending_projection_cannot_leak_extra_data():
    assert public_projection({'schema_version':'1.0','status':'pending','goal':'SECRET'},[])['arms']==dict(memory_on=[],memory_off=[])
    assert public_projection({'schema_version':'bad'},[])['status']=='pending'


def test_local_scoreboard_reads_private_curve_and_escapes_ids(tmp_path):
    episodes=tmp_path/'episodes';episodes.mkdir();curve=synthetic_curve()
    curve['arms']['memory_on'][0].update(episode_id='<img onerror="SECRET">',Q=None,model_usd=None,cost_status='unknown')
    save(tmp_path/'evals',curve)
    out=render_html([],episodes_root=episodes)
    assert 'Self-Improving' in out and 'LocalJsonl' in out and 'HydraDB' in out
    assert '&lt;img onerror=&quot;SECRET&quot;&gt;' in out
    assert '<img onerror=' not in out
    assert 'unknown' in out and 'N/A' in out and 'Unknown is N/A, not zero' in out
    (tmp_path/'evals/learning-curves/latest.json').unlink()
    assert 'Evaluation pending' in render_html([],episodes_root=episodes)


@pytest.mark.parametrize('kind', ['same-id','unknown-score','unknown-cost'])
def test_incomplete_or_self_pair_never_comparable(kind):
    curve=synthetic_curve()
    if kind=='same-id':curve['arms']['memory_off'][0]['episode_id']='synthetic-on'
    if kind=='unknown-score':curve['arms']['memory_on'][0]['J']=None
    if kind=='unknown-cost':curve['arms']['memory_on'][0]['cost_status']='unknown'
    delta=public_projection(curve,['synthetic-on','synthetic-off'])['deltas'][0]
    assert delta==dict(iteration=1,comparable=False,J=None,Q=None,model_usd=None)


@pytest.mark.parametrize('status', ['free',[],{}])
def test_unknown_cost_enum_cannot_reach_public_data(status):
    curve=synthetic_curve();curve['arms']['memory_on'][0]['cost_status']=status
    assert public_projection(curve,['synthetic-on','synthetic-off'])['status']=='pending'
