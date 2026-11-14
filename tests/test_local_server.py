"""Mocked process API tests, plus real synthetic artifacts; never provider dispatch."""
from pathlib import Path
from types import SimpleNamespace
import json,shutil,subprocess
import pytest
from fastapi.testclient import TestClient
from pipelines.ffmpeg.local_server import create_app
from pipelines.ffmpeg.contracts import atomic_write_json,EpisodeContext,StageError,failure_report
from pipelines.ffmpeg.goal_parser import parse_goal
from pipelines.ffmpeg import factory_history

class Process:
    pid=99999999
    returncode=None
    def poll(self):return self.returncode
    def wait(self,timeout=None):self.returncode=-15;return self.returncode

@pytest.fixture
def rig(tmp_path,monkeypatch):
    for key in ('OPENAI_API_KEY','ELEVENLABS_API_KEY','ANTHROPIC_API_KEY','FRAMEFOUNDRY_EPISODE_API_BUDGET_USD'):
        monkeypatch.delenv(key,raising=False)
    spawned=[]
    def spawn(args,**kwargs):
        process=Process();spawned.append((args,kwargs,process));return process
    app=create_app(episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals',data_root=tmp_path/'data',spawn=spawn)
    folder=tmp_path/'raw';folder.mkdir();(folder/'clip.mov').write_bytes(b'mocked source; not valid video')
    with TestClient(app,base_url='http://127.0.0.1') as client:
        yield client,app.state.manager,folder,spawned

def payload(folder,**extra):return dict(goal='Family-trip teaser',footage_path=str(folder),aspect='9:16',voiceover=False,**extra)


def test_read_routes_are_empty_and_never_dispatch(rig,monkeypatch):
    c,m,folder,spawned=rig
    from pipelines.ffmpeg.llm_provider import LLMProvider
    monkeypatch.setattr(LLMProvider,'from_env',lambda:pytest.fail('provider initialized'))
    monkeypatch.setattr(factory_history.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    assert c.get('/').status_code==200
    assert c.get('/api/config').json()['max_upload_bytes']==32*1024**3
    assert c.get('/api/episodes').json()=={'episodes':[]}
    assert 'Presenter view' in c.get('/scoreboard').text
    assert not spawned


def test_generate_poll_refresh_and_conflict(rig,monkeypatch):
    c,m,folder,spawned=rig;monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.035')
    request=payload(folder,target_seconds=60)
    response=c.post('/api/jobs',json=request);assert response.status_code==202,response.text
    e=response.json();eid=e['episode_id']
    assert e['active'] and e['stage']=='starting' and e['can_regenerate'] and not e['video_url']
    args,options,process=spawned[0]
    assert '--duration-seconds' in args and '--aspect' in args and '--no-planner-retry' in args
    assert options['env']['FRAMEFOUNDRY_EPISODE_API_BUDGET_USD']=='0.035'
    assert '--goal=Family-trip teaser' in args and 'shell' not in options
    for _ in range(3):assert c.get('/api/episodes/'+eid).json()['active']
    assert c.post('/api/jobs',json=request).status_code==409 and len(spawned)==1
    directory=m.root/eid;directory.mkdir();atomic_write_json(directory/'state.json',dict(stage='judge'))
    assert c.get('/api/episodes/'+eid).json()['stage']=='judge'
    atomic_write_json(directory/'state.json',dict(stage='failed'))
    process.returncode=1
    ended=c.get('/api/episodes/'+eid).json()
    assert not ended['active'] and ended['stage']=='failed' and ended['error']
    assert ended['elapsed_seconds']==c.get('/api/episodes/'+eid).json()['elapsed_seconds']

@pytest.mark.parametrize('extra',[{'goal':''},{'target_seconds':0},{'target_seconds':True},{'target_seconds':'bad'},
                                  {'target_seconds':121},{'aspect':'1:1'},{'voiceover':True}])
def test_invalid_settings_and_missing_vo_do_not_start(rig,extra):
    c,m,folder,spawned=rig;p=payload(folder);p.update(extra)
    assert c.post('/api/jobs',json=p).status_code in (400,422)
    assert not spawned


def test_missing_empty_and_too_many_folders(rig,tmp_path):
    c,m,folder,spawned=rig
    for path in (tmp_path/'missing',tmp_path):
        assert c.post('/api/jobs',json=payload(path)).status_code==400
    for i in range(40):(folder/f'{i}.mp4').write_bytes(b'x')
    assert c.post('/api/jobs',json=payload(folder)).status_code==400
    assert not spawned

@pytest.mark.parametrize('headers',[{'Origin':'https://evil.example'},{'Origin':'null'},{'Host':'evil.example'},
                                     {'Sec-Fetch-Site':'cross-site'},{'Host':'[bad'}])
def test_foreign_origin_and_rebinding_blocked(rig,headers):
    c,m,folder,spawned=rig
    assert c.post('/api/jobs',json=payload(folder),headers=headers).status_code==403
    assert not spawned


def test_local_origin_and_server_singleton(rig,tmp_path):
    c,m,folder,spawned=rig
    assert c.get('/api/config',headers={'Origin':'http://127.0.0.1'}).status_code==200
    other=create_app(episodes_root=m.root,evals_root=m.evals,data_root=m.data)
    with pytest.raises(RuntimeError,match='Another UI server'):
        with TestClient(other,base_url='http://127.0.0.1'):pass


def test_existing_cli_lock_rejects_without_dispatch(rig):
    c,m,folder,spawned=rig
    from pipelines.ffmpeg.episode import episode_lock
    with episode_lock(m.root):assert c.post('/api/jobs',json=payload(folder)).status_code==409
    assert not spawned


def test_streamed_folder_upload_preserves_unique_files_and_source(rig):
    c,m,folder,spawned=rig
    response=c.post('/api/uploads',files=[('files',('trip/a.mov',b'one','video/quicktime')),
                                        ('files',('trip/sub/a.mov',b'two','video/quicktime'))])
    assert response.status_code==201,response.text
    uploaded=response.json();assert uploaded['file_count']==2 and uploaded['bytes']==6
    assert sorted(p.read_bytes() for p in (m.data/'uploads'/uploaded['upload_id']).iterdir())==[b'one',b'two']
    p=payload(folder);p['footage_path']=None;p['upload_id']=uploaded['upload_id']
    assert c.post('/api/jobs',json=p).status_code==202
    assert (folder/'clip.mov').read_bytes()==b'mocked source; not valid video'

@pytest.mark.parametrize('name',['../escape.mov','/absolute.mov','trip/../bad.mp4','a\\b.mov','x.txt','a//b.mov'])
def test_upload_names_cannot_traverse(rig,name):
    c,m,folder,spawned=rig
    assert c.post('/api/uploads',files={'files':(name,b'x')}).status_code==400
    assert not list((m.data/'uploads').iterdir()) and not spawned


def test_duplicate_and_count_uploads_rejected(rig):
    c,m,folder,spawned=rig
    files=[('files',('a.mov',b'x')),('files',('a.mov',b'y'))]
    assert c.post('/api/uploads',files=files).status_code==400
    assert c.post('/api/uploads',files=[('files',(f'{i}.mov',b'x')) for i in range(41)]).status_code==400


def test_upload_budget_enforced_with_cleanup(tmp_path):
    app=create_app(episodes_root=tmp_path/'e',data_root=tmp_path/'d',max_upload_bytes=3)
    with TestClient(app,base_url='http://localhost') as c:
        assert c.post('/api/uploads',files={'files':('a.mov',b'1234')}).status_code==413
        assert not list((tmp_path/'d/uploads').iterdir())
        assert c.post('/api/uploads',content=b'x',headers={'Content-Length':str(3*1024**3)}).status_code==413


def test_failed_report_preview_and_traversal(rig,tmp_path):
    c,m,folder,spawned=rig;eid='failed';d=m.root/eid;d.mkdir()
    report=failure_report(EpisodeContext(eid,d,'Goal'),StageError('FAIL','refinery','Specific failure'))
    atomic_write_json(d/'qa-report.json',report);atomic_write_json(d/'state.json',dict(stage='failed'))
    e=c.get('/api/episodes/'+eid).json();assert e['qa_passed'] is False and e['error']=='Specific failure'
    assert e['Q'] is None and not e['video_url']
    outside=tmp_path/'private';outside.write_text('secret');(d/'contact-sheet.jpg').symlink_to(outside)
    assert c.get('/api/episodes/failed/artifacts/contact-sheet.jpg').status_code==400
    assert c.get('/api/episodes/failed/artifacts/private').status_code==400
    assert c.get('/api/episodes/missing').status_code==404
    (m.root/'alias').symlink_to(d,target_is_directory=True)
    assert c.get('/api/episodes/alias').status_code==400


def test_restart_retains_identity_without_dispatch(tmp_path):
    data=tmp_path/'data';(data/'jobs').mkdir(parents=True)
    atomic_write_json(data/'jobs'/'interrupted.json',dict(episode_id='interrupted',stage='planner',goal='original',
        footage_path=str(tmp_path/'missing'),settings=dict(target_seconds=None,aspect='9:16',voiceover=False),created_at='2026-11-14T00:00:00+00:00'))
    def fail(*a,**k):pytest.fail('restart dispatched a process')
    app=create_app(episodes_root=tmp_path/'e',data_root=data,spawn=fail)
    with TestClient(app,base_url='http://localhost') as c:
        e=c.get('/api/episodes/interrupted').json()
        assert e['stage']=='interrupted' and 'no automatic resume' in e['error'] and not e['active']


def test_regeneration_retains_parent_and_original_settings(rig):
    c,m,folder,spawned=rig
    e=c.post('/api/jobs',json=payload(folder,target_seconds=30)).json();eid=e['episode_id']
    spawned[0][2].returncode=1
    c.get('/api/episodes/'+eid) # settle process before snapshot
    original=(m.data/'jobs'/(eid+'.json')).read_bytes()
    response=c.post('/api/episodes/'+eid+'/regenerate',json={'feedback':'More people in the closing'})
    assert response.status_code==202,response.text
    new=response.json();assert new['episode_id']!=eid and new['settings']['target_seconds']==30
    assert 'Feedback hint: More people' in new['goal'] and len(spawned)==2
    assert (m.data/'jobs'/(eid+'.json')).read_bytes()==original


def test_explicit_controls_keep_goal_and_crop_constraints():
    intent=parse_goal('60 seconds landscape 16:9 trip teaser',target_seconds=30,aspect='9:16')
    assert intent.target_duration_ms==30000 and intent.aspect_ratio=='9:16'
    assert parse_goal('teaser').target_duration_ms==45000


def test_synthetic_finished_video_is_range_playable_and_stale_eval_hidden(tmp_path,monkeypatch):
    # Genuine FFmpeg audio/contact-sheet gates, not a fake passing media fixture.
    from tests.fixtures.generate_footage import generate_footage
    from pipelines.ffmpeg.cli import run_episode
    for name in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','ELEVENLABS_API_KEY'):monkeypatch.delenv(name,raising=False)
    footage=tmp_path/'footage';generate_footage(footage)
    root=tmp_path/'episodes'
    report=run_episode(footage,'10 second teaser',episodes_root=root,episode_id='synthetic',provider=None,
        transcript_fixture=footage/'transcript-fixture.json',planner_retry=False,evals_root=tmp_path/'evals')
    assert report['passed'] and -15<=report['integrated_lufs']<=-13
    app=create_app(episodes_root=root,data_root=tmp_path/'data',evals_root=tmp_path/'evals')
    with TestClient(app,base_url='http://localhost') as c:
        e=c.get('/api/episodes/synthetic').json()
        assert e['video_url'] and e['contact_sheet_url'] and e['shots']
        q=c.get(e['video_url'],headers={'Range':'bytes=0-31'})
        assert q.status_code==206 and len(q.content)==32 and q.headers['content-range'].startswith('bytes 0-31/')
        assert c.get(e['video_url']+'?download=1').headers['content-disposition'].startswith('attachment')
        assert c.get(e['contact_sheet_url']).content
        (root/'synthetic'/'finished.mp4').write_bytes(b'changed')
        e=c.get('/api/episodes/synthetic').json()
        assert e['evaluation_status']=='stale' and e['Q'] is None and not e['video_url']


@pytest.mark.parametrize('budget',['NaN','inf','-1','bad'])
def test_invalid_budget_never_dispatches(rig,monkeypatch,budget):
    c,m,folder,spawned=rig
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD',budget)
    assert c.post('/api/jobs',json=payload(folder)).status_code==400
    assert not spawned


def test_demo_cap_cannot_be_raised(rig,monkeypatch):
    c,m,folder,spawned=rig;monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','5')
    assert c.post('/api/jobs',json=payload(folder)).status_code==202
    assert spawned[0][1]['env']['FRAMEFOUNDRY_EPISODE_API_BUDGET_USD']=='0.10'


def test_unavailable_footage_regeneration_no_new_job(rig):
    c,m,folder,spawned=rig
    eid=c.post('/api/jobs',json=payload(folder)).json()['episode_id'];spawned[0][2].returncode=1
    shutil.rmtree(folder)
    assert not c.get('/api/episodes/'+eid).json()['can_regenerate']
    assert c.post('/api/episodes/'+eid+'/regenerate',json={'feedback':'More energy'}).status_code==400
    assert len(spawned)==1


def test_malformed_usage_is_unavailable_not_zero_total(rig):
    c,m,folder,spawned=rig;d=m.root/'invalid';d.mkdir()
    atomic_write_json(d/'llm-usage.json',dict(calls=['not a call']))
    e=c.get('/api/episodes/invalid').json()
    assert e['model_cost']['usd'] is None and e['reservations_usd'] is None


def test_scoreboard_route_uses_actual_jsonl(rig,monkeypatch):
    c,m,folder,spawned=rig
    monkeypatch.setattr(factory_history.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    m.evals.mkdir(exist_ok=True)
    # An invalid line must be counted; it must not silently load the wrong path.
    (m.evals/'scoreboard.jsonl').write_text('{bad\n')
    response=c.get('/scoreboard');assert response.status_code==200
    assert 'invalid/corrupt lines: 1' in response.text and 'Scoreboard input: unavailable' not in response.text


def test_invalid_narrative_does_not_crash_preview(rig):
    c,m,folder,spawned=rig;d=m.root/'bad-spec';d.mkdir()
    atomic_write_json(d/'narrative-spec.json',dict(target_duration_ms='not a duration',voiceover='not metadata'))
    response=c.get('/api/episodes/bad-spec');assert response.status_code==200
    assert 'Narrative artifact invalid or unavailable' in response.json()['limitations']
    assert not response.json()['video_url']
