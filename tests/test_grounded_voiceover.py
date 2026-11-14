"""Mock semantic claims test provenance/plumbing, not real scene recognition."""
from copy import deepcopy
import hashlib,json
import pytest
from PIL import Image
from pipelines.ffmpeg import semantics,planner,voiceover
from pipelines.ffmpeg.contracts import validate_narrative,StageError
from tests.test_short_speech_edit import analysis
from tests.test_visual_duration import context,FakeProvider


def framed(tmp_path):
    a=analysis()
    for s in a['sources']:
        s['keyframes']=[]
        original=tmp_path/(s['source_id']+'.mp4');original.write_bytes(b'fingerprint fixture, not render media')
        s['path']=str(original)
    for i,c in enumerate(a['candidates']):
        path=f'frame-{i}.jpg';Image.new('RGB',(240,160),(50,100,150)).save(tmp_path/path)
        t=(c['start_ms']+c['end_ms'])//2
        c.update(frame_path=path,canonical_frame_path=path,frame_time_ms=t,frame_in_window=True,quality_frame_time_ms=t)
        c['quality'].update(exposure=50,sharpness=30,face_count=1,person_count=0)
        next(s for s in a['sources'] if s['source_id']==c['source_id'])['keyframes'].append(dict(path=path,source_time_ms=t,proxy_time_ms=t))
    return a


class SubjectProvider(FakeProvider):
    def complete_json(self,**kwargs):
        rows=json.loads(kwargs['prompt'])['candidates']
        self.data=dict(interpretation='Selected trip moments',assumptions=[],coverage=dict(status='full',limitations=[]),
          segments=[dict(source_id=r['source_id'],raw_start_ms=r['start_ms'],raw_end_ms=r['end_ms'],reason='Mock subject label',subject_tags=[tag],semantic_labels=['people']) for r,tag in zip(rows,['night_parade','market','street_performers'])])
        return super().complete_json(**kwargs)


def test_same_call_subjects_survive_only_with_retained_frame_evidence(tmp_path):
    a=framed(tmp_path);before=deepcopy(a);provider=SubjectProvider(None)
    spec=planner.plan_teaser(context(tmp_path,a),a,provider=provider)
    assert spec['planner_mode']=='llm' and len(provider.calls)==1
    assert len(json.loads(provider.calls[0]['prompt'])['candidates'])==8
    proofs=[s['semantic_evidence'] for s in spec['segments'] if s.get('semantic_evidence')]
    assert len(proofs)==3
    assert all(p['view']=='center_crop_9_16' and p['method']=='planner_in_window_tile' for p in proofs)
    semantics.validate_evidence(spec,a,tmp_path);validate_narrative(spec,a)
    text=voiceover.intro_text(spec)
    assert 'Night parade' in text and 'market' in text and 'street performers' in text
    assert len(text)<90 and 1<=len(text.splitlines())<=2
    assert a==before
    # Persisted cached decisions still rebind to current frames/final selections.
    again=planner.plan_teaser(context(tmp_path,a),a,provider=provider)
    assert len(provider.calls)==1 and voiceover.intro_text(again)==text


def test_specific_labels_never_come_from_goal_or_reason():
    spec=dict(goal='Arizona Disneyland night parade market',segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,reason='Night parade, Arizona')])
    assert voiceover.intro_text(spec)=="A few selected moments.\nLet's take a look."


def test_subject_outside_selected_frame_rejected():
    r=dict(candidate_id='c',source_id='a',start_ms=0,end_ms=3000,frame_in_window=True,tile_index=0,frame_time_ms=2000)
    data=dict(interpretation='x',assumptions=[],coverage=dict(status='full',limitations=[]),segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,reason='x',subject_tags=['market'])])
    with pytest.raises(ValueError):planner._check_candidate_mapping(data,[r])


def test_unsupported_place_tag_rejected():
    data=dict(interpretation='x',assumptions=[],coverage=dict(status='full',limitations=[]),segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,reason='x',subject_tags=['Arizona'])])
    from jsonschema import ValidationError
    with pytest.raises(ValidationError):planner._check_candidate_mapping(data,[])


def test_reordering_or_replacement_does_not_transfer_subjects():
    r=dict(candidate_id='c',source_id='a',start_ms=0,end_ms=3000,frame_in_window=True,tile_index=0,frame_time_ms=2000,frame_path='x.jpg',frame_sha256='a'*64,view='center_crop_9_16')
    data=dict(segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=3000,subject_tags=['market'])])
    spec=dict(segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000),dict(source_id='b',raw_start_ms=0,raw_end_ms=3000)])
    semantics.attach(spec,data,[r]);assert all('semantic_evidence' not in s for s in spec['segments'])


def test_tampered_frame_hash_or_interval_fails(tmp_path):
    a=framed(tmp_path);spec=planner.plan_teaser(context(tmp_path,a),a,provider=SubjectProvider(None))
    s=next(s for s in spec['segments'] if s.get('semantic_evidence'))
    (tmp_path/s['semantic_evidence']['frame_path']).write_bytes(b'tampered')
    with pytest.raises(ValueError):semantics.validate_evidence(spec,a,tmp_path)
    s['semantic_evidence']['frame_time_ms']=s['raw_end_ms']
    with pytest.raises(StageError):validate_narrative(spec,a)


def test_screen_event_does_not_become_trip_subject():
    spec=dict(segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,semantic_evidence=dict(source_id='a',method='planner_in_window_tile',frame_time_ms=500,subjects=['night_parade'],labels=['screen']))])
    assert 'parade' not in voiceover.intro_text(spec)


def test_complete_mocked_pipeline_speaks_retained_subjects(tmp_path,monkeypatch):
    import subprocess
    from tests.fixtures.generate_footage import generate_footage
    from pipelines.ffmpeg.cli import run_episode
    from pipelines.ffmpeg import llm_provider
    footage=tmp_path/'footage';generate_footage(footage)
    audio=tmp_path/'mock.mp3'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','sine=frequency=730:sample_rate=44100:duration=1.2','-c:a','libmp3lame',str(audio)],check=True,capture_output=True)
    calls=[]
    def send(url,headers,body,timeout):calls.append(json.loads(body));return 200,audio.read_bytes()
    monkeypatch.setenv('ELEVENLABS_API_KEY','fake');monkeypatch.setattr(llm_provider,'_transport',send)
    provider=SubjectProvider(None)
    report=run_episode(footage,'10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='grounded',provider=provider,transcript_fixture=footage/'transcript-fixture.json',voiceover=True,planner_retry=False)
    assert report['passed'] and len(calls)==1 and len(provider.calls)==1
    spec=json.loads((tmp_path/'episodes'/'grounded'/'narrative-spec.json').read_text())
    assert spec['voiceover']['text']==calls[0]['text']==voiceover.intro_text(spec)
    assert spec['voiceover']['grounding_version']=='subjects-v1' and len(calls[0]['text'])<90
    assert any(s.get('semantic_evidence') for s in spec['segments'])
    evaluation=json.loads((tmp_path/'episodes'/'grounded'/'eval-report.json').read_text())
    assert next(c for c in evaluation['gates']['checks'] if c['name']=='voiceover_integrity')['status']=='pass'
