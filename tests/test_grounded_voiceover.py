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
    assert 'night parade' in text and 'street performers' in text and 'market' not in text
    assert len(text)<90 and 1<=len(text.splitlines())<=2
    assert a==before
    # Persisted cached decisions still rebind to current frames/final selections.
    again=planner.plan_teaser(context(tmp_path,a),a,provider=provider)
    assert len(provider.calls)==1 and voiceover.intro_text(again)==text


def test_specific_labels_never_come_from_goal_or_reason():
    spec=dict(goal='Arizona Disneyland night parade market',segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,reason='Night parade, Arizona')])
    assert voiceover.intro_text(spec)=='Come along for these moments!'


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
    assert any(noun in calls[0]['text'].lower() for noun in ('night parade','market','street performers'))
    assert spec['voiceover']['grounding_version']=='subjects-v4' and len(calls[0]['text'])<90
    assert any(s.get('semantic_evidence') for s in spec['segments'])
    evaluation=json.loads((tmp_path/'episodes'/'grounded'/'eval-report.json').read_text())
    assert next(c for c in evaluation['gates']['checks'] if c['name']=='voiceover_integrity')['status']=='pass'


def test_closed_subject_aliases_do_not_trigger_whole_plan_fallback(tmp_path):
    class Aliases(SubjectProvider):
        def complete_json(self,**kwargs):
            response=super().complete_json(**kwargs)
            response.data['segments'][0]['subject_tags']=['Night parade','people']
            return response
    a=framed(tmp_path)
    spec=planner.plan_teaser(context(tmp_path,a),a,provider=Aliases(None))
    assert spec['planner_mode']=='llm'
    assert 'night parade' in voiceover.intro_text(spec).lower()
    validate_narrative(spec,a);semantics.validate_evidence(spec,a,tmp_path)


def test_shortlist_restores_baseline_rank_even_with_sparse_frame_evidence(tmp_path):
    from pipelines.ffmpeg.candidates import rank_candidates
    a=framed(tmp_path)
    for c in a['candidates'][3:]:
        c.update(frame_path=None,frame_in_window=False,frame_time_ms=None)
        c['quality']['visual_score']=100
    ctx=context(tmp_path,a)
    rows,images=planner._compact_evidence(ctx,a,tmp_path,a['candidates'])
    assert [r['candidate_id'] for r in rows]==[c['candidate_id'] for c in rank_candidates(a['candidates'],ctx.goal,top_n=8)]
    assert len(rows)==8 and len(images)<=1


def test_relative_pronoun_is_not_requested_trip_content():
    from pipelines.ffmpeg.goal_parser import parse_goal
    intent=parse_goal('Create a teaser from this raw family-trip footage that highlights the important moments')
    assert not intent.requested_terms


def test_distinctive_retained_subjects_exclude_generic_family_filler():
    s=dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,semantic_evidence=dict(method='planner_in_window_tile',source_id='a',frame_time_ms=500,
          subjects=['family_walk','family_activity','night_parade','market','street_performers'],labels=['people']))
    spec=dict(segments=[s]);text=voiceover.intro_text(spec)
    assert 'night parade' in text and 'street performers' in text and 'market' not in text
    assert 'family activit' not in text.lower() and len(text)<90
    assert semantics.intro_v1(spec)!=text
    # Losing that evidence must remove the named subject, not transfer it.
    s['raw_end_ms']=400
    assert 'parade' not in voiceover.intro_text(spec)


def test_validated_baseline_cache_is_reused_for_matching_ui_defaults(tmp_path):
    a=framed(tmp_path);provider=SubjectProvider(None);ctx=context(tmp_path,a)
    first=planner.plan_teaser(ctx,a,provider=provider)
    ctx.aspect_override='9:16';ctx.target_seconds=45
    second=planner.plan_teaser(ctx,a,provider=provider)
    assert len(provider.calls)==1
    assert first['segments']==second['segments']
    ctx.target_seconds=30
    third=planner.plan_teaser(ctx,a,provider=provider)
    assert len(provider.calls)==2 and third['target_duration_ms']==30000


def subject_spec(tags):
    return dict(segments=[dict(source_id='a',raw_start_ms=0,raw_end_ms=1000,
        semantic_evidence=dict(method='planner_in_window_tile',source_id='a',frame_time_ms=500,subjects=tags,labels=[]))])


@pytest.mark.parametrize('tag',list(semantics.SUBJECTS))
def test_single_subject_is_a_complete_sentence_not_a_bare_line(tag):
    text=voiceover.intro_text(subject_spec([tag]))
    assert text.startswith('Come explore ')
    assert semantics.SUBJECTS[tag] in text
    assert '\n' not in text and text.count('!')==1 and len(text)<90


def test_natural_pair_names_only_two_distinctive_retained_subjects():
    spec=subject_spec(['parade','market','family_activity'])
    assert voiceover.intro_text(spec)=='Come explore the parade and market!'
    assert semantics.intro_v2(subject_spec(['market']))=='Market.\nThe moments that stood out.'
    assert voiceover.intro_text(subject_spec(['market']))=='Come explore the market with us!'


def test_one_distinctive_subject_does_not_pad_narration_with_generic_family():
    assert voiceover.intro_text(subject_spec(['market','family_walk','family_activity']))=='Come explore the market with us!'


def place_words(text="We're in Arizona.",*,sid='a'):
    return [dict(source_id=sid,text=t,start_ms=100+i*150,end_ms=220+i*150) for i,t in enumerate(text.split())]


def test_place_requires_positive_retained_source_assertion_not_goal_or_filename(tmp_path):
    import json
    spec=subject_spec(['parade','market']);spec['goal']='Arizona trip'
    assert 'Arizona' not in voiceover.intro_text(spec)
    words=place_words();(tmp_path/'transcript.json').write_text(json.dumps(dict(words=words)))
    semantics.attach_place(spec,tmp_path)
    assert voiceover.intro_text(spec)=="Come explore Arizona's parade and market!"
    semantics.validate_place(spec,root=tmp_path)
    words[-1]['text']='California.'
    (tmp_path/'transcript.json').write_text(json.dumps(dict(words=words)))
    with pytest.raises(ValueError):semantics.validate_place(spec,root=tmp_path)


@pytest.mark.parametrize('text',["We're not in Arizona.","We are never in Arizona.","If we are in Arizona.","They said we're in Arizona.","Maybe welcome to Arizona."])
def test_negated_conditional_or_reported_place_is_not_trip_location(tmp_path,text):
    import json
    spec=subject_spec(['market'])
    (tmp_path/'transcript.json').write_text(json.dumps(dict(words=place_words(text))))
    semantics.attach_place(spec,tmp_path)
    assert 'narration_place' not in spec and 'Arizona' not in voiceover.intro_text(spec)


@pytest.mark.parametrize('case',['other_source','outside','screen','gap'])
def test_place_assertion_must_be_contiguous_retained_non_screen_speech(tmp_path,case):
    import json
    spec=subject_spec(['market']);words=place_words()
    if case=='other_source':words=place_words(sid='other')
    if case=='outside':spec['segments'][0]['raw_end_ms']=400
    if case=='screen':spec['segments'][0]['semantic_evidence']['labels']=['screen']
    if case=='gap':words[-1].update(start_ms=1500,end_ms=1700);spec['segments'][0]['raw_end_ms']=2000
    (tmp_path/'transcript.json').write_text(json.dumps(dict(words=words)))
    semantics.attach_place(spec,tmp_path)
    assert 'narration_place' not in spec


def test_complex_grounded_place_hook_stays_compact():
    spec=subject_spec(['night_parade','street_performers'])
    spec['narration_place']=dict(name='Arizona',method='retained_transcript_assertion')
    text=voiceover.intro_text(spec)
    assert len(text.split())<=8 and 'Arizona' in text and 'night parade' in text and 'street performers' in text
