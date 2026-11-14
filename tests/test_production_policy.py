from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from pipelines.ffmpeg import production,assembler,overlays
from pipelines.ffmpeg.contracts import EpisodeContext,validate_narrative
from tests.test_assembler import narrative,selection

def fixture(tmp_path):
    spec,a=narrative(tmp_path);spec['segments']=[selection('a',0,7000,'a'),selection('b',0,7000,'b')];spec['target_duration_ms']=14000
    return EpisodeContext('episode',tmp_path,'teaser',production=True),spec,a

def evidence(tmp_path,segment,*,frame=1000,labels=('action',)):
    p=tmp_path/'hook.jpg';p.write_bytes(b'test-only-proof')
    segment['semantic_evidence']=dict(method='planner_in_window_tile',candidate_id='a',source_id=segment['source_id'],source_start_ms=0,source_end_ms=10000,frame_time_ms=frame,frame_path='hook.jpg',frame_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),view='center_crop_9_16',subjects=[],labels=list(labels))

def test_legacy_byte_structure_unchanged(tmp_path):
    ctx,s,a=fixture(tmp_path);ctx.production=False
    assert production.apply(ctx,s,a) is s

def test_verified_hook_promoted_not_hallucinated(tmp_path):
    ctx,s,a=fixture(tmp_path);evidence(tmp_path,s['segments'][1])
    s['sources'][1]['keyframes']=[dict(path='hook.jpg',source_time_ms=1000,proxy_time_ms=1000)];a['sources']=deepcopy(s['sources'])
    p=production.apply(ctx,s,a);assert p['segments'][0]['source_id']=='b';assert p['editing_policy']['hook_status']=='verified';validate_narrative(p,a)
    s['segments'][1]['semantic_evidence']['frame_sha256']='f'*64
    assert production.apply(ctx,s,a)['editing_policy']['hook_status']=='unavailable'

def test_chronology_and_late_tile_not_misrepresented(tmp_path):
    ctx,s,a=fixture(tmp_path);s['goal']='chronological teaser';evidence(tmp_path,s['segments'][1])
    assert production.apply(ctx,s,a)['editing_policy']['hook_status']=='unavailable'
    s['goal']='teaser';evidence(tmp_path,s['segments'][1],frame=4000)
    assert production.apply(ctx,s,a)['editing_policy']['hook_status']=='unavailable'

def test_retakes_exact_final_complete_only(tmp_path):
    ctx,s,a=fixture(tmp_path);s['segments']=[selection('a',0,2500,'one'),selection('a',4000,6500,'two')]
    a['words']=[dict(source_id='a',text='Hi.',start_ms=1000,end_ms=1500),dict(source_id='a',text='Hi.',start_ms=5000,end_ms=5500)]
    p=production.apply(ctx,s,a);assert len(p['segments'])==1 and p['segments'][0]['raw_start_ms']==4000
    a['words'][1]['text']='Different.';assert len(production.apply(ctx,s,a)['segments'])==2

def test_silence_removal_preserves80_120_speech(tmp_path,monkeypatch):
    ctx,s,a=fixture(tmp_path);s['segments']=[selection('a',0,7000,'a')]
    a['words']=[dict(source_id='a',text='Hello',start_ms=500,end_ms=1500),dict(source_id='a',text='world.',start_ms=4500,end_ms=5000)]
    bins=[dict(start_ms=0,end_ms=7000,active=False)]
    monkeypatch.setattr(production,'_bins',lambda *args:bins)
    p=production.apply(ctx,s,a);assert [(r['raw_start_ms'],r['raw_end_ms']) for r in p['segments']]==[(0,1620),(4420,7000)]
    timeline,amended=assembler.build_timeline(p,a);assert len(timeline.tracks[0])==2
    bins[0]['active']=True;assert len(production.apply(ctx,s,a)['segments'])==1

def test_visual_split_not_coalesced_and_unknown_speech_preserved(tmp_path):
    ctx,s,a=fixture(tmp_path);a['transcription_status']='visual_only_unavailable'
    p=production.apply(ctx,s,a);timeline,_=assembler.build_timeline(p,a);assert len(timeline.tracks[0])==4
    a['transcription_status']='degraded';assert len(production.apply(ctx,s,a)['segments'])==2

def test_policy_receipt_idempotent_and_required_audio(tmp_path):
    ctx,s,a=fixture(tmp_path);p=production.apply(ctx,s,a);assert production.apply(ctx,p,a)==p
    assert next(r for r in production.verify(p,a) if r['name']=='policy_audio')['status']=='fail'
    p['editing_policy']['audio']={'status':'absent'};assert all(r['status']=='pass' for r in production.verify(p,a))

def test_production_safe_zone_and_legacy_layout():
    legacy=overlays.layout('Things to do in Sedona with Kids',1080,1920)
    p=overlays.layout('Things to do in Sedona with Kids',1080,1920,production=True)
    assert p['x']+p['width']<=1080*.85 and p['cap_height']>=116
    assert legacy['width']>p['width']

def test_dialogue_intervals_include_cached_vo_and_word_padding(tmp_path):
    ctx,s,a=fixture(tmp_path);s['segments']=[selection('a',0,7000,'a')];a['words']=[dict(source_id='a',text='Hi.',start_ms=1000,end_ms=1300)]
    t,_=assembler.build_timeline(s,a);s['voiceover']={'duration_ms':2990}
    assert production.dialogue_intervals(t,a,s)==[(.92,1.42),(0,2.99)]

def test_project_limit_can_only_lower_existing_cap(monkeypatch):
    from pipelines.ffmpeg.budget import project_limit_usd
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_API_BUDGET_USD','300');assert project_limit_usd()==300
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_API_BUDGET_USD','501')
    with pytest.raises(Exception):project_limit_usd()

def test_production_real_pipeline_preserves_audio_and_schema(tmp_path,monkeypatch):
    from tests.fixtures.generate_footage import generate_footage
    from pipelines.ffmpeg.cli import run_episode
    from pipelines.ffmpeg.contracts import read_json,validate_report
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(tmp_path/'budget'))
    inputs=tmp_path/'input';generate_footage(inputs)
    qa=run_episode(inputs,'Create a 10 second family-trip teaser',episodes_root=tmp_path/'runs',episode_id='production',transcript_fixture=inputs/'transcript-fixture.json',provider=None,production=True,overlays=True,reuse_history=False)
    validate_report(qa)
    assert qa['passed'],qa.get('error')
    spec=read_json(tmp_path/'runs/production/narrative-spec.json')
    assert spec['editing_policy']['audio']['status']=='absent'
    assert spec['overlays']['production'] is True
    assert all(c['status']=='pass' for c in qa['checks'] if c['name'].startswith('policy_'))
    assert (tmp_path/'runs/production/contact-sheet.jpg').is_file()
    index=read_json(tmp_path/'runs/production/contact-sheet-index.json');assert any(s['side']=='after_three' for s in index['samples'])
    assert qa['ai_usage']==[] and -15<=qa['integrated_lufs']<=-13 and qa['true_peak_dbtp']<=-1
