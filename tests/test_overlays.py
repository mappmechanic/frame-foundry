from copy import deepcopy
from pathlib import Path
import json
import subprocess
import pytest
from pipelines.ffmpeg import overlays,assembler
from pipelines.ffmpeg.contracts import EpisodeContext,validate_narrative
from tests.test_assembler import narrative,selection


def test_default_override_and_goal_list(tmp_path):
    ctx=EpisodeContext('test',tmp_path,'Things to do in Sedona with Kids')
    assert not overlays.active(ctx)
    ctx.overlay_sections=['Halloween in Uptown Sedona','Hilton Sedona Resort at Bell Rock']
    assert overlays.active(ctx)
    ctx.overlay_mode=False;assert not overlays.active(ctx)
    ctx.overlay_sections=[];ctx.overlay_mode=True;assert overlays.active(ctx)
    ctx.overlay_mode=None;ctx.goal='Sedona; Halloween in Uptown; Hilton Bell Rock';assert overlays.active(ctx)


def setup_plan(tmp_path,monkeypatch,words=True):
    spec,analysis=narrative(tmp_path)
    spec['segments']=[selection('a',0,3000,'one'),selection('b',0,3000,'two')]
    spec['target_duration_ms']=6000
    ctx=EpisodeContext('episode',tmp_path,'Things to do in Sedona with Kids',overlay_mode=True,
                       overlay_sections=['Halloween in Uptown Sedona','Hilton Sedona Resort at Bell Rock'])
    spec['goal']=analysis['goal']=ctx.goal
    if words:analysis['words']=[dict(source_id='b',text='pool',start_ms=1000,end_ms=1100,confidence=1)]
    overlays.attach(ctx,spec)
    monkeypatch.setattr(overlays,'face_boxes',lambda *a:None)
    timeline,amended=assembler.build_timeline(spec,analysis)
    compiled=overlays.compile_cards(amended,analysis,timeline,tmp_path)
    return ctx,compiled,analysis,timeline


def test_grounded_fallback_omits_unsupported_section_and_fits_cuts(tmp_path,monkeypatch):
    _,spec,_,_=setup_plan(tmp_path,monkeypatch)
    validate_narrative(spec)
    cards=spec['overlays']['cards']
    assert len(cards)==2 and cards[0]['kind']=='title'
    assert '2. Hilton' in cards[1]['text'] and cards[1]['evidence']['kind']=='retained_transcript'
    assert any('section 1 omitted' in x for x in spec['overlays']['limitations'])
    for c in cards:
        assert c['y']>=1920*.12 and c['y']+c['height']+8<=1920*.8
        assert (c['end_frame']-c['start_frame'])/30-.35>=1.2
        assert not c['start_frame']<=90<c['end_frame']


def test_never_maps_from_goal_or_filenames_alone(tmp_path,monkeypatch):
    _,spec,_,_=setup_plan(tmp_path,monkeypatch,False)
    assert [c['kind'] for c in spec['overlays']['cards']]==['title']
    assert len([x for x in spec['overlays']['limitations'] if 'omitted' in x])==2


def test_wrap_and_face_placement():
    top=overlays.layout('1. '+('Long place name '*10),1080,1920)
    assert len(top['text'].splitlines())<=2 and all(len(s)<=42 for s in top['text'].splitlines())
    lower=overlays.choose_position('Sedona with Kids',1080,1920,[[0,300,1080,400]])
    assert lower['position']=='lower'
    assert overlays.choose_position('Sedona',1080,1920,[])['position']=='top'


def test_short_shot_omits_instead_of_spanning_cuts(tmp_path,monkeypatch):
    _,spec,analysis,_=setup_plan(tmp_path,monkeypatch)
    spec['segments'][0].update(raw_end_ms=500,render_end_ms=None,render_start_ms=None)
    spec['target_duration_ms']=5000
    timeline,amended=assembler.build_timeline(spec,analysis)
    spec=overlays.compile_cards(amended,analysis,timeline,tmp_path)
    assert all(c['kind']!='title' for c in spec['overlays']['cards'])


def test_real_burn_and_receipt_detects_mutation(tmp_path,monkeypatch):
    _,spec,_,_=setup_plan(tmp_path,monkeypatch)
    source=tmp_path/'source.mp4';output=tmp_path/'finished.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=1080x1920:r=30:d=6',
        '-f','lavfi','-i','sine=frequency=500:duration=6','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(source)],check=True,capture_output=True)
    overlays.burn(source,output,spec['overlays'])
    assert all(c['status']=='pass' for c in overlays.verify(spec,output,[3000],6000))
    from PIL import Image
    image=tmp_path/'frame.png'
    subprocess.run(['ffmpeg','-v','error','-y','-ss','0.8','-i',str(output),'-frames:v','1',str(image)],check=True,capture_output=True)
    pixels=Image.open(image).convert('RGB')
    # Actual white glyphs occur in the safe top band, not just planned JSON.
    white=sum(r>220 and g>220 and b>220 for r,g,b in pixels.crop((0,230,1080,700)).getdata())
    assert white>300
    spec['overlays']['cards'][0]['y']=0
    checks={c['name']:c['status'] for c in overlays.verify(spec,output,[3000],6000)}
    assert checks['overlay_safe_area']=='fail' and checks['overlay_render_binding']=='fail'


def test_drawtext_special_text_is_literal(tmp_path):
    c=overlays.layout("Kids: 100% %{n} / Mom's trip",1080,1920)
    assert '%{n}' in c['text'] and "Mom's" in c['text']


def test_model_section_assignment_has_in_window_frame_proof_only(tmp_path):
    ctx=EpisodeContext('ep',tmp_path,'Sedona',overlay_mode=True,overlay_sections=['Hilton Bell Rock Resort'])
    data=dict(segments=[dict(source_id='s',raw_start_ms=0,raw_end_ms=2000,overlay_section=0,overlay_evidence='pool')])
    rows=[dict(source_id='s',frame_in_window=True,tile_index=0,frame_time_ms=1000,frame_path='pool.jpg',frame_sha256='a'*64)]
    spec={};overlays.attach(ctx,spec,data,rows)
    assert len(spec['overlays']['proofs'])==1
    rows[0]['frame_time_ms']=3000;overlays.attach(ctx,spec,data,rows)
    assert spec['overlays']['proofs']==[]


def test_control_text_and_too_many_sections_rejected(tmp_path):
    with pytest.raises(Exception,match='Control'):overlays.clean('Kids\x00hidden')
    ctx=EpisodeContext('ep',tmp_path,'Sedona',overlay_sections=['x']*9)
    with pytest.raises(Exception,match='eight'):overlays.requested(ctx)
