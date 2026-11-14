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
    spec['segments']=[selection('a',0,3000,'one'),selection('b',0,6000,'two')]
    spec['target_duration_ms']=9000
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
        assert c['y']>=1920*.12 and c['y']+c['height']+8<=1920/3
        assert overlays.readable_frames(c['start_frame'],c['end_frame'],c['cut_frames'])>=75
        assert not c['start_frame']<=90<c['end_frame'] or 90 in c['cut_frames']


def test_never_maps_from_goal_or_filenames_alone(tmp_path,monkeypatch):
    _,spec,_,_=setup_plan(tmp_path,monkeypatch,False)
    assert [c['kind'] for c in spec['overlays']['cards']]==['title']
    assert len([x for x in spec['overlays']['limitations'] if 'omitted' in x])==2


def test_wrap_and_face_placement():
    top=overlays.layout('1. '+('Long place name '*10),1080,1920)
    assert len(top['text'].splitlines())<=2 and all(len(s)<=42 for s in top['text'].splitlines())
    lower=overlays.choose_position('Sedona with Kids',1080,1920,[[0,300,1080,400]])
    assert lower['position']=='top'
    assert overlays.choose_position('Sedona',1080,1920,[])['position']=='top'


def test_title_bridges_short_shot_with_explicit_cut_clearance(tmp_path,monkeypatch):
    _,spec,analysis,_=setup_plan(tmp_path,monkeypatch)
    spec['segments'][0].update(raw_end_ms=500,render_end_ms=None,render_start_ms=None)
    spec['target_duration_ms']=5000
    timeline,amended=assembler.build_timeline(spec,analysis)
    spec=overlays.compile_cards(amended,analysis,timeline,tmp_path)
    assert spec['overlays']['cards'][0]['kind']=='title' and 15 in spec['overlays']['cards'][0]['cut_frames']


@pytest.mark.parametrize('limited_filters',[False,True])
def test_real_burn_and_receipt_detects_mutation(tmp_path,monkeypatch,limited_filters):
    _,spec,_,_=setup_plan(tmp_path,monkeypatch)
    source=tmp_path/'source.mp4';output=tmp_path/'finished.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=1080x1920:r=30:d=9',
        '-f','lavfi','-i','sine=frequency=500:duration=9','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(source)],check=True,capture_output=True)
    if limited_filters:
        import sys,shutil
        from pipelines.ffmpeg import render
        wrapper=tmp_path/'ffmpeg-no-text'
        wrapper.write_text(f"#!{sys.executable}\n"+"import os,sys\n"+
            "if '-filters' in sys.argv:\n print(' TSC overlay VV->V\\n TS. fade V->V');sys.exit(0)\n"+
            "if any(x in ' '.join(sys.argv) for x in ('drawtext=', 'subtitles=', 'ass=')):\n sys.exit(99)\n"+
            f"os.execv({shutil.which('ffmpeg')!r},['ffmpeg']+sys.argv[1:])\n")
        wrapper.chmod(0o755)
        base=render._ffmpeg()
        monkeypatch.setattr(render,'_ffmpeg',lambda:[str(wrapper)]+base[1:])
        assert overlays.preflight()=='png'
    overlays.burn(source,output,spec['overlays'])
    assert all(c['status']=='pass' for c in overlays.verify(spec,output,[3000],9000))
    from PIL import Image
    image=tmp_path/'frame.png'
    subprocess.run(['ffmpeg','-v','error','-y','-ss','0.8','-i',str(output),'-frames:v','1',str(image)],check=True,capture_output=True)
    pixels=Image.open(image).convert('RGB')
    # Actual white glyphs occur in the safe top band, not just planned JSON.
    white=sum(r>220 and g>220 and b>220 for r,g,b in pixels.crop((0,230,1080,700)).getdata())
    assert white>300
    # Transition frame has neither white text nor a lingering dark shadow.
    subprocess.run(['ffmpeg','-v','error','-y','-ss','3.0','-i',str(output),'-frames:v','1',str(image)],check=True,capture_output=True)
    cut=Image.open(image).convert('RGB').crop((0,230,1080,700))
    assert sum(b<100 for r,g,b in cut.getdata())==0
    subprocess.run(['ffmpeg','-v','error','-y','-ss','5.0','-i',str(output),'-frames:v','1',str(image)],check=True,capture_output=True)
    later=Image.open(image).convert('RGB').crop((0,230,1080,700))
    assert sum(r>220 and g>220 and b>220 for r,g,b in later.getdata())>300
    spec['overlays']['cards'][0]['y']=0
    checks={c['name']:c['status'] for c in overlays.verify(spec,output,[3000],9000)}
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


def test_numbered_goal_list_enables_without_extra_flag(tmp_path):
    ctx=EpisodeContext('ep',tmp_path,'Sedona with Kids\n1. Halloween in Uptown\n2. Hilton Bell Rock')
    title,sections=overlays.requested(ctx)
    assert title=='Sedona with Kids' and len(sections)==2 and overlays.active(ctx)


def test_png_is_literal_transparent_and_contains_shadow(tmp_path):
    from PIL import Image
    card=overlays.layout("Kids: 100% %{n} / Mom's trip",1080,1920)
    path=tmp_path/'card.png';overlays.card_png(card,path)
    with Image.open(path) as image:
        assert image.mode=='RGBA' and image.size==(card['width'],card['height'])
        pixels=list(image.getdata())
        assert any(a==0 for r,g,b,a in pixels)
        assert any((r,g,b,a)==(255,255,255,255) for r,g,b,a in pixels)
        assert any(r==0 and g==0 and b==0 and a>=153 for r,g,b,a in pixels)


def test_missing_compositing_filter_is_typed_before_render(monkeypatch):
    from types import SimpleNamespace
    from pipelines.ffmpeg import render
    from pipelines.ffmpeg.contracts import StageError
    monkeypatch.setattr(render,'_run',lambda args:SimpleNamespace(stdout=' TS. fade V->V',stderr=''))
    with pytest.raises(StageError) as exc:overlays.preflight()
    assert exc.value.code=='OVERLAY_FILTERS' and exc.value.stage=='assembler'
    assert 'overlay' in str(exc.value)

@pytest.mark.parametrize('flags',['TS','TS.','T|'])
def test_ffmpeg_eight_nine_two_flag_filter_listing(monkeypatch,flags):
    from types import SimpleNamespace
    from pipelines.ffmpeg import render
    listing=f'Filters:\n {flags} fade V->V\n {flags} overlay VV->V\n'
    monkeypatch.setattr(render,'_run',lambda args:SimpleNamespace(stdout=listing,stderr=''))
    assert overlays.preflight()=='png'


def test_media_binary_overrides_and_path_fallback(monkeypatch):
    from pipelines.ffmpeg.media_tools import media_binary
    from pipelines.ffmpeg import render,proxy
    from types import SimpleNamespace
    monkeypatch.setenv('FRAMEFOUNDRY_FFMPEG','/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg')
    monkeypatch.setenv('FRAMEFOUNDRY_FFPROBE','/opt/homebrew/opt/ffmpeg-full/bin/ffprobe')
    assert render._ffmpeg()[0]==media_binary('ffmpeg')=='/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg'
    calls=[]
    monkeypatch.setattr(render,'_run',lambda args:(calls.append(args) or SimpleNamespace(stdout='{}')))
    render.probe_media('clip.mp4')
    assert calls[0][0]=='/opt/homebrew/opt/ffmpeg-full/bin/ffprobe'
    monkeypatch.delenv('FRAMEFOUNDRY_FFMPEG');monkeypatch.delenv('FRAMEFOUNDRY_FFPROBE')
    assert media_binary('ffmpeg')=='ffmpeg' and media_binary('ffprobe')=='ffprobe'


def test_cap_pill_upper_third_and_legibility_failures(tmp_path,monkeypatch):
    _,spec,_,_=setup_plan(tmp_path,monkeypatch)
    card=spec['overlays']['cards'][0]
    assert card['cap_height']>=116 and card['pill_opacity']==.6 and card['padding']==32
    assert card['y']>=1920*.12 and card['y']+card['height']+8<=1920/3
    render=tmp_path/'render.mp4';render.write_bytes(b'test receipt')
    plan=spec['overlays'];plan['render_sha256']=overlays.filehash(render)
    for key,value in [('cap_height',40),('pill_opacity',.4)]:
        altered=deepcopy(spec);altered['overlays']['cards'][0][key]=value
        results={c['name']:c['status'] for c in overlays.verify(altered,render,[3000],9000)}
        assert results['overlay_legibility']=='fail'
    assert (card['end_frame']-card['start_frame'])/30<=3.5


def test_section_label_never_bridges_ungrounded_clip(tmp_path,monkeypatch):
    _,spec,analysis,_=setup_plan(tmp_path,monkeypatch)
    spec['segments']=[selection('a',0,4000,'one'),selection('b',0,2000,'two'),selection('a',5000,9000,'three')]
    spec['target_duration_ms']=10000
    timeline,amended=assembler.build_timeline(spec,analysis)
    compiled=overlays.compile_cards(amended,analysis,timeline,tmp_path)
    assert [c['kind'] for c in compiled['overlays']['cards']]==['title']


def test_large_landscape_title_uses_fitting_upper_third_layout():
    card=overlays.layout('Things to do in Sedona with Kids',1920,1080)
    assert card['cap_height']>=65 and card['y']+card['height']+8<=360


def test_first_grounded_short_intervals_show_both_section_cards(tmp_path,monkeypatch):
    _,spec,analysis,_=setup_plan(tmp_path,monkeypatch,False)
    spec['segments']=[selection('a',0,4000,'opening'),
        selection('b',0,3066,'halloween'),selection('a',5000,8066,'resort'),
        selection('b',6000,9000,'outro')]
    spec['target_duration_ms']=13132
    analysis['words']=[dict(source_id='b',text='costumes',start_ms=1000,end_ms=1100,confidence=1),
        dict(source_id='a',text='pool',start_ms=5500,end_ms=5600,confidence=1)]
    timeline,amended=assembler.build_timeline(spec,analysis)
    compiled=overlays.compile_cards(amended,analysis,timeline,tmp_path)
    cards=compiled['overlays']['cards']
    assert [c['kind'] for c in cards]==['title','section','section']
    assert '1. Halloween' in cards[1]['text'] and '2. Hilton' in cards[2]['text']
    assert [c['shot_id'] for c in cards[1:]]==['shot-02','shot-03']
    assert not any('omitted:' in x for x in compiled['overlays']['limitations'])
    assert all(overlays.readable_frames(c['start_frame'],c['end_frame'],c['cut_frames'])>=75 for c in cards)
    # Previously both slots failed: masks in their fades were subtracted twice.
    assert all((c['end_frame']-c['start_frame']-len(c['cut_frames']))/30-.5<2.5 for c in cards[1:])
    for c in cards[1:]:
        assert c['start_frame'] in c['cut_frames'] and c['end_frame']-1 in c['cut_frames']
    receipt=tmp_path/'receipt.mp4';receipt.write_bytes(b'test-only')
    compiled['overlays']['render_sha256']=overlays.filehash(receipt)
    checks=overlays.verify(compiled,receipt,[4000,212/30*1000,304/30*1000],394/30*1000)
    assert all(c['status']=='pass' for c in checks)
    # Real rendered pixels for BOTH short section cards; generated media is test-only.
    from PIL import Image
    source=tmp_path/'blue.mp4';output=tmp_path/'two-sections.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
        'color=c=blue:s=1080x1920:r=30:d=13.14','-f','lavfi','-i','sine=frequency=500:duration=13.14','-frames:v','394',
        '-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(source)],check=True,capture_output=True)
    overlays.burn(source,output,compiled['overlays'])
    for t in (5.0,8.2):
        frame=tmp_path/'section.png'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(output),
            '-frames:v','1',str(frame)],check=True,capture_output=True)
        with Image.open(frame) as image:
            assert sum(r>220 and g>220 and b>220 for r,g,b in image.convert('RGB').crop((0,230,1080,700)).getdata())>300
    for cut in (120,212,304):
        for n in (cut-1,cut,cut+1):
            frame=tmp_path/'cut.png'
            subprocess.run(['ffmpeg','-v','error','-y','-ss',str(n/30),'-i',str(output),
                '-frames:v','1',str(frame)],check=True,capture_output=True)
            with Image.open(frame) as image:
                assert sum(b<130 for r,g,b in image.convert('RGB').crop((0,230,1080,700)).getdata())==0
    assert all(c['status']=='pass' for c in overlays.verify(compiled,output,[4000,212/30*1000,304/30*1000],394/30*1000))


def test_later_reserved_card_does_not_hide_earlier_grounded_slot(tmp_path,monkeypatch):
    _,spec,analysis,_=setup_plan(tmp_path,monkeypatch,False)
    spec['segments']=[selection('a',0,4000,'opening'),selection('b',0,4000,'resort'),
                      selection('a',5000,9000,'mixed')]
    spec['target_duration_ms']=12000
    analysis['words']=[dict(source_id='b',text='pool',start_ms=1000,end_ms=1100,confidence=1),
        dict(source_id='a',text='costumes pool',start_ms=5500,end_ms=5600,confidence=1)]
    timeline,amended=assembler.build_timeline(spec,analysis)
    cards=overlays.compile_cards(amended,analysis,timeline,tmp_path)['overlays']['cards']
    assert len(cards)==3
    assert cards[1]['shot_id']=='shot-03' and cards[2]['shot_id']=='shot-02'
    assert cards[2]['end_frame']<=cards[1]['start_frame']


def test_only_stationary_cut_masks_reduce_readable_hold():
    assert overlays.readable_frames(0,91,[0,1,89,90])==75
    assert overlays.readable_frames(0,91,[44,45,46])==72
    assert list(overlays.free_intervals(0,240,[(150,241),(0,30)]))==[(31,150)]


def test_recompiled_overlay_removes_only_its_stale_coverage_omission(tmp_path,monkeypatch):
    _,spec,analysis,timeline=setup_plan(tmp_path,monkeypatch)
    old='Requested section 2 omitted: no grounded cut-safe readable interval'
    spec['overlays']['limitations'].append(old)
    spec['coverage']['limitations'] += [old,'Independent source coverage remains partial']
    amended=overlays.compile_cards(spec,analysis,timeline,tmp_path)
    assert old not in amended['coverage']['limitations']
    assert 'Independent source coverage remains partial' in amended['coverage']['limitations']
    assert amended['coverage']['status']=='partial'
