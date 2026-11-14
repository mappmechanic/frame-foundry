"""Crop-aware content analysis and encoded-time mapping, with legacy compatibility."""
from pipelines.ffmpeg.eval_gates import content_box, read_shots
from pipelines.ffmpeg.assembler import build_timeline
from tests.test_qa_render_regressions import plan
import opentimelineio as otio


def test_portrait_crop_analyzes_whole_image_not_old_letterbox_band():
    box=content_box(1080,1920,1920,1080,framing_mode='center_crop')
    assert box==(10,19,1070,1901)
    legacy=content_box(1080,1920,1920,1080)
    assert legacy[3]-legacy[1]<650
    assert box[3]-box[1]>1800


def test_saved_timeline_marks_crop_and_readback_uses_frame_grid(tmp_path):
    spec,analysis=plan(tmp_path)
    spec['aspect_ratio']='9:16'
    timeline,_=build_timeline(spec,analysis)
    otio.adapters.write_to_file(timeline,str(tmp_path/'timeline.otio'))
    (tmp_path/'source.mp4').touch()
    shots,aspect=read_shots(tmp_path)
    assert aspect=='9:16' and all(s['framing_mode']=='center_crop' for s in shots)
    assert [s['timeline_start_ms'] for s in shots]==[0,333,667]
    assert [s['timeline_end_ms'] for s in shots]==[333,667,1000]
    assert [s['source_end_ms']-s['source_start_ms'] for s in shots]==[337,337,337]
    del timeline.metadata['framing_mode']
    otio.adapters.write_to_file(timeline,str(tmp_path/'timeline.otio'))
    assert all(s['framing_mode']=='fit_pad' for s in read_shots(tmp_path)[0])
