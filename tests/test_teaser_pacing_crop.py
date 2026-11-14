"""Approved center-fill/pacing contracts; synthetic FFmpeg, no models or detectors."""
from copy import deepcopy
import subprocess

import opentimelineio as otio
from PIL import Image
import pytest

from pipelines.ffmpeg.assembler import assemble, build_timeline
from pipelines.ffmpeg.contact_sheet import build_contact_sheet
from pipelines.ffmpeg.contracts import validate_narrative
from pipelines.ffmpeg.eval_judge import build_judge_sheet, SYSTEM as JUDGE_SYSTEM
from pipelines.ffmpeg.planner import plan_teaser, _pace_local_visual_response
from pipelines.ffmpeg.render import probe_media
from tests.test_planner import FakeProvider, analysis_for, context
from tests.test_render import loudness
from tests.test_visual_duration import visual_analysis, response_for


def make_source(path, *, size='160x90', duration=1, montage=False):
    video = f'color=red:s={size}:r=30:d={duration}'
    if montage:
        for color, start, end in [('blue', 2.5, 5), ('green', 5, 7.5), ('yellow', 7.5, 10)]:
            video += f",drawbox=x=0:y=0:w=iw:h=ih:color={color}:t=fill:enable='gte(t,{start})*lt(t,{end})'"
    elif size == '160x90':
        # Only the red center fits the portrait crop; side bands must disappear.
        video += ',drawbox=x=0:y=0:w=50:h=ih:color=blue:t=fill,drawbox=x=110:y=0:w=50:h=ih:color=green:t=fill'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', video,
                    '-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=48000:duration={duration}',
                    '-t', str(duration), '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                    '-threads', '1', '-c:a', 'aac', str(path)], check=True, capture_output=True, timeout=60)


def source_analysis(tmp_path, *, size='160x90', duration=1, montage=False):
    analysis = analysis_for('10 second family trip teaser', count=1, duration=duration * 1000)
    source = analysis['sources'][0]
    source['path'] = str(tmp_path / 'source.mp4')
    source['width'], source['height'] = map(int, size.split('x'))
    source['has_audio'] = True
    make_source(source['path'], size=size, duration=duration, montage=montage)
    return analysis


def frame(path, time=0):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(time), '-i', str(path), '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'],
                         capture_output=True, check=True, timeout=30).stdout
    stream = next(s for s in probe_media(path)['streams'] if s['codec_type'] == 'video')
    return Image.frombytes('RGB', (stream['width'], stream['height']), raw)


@pytest.mark.parametrize('aspect,size,dimensions', [
    ('9:16', '160x90', (1080, 1920)),
    ('16:9', '90x160', (1920, 1080)),
    ('16:9', '160x90', (1920, 1080)),
])
def test_real_center_crop_and_horizontal_fit_preserve_audio_cfr(tmp_path, aspect, size, dimensions):
    analysis = source_analysis(tmp_path, size=size)
    ctx = context(tmp_path, analysis)
    spec = plan_teaser(ctx, analysis, provider=None)
    assert spec['aspect_ratio'] == '9:16'  # default Goal contract
    spec['aspect_ratio'] = aspect
    result = assemble(ctx, spec, analysis)
    image = frame(result.candidate_path)
    assert image.size == dimensions
    w, h = dimensions
    assert image.getpixel((w // 2, h // 2))[0] > 200
    if aspect == '9:16':
        # Red at all four edges proves center-fill rather than black padding,
        # stretching, or left/right-biased cropping of the colored side bands.
        for point in [(w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]:
            red, green, blue = image.getpixel(point)
            assert red > 200 and green < 25 and blue < 25
    elif size == '90x160':
        assert max(image.getpixel((0, h // 2))) < 10
        assert max(image.getpixel((w - 1, h // 2))) < 10
    else:
        assert image.getpixel((0, h // 2))[2] > 200
        assert image.getpixel((w - 1, h // 2))[1] > 100
    streams = probe_media(result.candidate_path)['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    assert video['avg_frame_rate'] == '30/1' and int(video['nb_frames']) == 30
    assert audio['codec_name'] == 'aac' and audio['sample_rate'] == '48000'
    assert result.duration_ms == pytest.approx(1000, abs=70)
    stats = loudness(result.candidate_path)
    assert -15 <= float(stats['input_i']) <= -13
    assert float(stats['input_tp']) <= -1


def test_local_montage_pacing_survives_written_otio_and_real_frames(tmp_path):
    analysis = source_analysis(tmp_path, duration=10, montage=True)
    ctx = context(tmp_path, analysis)
    spec = plan_teaser(ctx, analysis, provider=None)
    assert [s['raw_end_ms'] - s['raw_start_ms'] for s in spec['segments']] == [2500] * 4
    assert [s['raw_start_ms'] for s in spec['segments']] == [0, 5000, 2500, 7500]
    result = assemble(ctx, spec, analysis)
    saved = otio.adapters.read_from_file(str(result.timeline_path))
    assert len(saved.tracks[0]) == 4  # not coalesced back into a ten-second shot
    assert result.cut_times_ms == [2500, 5000, 7500]
    assert result.duration_ms == pytest.approx(10000, abs=70)
    pixels = [frame(result.candidate_path, t).getpixel((540, 960)) for t in (1, 3.5, 6, 8.5)]
    assert pixels[0][0] > 200 and pixels[0][2] < 25  # red
    assert pixels[1][1] > 100 and pixels[1][0] < 25  # green
    assert pixels[2][2] > 200 and pixels[2][0] < 25  # blue
    assert pixels[3][0] > 200 and pixels[3][1] > 200  # yellow
    shots = [dict(shot_id=f'shot-{i + 1:02d}', timeline_start_ms=i * 2500,
                  timeline_end_ms=(i + 1) * 2500) for i in range(4)]
    sheet = build_judge_sheet(result.candidate_path, shots, tmp_path / 'judge.jpg', render_w=1080, render_h=1920)
    assert max(sheet['size']) <= 640 and sheet['tiles'] == 4
    with Image.open(sheet['path']) as image:
        tw = image.width // sheet['columns']
        th = image.height // ((4 + sheet['columns'] - 1) // sheet['columns']) - 18
        assert tw / th == pytest.approx(9 / 16, abs=.02)
        assert image.getpixel((tw // 2, 0))[0] > 200
        assert image.getpixel((tw // 2, th - 3))[0] > 200
        assert image.getpixel((tw // 2, th + 16))[0] < 30  # separate label strip
    evidence = build_contact_sheet(result.candidate_path, result.cut_times_ms, tmp_path / 'boundary.jpg')
    assert len(evidence['samples']) == 6
    with Image.open(evidence['path']) as image:
        assert image.size == (4 * 135, 2 * (240 + 36))


def test_visual_fill_is_paced_unique_and_preserves_model_choices(tmp_path):
    analysis = visual_analysis()
    response = response_for(analysis['candidates'][:1], 2000)
    provider = FakeProvider(response)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    chosen = response['segments'][0]
    assert all(spec['segments'][0][k] == v for k, v in chosen.items())
    assert sum(s['raw_end_ms'] - s['raw_start_ms'] for s in spec['segments']) == 45000
    additions = spec['segments'][1:]
    assert max(s['raw_end_ms'] - s['raw_start_ms'] for s in additions) <= 3000
    timeline, _ = build_timeline(spec, analysis)
    assert timeline.duration().to_seconds() == 45
    assert 2 <= 45 / len(timeline.tracks[0]) <= 3
    validate_narrative(spec, analysis)
    assert len(provider.calls) == 1 and len(provider.calls[0]['images']) <= 1
    prompt = provider.calls[0]['system']
    assert '2-3 second' in prompt and 'family trip' in prompt and '80ms' in prompt and '120ms' in prompt
    assert 'word-integrity gate' in JUDGE_SYSTEM and 'family-trip Goals' in JUDGE_SYSTEM


def test_genuine_speech_padding_overrides_pacing_without_reordering(tmp_path):
    analysis = source_analysis(tmp_path, duration=10)
    analysis['transcription_status'] = 'local_asr'
    analysis['words'] = [dict(source_id='source-0', text='continuous', start_ms=1000, end_ms=4500),
                         dict(source_id='source-0', text='phrase', start_ms=4600, end_ms=7000)]
    response = dict(interpretation='Protected phrase', assumptions=[],
                    segments=[dict(source_id='source-0', raw_start_ms=1000, raw_end_ms=4500, reason='Genuine speech')],
                    coverage=dict(status='partial', limitations=['Shorter than target']))
    # Use a source-backed candidate large enough for this phrase.
    analysis['candidates'] = [dict(candidate_id='phrase', source_id='source-0', start_ms=0, end_ms=10000,
                                  transcript_snippet='continuous phrase', quality={}, quality_method='fixture')]
    before = deepcopy(analysis)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=FakeProvider(response))
    assert spec['planner_mode'] == 'llm' and len(spec['segments']) == 1
    timeline, amended = build_timeline(spec, analysis)
    segment = amended['segments'][0]
    assert (segment['render_start_ms'], segment['render_end_ms']) == (920, 7120)
    assert timeline.duration().to_seconds() == pytest.approx(6.2)
    assert analysis == before  # no ASR provenance or word changes
    assert not any('duration completion' in a for a in spec['assumptions'])
    result = assemble(context(tmp_path, analysis), spec, analysis)
    saved = otio.adapters.read_from_file(str(result.timeline_path))
    clip = saved.tracks[0][0]
    assert clip.source_range.start_time.to_seconds() == pytest.approx(.92)
    assert clip.source_range.end_time_exclusive().to_seconds() == pytest.approx(7.12)
    assert result.duration_ms == pytest.approx(6200, abs=70) and result.cut_times_ms == []
    video = next(s for s in probe_media(result.candidate_path)['streams'] if s['codec_type'] == 'video')
    assert video['avg_frame_rate'] == '30/1' and int(video['nb_frames']) == 186


def test_single_source_two_shots_do_not_unnecessarily_coalesce(tmp_path):
    analysis = analysis_for('5 second teaser', count=1, duration=5000)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    timeline, _ = build_timeline(spec, analysis)
    assert len(timeline.tracks[0]) == 2
    assert timeline.duration().to_seconds() == 5
    assert all(clip.duration().to_seconds() == 2.5 for clip in timeline.tracks[0])


@pytest.mark.parametrize('count', range(2, 13))
def test_local_visual_order_does_not_strand_touching_final_pair(count):
    response = dict(segments=[dict(source_id='one', raw_start_ms=0,
        raw_end_ms=count * 2500, reason='Local visual window')], assumptions=[])
    paced = _pace_local_visual_response(dict(words=[]), response)
    segments = paced['segments']
    assert len(segments) == count
    assert sorted((s['raw_start_ms'], s['raw_end_ms']) for s in segments) == [
        (i * 2500, (i + 1) * 2500) for i in range(count)]
    assert all(a['raw_end_ms'] != b['raw_start_ms'] for a, b in zip(segments, segments[1:]))
