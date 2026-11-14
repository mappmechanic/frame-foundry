"""Real synthetic-media rendering tests; no checked-in media or planner dependency."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import opentimelineio as otio
import pytest

from pipelines.ffmpeg.assembler import assemble, build_timeline
from pipelines.ffmpeg.contracts import EpisodeContext, StageError
from pipelines.ffmpeg import render


def generate(path, color, *, audio=True, delayed=False):
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1',
            '-filter_threads', '1', '-f', 'lavfi', '-i', f'color=c={color}:s=160x90:r=30:d=3']
    if audio:
        if delayed:
            args += ['-itsoffset', '0.5']
        args += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=3']
    args += ['-t', '3', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-threads', '1']
    if audio:
        args += ['-c:a', 'aac']
    subprocess.run(args + [str(path)], check=True, capture_output=True, timeout=60)


def inputs(tmp_path, aspect, *, audio=True, delayed=False):
    source = tmp_path / 'original.mp4'
    generate(source, 'red', audio=audio, delayed=delayed)
    descriptor = dict(source_id='original', path=str(source), duration_ms=3000, fps_num=30, fps_den=1,
                      width=160, height=90, has_audio=audio, creation_time=None,
                      proxy_path=str(tmp_path / 'NONEXISTENT-analysis-proxy.mp4'), proxy_duration_ms=3000,
                      proxy_time_offset_ms=0, rotation_degrees=0, keyframes=[])
    segments = [dict(segment_id='s1', source_id='original', raw_start_ms=0, raw_end_ms=1000,
                     render_start_ms=None, render_end_ms=None, reason='Red opening'),
                dict(segment_id='s2', source_id='original', raw_start_ms=2000, raw_end_ms=3000,
                     render_start_ms=None, render_end_ms=None, reason='Red closing')]
    spec = dict(schema_version='1.0', episode_id='episode', goal='teaser', planner_mode='deterministic',
                interpretation='Source highlights', assumptions=[], target_duration_ms=5000,
                aspect_ratio=aspect, sources=[descriptor], segments=segments,
                coverage=dict(status='full', limitations=[]), exclusions=[])
    analysis = dict(episode_id='episode', goal='teaser', sources=deepcopy(spec['sources']), words=[])
    return spec, analysis


def loudness(path):
    stderr = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-threads', '1', '-i', str(path),
        '-vn', '-af', 'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'],
        capture_output=True, text=True, check=True, timeout=60).stderr
    return json.loads(stderr[stderr.rfind('{'):stderr.rfind('}') + 1])


@pytest.mark.parametrize('aspect,dimensions', [('9:16', (1080, 1920)), ('16:9', (1920, 1080))])
def test_real_render_both_aspects_duration_normalization_and_fit(tmp_path, aspect, dimensions):
    spec, analysis = inputs(tmp_path, aspect)
    result = assemble(EpisodeContext('episode', tmp_path, 'teaser'), spec, analysis)
    info = render.probe_media(result.candidate_path)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in info['streams'] if s['codec_type'] == 'audio')
    assert (video['width'], video['height']) == dimensions
    assert video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p'
    assert video['avg_frame_rate'] == '30/1'
    assert audio['codec_name'] == 'aac' and audio['sample_rate'] == '48000'
    assert result.duration_ms == pytest.approx(2000, abs=70)
    assert result.cut_times_ms == [1000]
    assert result.limitations
    stats = loudness(result.candidate_path)
    assert -15 <= float(stats['input_i']) <= -13
    assert float(stats['input_tp']) <= -1
    # Pixel checks prove proportional letterboxing, not a crop or stretch.
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(result.candidate_path),
                          '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'],
                         capture_output=True, check=True, timeout=30).stdout
    w, h = dimensions
    def pixel(x, y):
        return raw[(y*w+x)*3:(y*w+x)*3+3]
    if aspect == '9:16':
        assert max(pixel(w//2, 0)) < 10
    else:
        assert pixel(w//2, 0)[0] > 200
    assert pixel(w//2, h//2)[0] > 200
    assert pixel(w//2, h//2)[1] < 20


def test_no_audio_is_silent_but_continuous(tmp_path):
    spec, analysis = inputs(tmp_path, '16:9', audio=False)
    result = assemble(EpisodeContext('episode', tmp_path, 'teaser'), spec, analysis)
    info = render.probe_media(result.candidate_path)
    assert any(s['codec_type'] == 'audio' for s in info['streams'])
    assert result.duration_ms == pytest.approx(2000, abs=70)
    assert loudness(result.candidate_path)['input_i'] == '-inf'


def test_audio_timestamp_alignment_preserves_delayed_onset(tmp_path):
    spec, analysis = inputs(tmp_path, '16:9', delayed=True)
    result = assemble(EpisodeContext('episode', tmp_path, 'teaser'), spec, analysis)
    import array
    audio = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(result.candidate_path), '-vn',
                            '-f', 'f32le', '-ac', '1', '-ar', '48000', '-'],
                           capture_output=True, check=True, timeout=30).stdout
    samples = array.array('f', audio)
    def rms(start, end):
        chunk = samples[int(start*48000):int(end*48000)]
        return (sum(x*x for x in chunk) / len(chunk))**.5
    assert rms(.05, .3) < .0001
    assert rms(.65, .85) > .01


def test_missing_original_never_falls_back_to_proxy(tmp_path):
    spec, analysis = inputs(tmp_path, '9:16')
    Path(spec['sources'][0]['path']).unlink()
    timeline, _ = build_timeline(spec, analysis)
    with pytest.raises(StageError, match='Original source does not exist'):
        render.render_timeline(timeline, tmp_path / 'candidate.mp4', episode_dir=tmp_path)


def test_real_otio_readback_controls_source_order(tmp_path):
    spec, analysis = inputs(tmp_path, '16:9', audio=False)
    blue = tmp_path / 'blue.mp4'
    generate(blue, 'blue', audio=False)
    second = dict(spec['sources'][0], source_id='blue', path=str(blue))
    spec['sources'].append(second)
    analysis['sources'] = deepcopy(spec['sources'])
    spec['segments'][1].update(source_id='blue', raw_start_ms=0, raw_end_ms=1000)
    timeline, _ = build_timeline(spec, analysis)
    # Reorder the saved document, to prove rendering honors readback order.
    for track in timeline.tracks:
        first = track.pop(0)
        track.append(first)
    saved_path = tmp_path / 'edited.otio'
    otio.adapters.write_to_file(timeline, str(saved_path))
    saved = otio.adapters.read_from_file(str(saved_path))
    candidate = tmp_path / 'candidate.mp4'
    meta = render.render_timeline(saved, candidate, episode_dir=tmp_path)
    assert meta['cut_times_ms'] == [1000]
    for timestamp, expected in ((.2, 'blue'), (1.2, 'red')):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(timestamp), '-i', str(candidate),
                              '-frames:v', '1', '-vf', 'scale=1:1', '-f', 'rawvideo',
                              '-pix_fmt', 'rgb24', '-threads', '1', '-'],
                             capture_output=True, check=True, timeout=30).stdout
        assert raw[2 if expected == 'blue' else 0] > raw[0 if expected == 'blue' else 2] + 50


def test_subprocess_errors_timeout_and_argument_arrays(monkeypatch):
    def fail(args, **kwargs):
        assert isinstance(args, list) and not kwargs.get('shell', False)
        assert kwargs['timeout'] > 0
        return subprocess.CompletedProcess(args, 1, '', 'specific stderr failure')
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(StageError, match='specific stderr failure'):
        render._run(['ffmpeg', '-version'])
    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(StageError, match='could not complete'):
        render._run(['ffmpeg', '-version'])
