"""Isolated legacy QA/CFR/audio regressions; all media is temporary synthetic data."""
from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import opentimelineio as otio
import pytest

from pipelines.ffmpeg import qa_validator as qa, render
from pipelines.ffmpeg.assembler import assemble, build_timeline
from pipelines.ffmpeg.contracts import AssemblyResult, EpisodeContext, atomic_write_json


def plan(tmp_path, intervals=((2007, 2344), (3000, 3337), (4000, 4337))):
    source = tmp_path / 'source.mp4'
    descriptor = dict(source_id='original', path=str(source), duration_ms=6000,
                      fps_num=30, fps_den=1, width=160, height=90, has_audio=True,
                      creation_time=None, proxy_path=str(tmp_path / 'unused.mp4'),
                      proxy_duration_ms=6000, proxy_time_offset_ms=0,
                      rotation_degrees=0, keyframes=[])
    segments = [dict(segment_id=f's{i}', source_id='original', raw_start_ms=a,
                     raw_end_ms=b, render_start_ms=None, render_end_ms=None,
                     reason='Synthetic highlight') for i, (a, b) in enumerate(intervals)]
    spec = dict(schema_version='1.0', episode_id='regression', goal='5 second teaser',
                planner_mode='deterministic', interpretation='Synthetic highlights',
                assumptions=[], target_duration_ms=5000, aspect_ratio='16:9',
                sources=[descriptor], segments=segments,
                coverage=dict(status='partial', limitations=['Short synthetic selections']), exclusions=[])
    analysis = dict(episode_id=spec['episode_id'], goal=spec['goal'],
                    sources=deepcopy(spec['sources']), words=[])
    return spec, analysis


@pytest.fixture
def mocked_candidate(tmp_path, monkeypatch):
    spec, analysis = plan(tmp_path)
    (tmp_path / 'source.mp4').touch()
    timeline, spec = build_timeline(spec, analysis)
    path = tmp_path / 'timeline.otio'
    otio.adapters.write_to_file(timeline, str(path))
    result = AssemblyResult('regression', tmp_path / 'candidate.mp4', path,
                            tmp_path / 'narrative-spec.json', [333, 667], 1000)
    video = dict(codec_name='h264', width=1920, height=1080, time_base='1/15360',
                 avg_frame_rate='999999/33333', r_frame_rate='30/1')
    info = dict(video=video, data=dict(streams=[dict(codec_type='audio', codec_name='aac')]),
                duration_ms=1000)
    monkeypatch.setattr(qa, 'probe', lambda p: info)
    frames = [dict(best_effort_timestamp=i*512, pkt_duration=512) for i in range(30)]
    monkeypatch.setattr(qa, 'run', lambda args: SimpleNamespace(stdout=json.dumps(dict(frames=frames))))
    monkeypatch.setattr(qa, 'measure_audio', lambda p: dict(integrated_lufs=-14, true_peak_dbtp=-2))
    monkeypatch.setattr(qa, 'build_contact_sheet', lambda *a: None)
    ctx = EpisodeContext('regression', tmp_path, spec['goal'])
    return ctx, result, spec, info, frames


def validate(candidate):
    ctx, result, spec, _, _ = candidate
    return qa.validate_candidate(ctx, result, spec, judge_provider=None)


def gate(report, name):
    return next(c for c in report['checks'] if c['name'] == name)


def test_float_lower_boundary_and_cumulative_frame_rounding(mocked_candidate):
    report = validate(mocked_candidate)
    assert report['passed'], report['checks']
    assert gate(report, 'output_fps')['status'] == 'pass'  # fractional aggregate metadata ignored
    # This source start really round-trips above the integer millisecond boundary.
    ctx, result, spec, _, _ = mocked_candidate
    clip = otio.adapters.read_from_file(str(result.timeline_path)).tracks[0][0]
    assert clip.source_range.start_time.to_seconds()*1000 > spec['segments'][0]['render_start_ms']


@pytest.mark.parametrize('offset,passed', [(0.999, True), (1.001, False)])
def test_selection_lower_tolerance_is_bounded(mocked_candidate, offset, passed):
    _, result, _, _, _ = mocked_candidate
    timeline = otio.adapters.read_from_file(str(result.timeline_path))
    clip = timeline.tracks[0][0]
    old = clip.source_range
    clip.source_range = otio.opentime.TimeRange(
        otio.opentime.RationalTime(2007+offset, 1000), old.duration)
    otio.adapters.write_to_file(timeline, str(result.timeline_path))
    report = validate(mocked_candidate)
    assert (gate(report, 'timeline_and_padding')['status'] == 'pass') == passed
    if not passed:
        assert 'Selection s0' in gate(report, 'timeline_and_padding')['reason']


@pytest.mark.parametrize('mutation', ['order', 'source', 'cut', 'padding'])
def test_real_mapping_errors_are_not_hidden(mocked_candidate, mutation):
    ctx, result, spec, _, _ = mocked_candidate
    timeline = otio.adapters.read_from_file(str(result.timeline_path))
    if mutation == 'order':
        timeline.tracks[0].append(timeline.tracks[0].pop(0))
    elif mutation == 'source':
        timeline.tracks[0][0].media_reference.target_url = str(ctx.episode_dir / 'wrong.mp4')
    elif mutation == 'cut':
        result.cut_times_ms[1] += 2
    else:
        # Spec and clip agree, but the known word lacks 80ms/120ms protection.
        atomic_write_json(ctx.episode_dir / 'transcript.json', dict(words=[
            dict(source_id='original', start_ms=2020, end_ms=2310)]))
    otio.adapters.write_to_file(timeline, str(result.timeline_path))
    report = validate(mocked_candidate)
    assert gate(report, 'timeline_and_padding')['status'] == 'fail'
    assert gate(report, 'timeline_and_padding')['reason']


@pytest.mark.parametrize('kind', ['25fps', 'vfr', 'missing', 'duplicate', 'duration'])
def test_fps_failure_has_timestamp_reason_even_with_30fps_tags(mocked_candidate, kind):
    _, _, _, info, frames = mocked_candidate
    info['video'].update(avg_frame_rate='30/1', r_frame_rate='30/1')
    if kind == '25fps':
        for i, frame in enumerate(frames):
            frame['best_effort_timestamp'] = round(i*15360/25)
    elif kind == 'vfr':
        frames[5]['best_effort_timestamp'] += 256
    elif kind == 'missing':
        frames[5].pop('best_effort_timestamp')
    elif kind == 'duplicate':
        frames[5]['best_effort_timestamp'] = frames[4]['best_effort_timestamp']
    else:
        frames[-1]['pkt_duration'] = 1024
    report = validate(mocked_candidate)
    check = gate(report, 'output_fps')
    assert not report['passed'] and check['status'] == 'fail'
    assert check['reason'] and ('observed' in check['reason'] or 'Cannot verify' in check['reason'])


def test_failed_encoding_dimensions_audio_reasons_include_observations(mocked_candidate, monkeypatch):
    mocked_candidate[3]['video'].update(codec_name='hevc', width=640, height=360)
    mocked_candidate[3]['data']['streams'][0]['codec_name'] = 'opus'
    monkeypatch.setattr(qa, 'measure_audio', lambda p: dict(integrated_lufs=-20, true_peak_dbtp=-0.88))
    report = validate(mocked_candidate)
    for name in ('output_encoding', 'output_dimensions', 'loudness', 'true_peak'):
        check = gate(report, name)
        assert check['status'] == 'fail' and 'observed' in check['reason']
    assert '-0.88' in gate(report, 'true_peak')['reason']
    assert all(c['reason'] for c in report['checks'] if c['status'] == 'fail')


def test_bounded_loudnorm_retry_uses_original_and_preserves_headroom(tmp_path, monkeypatch):
    calls = []
    stats = dict(input_i='-20', input_tp='-3', input_lra='2', input_thresh='-30', target_offset='0')
    def run(args):
        calls.append(args)
        if str(tmp_path / 'candidate.mp4') in args and args[-1] == '-':
            return SimpleNamespace(stderr=json.dumps(dict(input_i='-14', input_tp='-0.88')))
        return SimpleNamespace(stderr=json.dumps(stats))
    monkeypatch.setattr(render, '_run', run)
    render._normalize(tmp_path / 'joined.mkv', tmp_path / 'candidate.mp4')
    encodes = [c for c in calls if c[-1] == str(tmp_path / 'candidate.mp4')]
    assert len(encodes) == 3
    assert all(c[c.index('-i')+1] == str(tmp_path / 'joined.mkv') for c in encodes)
    assert all('TP=-2' in c[c.index('-af')+1] for c in encodes)
    assert 'I=-13:' in encodes[-1][encodes[-1].index('-af')+1]
    assert 'alimiter=' in encodes[-1][encodes[-1].index('-af')+1]
    assert 'level=false:latency=true' in encodes[-1][encodes[-1].index('-af')+1]


@pytest.mark.parametrize('audio', ['sine=frequency=440:sample_rate=48000:duration=6',
    "aevalsrc=0.8*sin(2*PI*9000*t)*(0.12+0.88*lt(mod(t\\,0.2)\\,0.03)):s=48000:d=6"])
def test_real_fractional_cuts_cfr_and_final_encoded_true_peak(tmp_path, audio):
    spec, analysis = plan(tmp_path, ((201, 538), (1000, 1337), (2000, 2337), (3000, 3337), (4000, 4337)))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
        '-f', 'lavfi', '-i', 'color=c=red:s=160x90:r=30:d=6', '-f', 'lavfi', '-i', audio,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', '-c:a', 'pcm_s16le',
        '-f', 'matroska', spec['sources'][0]['path']], check=True, capture_output=True, timeout=60)
    ctx = EpisodeContext('regression', tmp_path, spec['goal'])
    result = assemble(ctx, spec, analysis)
    amended = json.loads(result.spec_path.read_text())
    report = qa.validate_candidate(ctx, result, amended, judge_provider=None)
    print('Encoded evidence:', report['integrated_lufs'], report['true_peak_dbtp'], report['checks'])
    assert result.cut_times_ms == [333, 667, 1000, 1333]
    assert report['passed'], report['checks']
    assert -15 <= report['integrated_lufs'] <= -13
    assert report['true_peak_dbtp'] <= -1
    assert (tmp_path / 'contact-sheet.jpg').is_file()


def test_real_sparse_high_crest_audio_keeps_peak_gate_and_rejects_lufs_undershoot(tmp_path):
    spec, analysis = plan(tmp_path, ((0, 6000),))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
        '-f', 'lavfi', '-i', 'color=c=red:s=160x90:r=30:d=6', '-f', 'lavfi', '-i',
        'aevalsrc=0.95*sin(2*PI*440*t)*(0.003+0.997*lt(mod(t\\,0.25)\\,0.01)):s=48000:d=6',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', '-c:a', 'pcm_s16le',
        '-f', 'matroska', spec['sources'][0]['path']], check=True, capture_output=True, timeout=60)
    ctx = EpisodeContext('regression', tmp_path, spec['goal'])
    result = assemble(ctx, spec, analysis)
    report = qa.validate_candidate(ctx, result, json.loads(result.spec_path.read_text()), judge_provider=None)
    print('Sparse encoded evidence:', report['integrated_lufs'], report['true_peak_dbtp'])
    assert report['true_peak_dbtp'] <= -1
    assert gate(report, 'true_peak')['status'] == 'pass'
    acceptable_lufs = -15 <= report['integrated_lufs'] <= -13
    assert (gate(report, 'loudness')['status'] == 'pass') == acceptable_lufs
    assert report['passed'] == acceptable_lufs  # Peak success must never waive LUFS.
    if not acceptable_lufs:
        assert 'observed' in gate(report, 'loudness')['reason']
    assert (tmp_path / 'contact-sheet.jpg').is_file()


def test_encoded_lufs_undershoot_gets_one_bounded_post_normalization_limiter(tmp_path,monkeypatch):
    calls=[];measurements=iter([-16.7,-15.59])
    original=tmp_path/'joined.mov';candidate=tmp_path/'candidate.mp4'
    stats=dict(input_i='-28',input_tp='-9',input_lra='4',input_thresh='-35',target_offset='-.02')
    def run(args):
        calls.append(args)
        if args[args.index('-i')+1]==str(candidate):
            return SimpleNamespace(stderr=json.dumps(dict(input_i=str(next(measurements)),input_tp='-1.6')))
        return SimpleNamespace(stderr=json.dumps(stats))
    monkeypatch.setattr(render,'_run',run)
    render._normalize(original,candidate)
    encodes=[c for c in calls if c[-1]==str(candidate)]
    assert len(encodes)==3 and len(calls)==7
    assert all(c[c.index('-i')+1]==str(original) for c in encodes)
    correction=encodes[-1][encodes[-1].index('-af')+1]
    assert 'volume=1.590000dB' in correction
    assert 'aresample=192000' in correction and correction.endswith('aresample=48000')
    assert 'alimiter=limit=0.7943282347' in correction


def test_valid_encoded_audio_needs_no_gain_or_limiter_retry(tmp_path,monkeypatch):
    calls=[];candidate=tmp_path/'candidate.mp4'
    stats=dict(input_i='-20',input_tp='-3',input_lra='2',input_thresh='-30',target_offset='0')
    def run(args):
        calls.append(args)
        return SimpleNamespace(stderr=json.dumps(dict(input_i='-14.49',input_tp='-1.6') if args[args.index('-i')+1]==str(candidate) else stats))
    monkeypatch.setattr(render,'_run',run)
    render._normalize(tmp_path/'joined.mov',candidate)
    assert len([c for c in calls if c[-1]==str(candidate)])==1
    assert not any('alimiter=' in c[c.index('-af')+1] for c in calls)
