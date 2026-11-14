"""Real plain-FFmpeg policy tests; media exists only in pytest temp directories."""
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from pipelines.ffmpeg import audio_policy as audio
from pipelines.ffmpeg.media_tools import media_binary
from pipelines.ffmpeg import render


def run(*args):
    return subprocess.run([media_binary('ffmpeg'), '-hide_banner', '-nostdin', '-y',
        '-threads', '2', '-filter_threads', '1', '-filter_complex_threads', '1', *map(str, args)],
        capture_output=True, check=True, timeout=120)


@pytest.fixture(scope='module')
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp('real-audio')
    video, music = root/'source.mp4', root/'music.wav'
    # Stable 440-Hz source dialogue + pre-existing 880-Hz cached VO intro.
    run('-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=30:d=10',
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=10',
        '-f', 'lavfi', '-i', 'sine=frequency=880:sample_rate=48000:duration=1',
        '-filter_complex', '[1:a]volume=0.7[a];[a][2:a]amix=inputs=2:normalize=0,apad,atrim=duration=10[out]',
        '-map', '0:v', '-map', '[out]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-t', '10', video)
    normalized = root/'normalized.mp4'
    render._normalize(video, normalized)
    run('-f', 'lavfi', '-i', 'sine=frequency=2000:sample_rate=48000:duration=2',
        '-c:a', 'pcm_s16le', music)
    return normalized, music


def pcm(path, start=0, duration=None):
    args = ['-i', path, '-ss', str(start)]
    if duration is not None:
        args += ['-t', str(duration)]
    return array.array('f', run(*args, '-vn', '-ac', '1', '-ar', '48000', '-f', 'f32le', '-').stdout)


def amplitude(path, frequency, start, duration):
    values = pcm(path, start, duration)
    # Frequency-selective real amplitude isolates music from original dialogue/VO.
    sine = sum(x*math.sin(2*math.pi*frequency*i/audio.RATE) for i,x in enumerate(values))
    cosine = sum(x*math.cos(2*math.pi*frequency*i/audio.RATE) for i,x in enumerate(values))
    return 2*math.hypot(sine, cosine)/len(values)


def video_packets(path):
    result = subprocess.run([media_binary('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
        '-show_packets', '-show_data_hash', 'sha256', '-of', 'json', str(path)],
        capture_output=True, check=True, text=True)
    return [(p['pts_time'], p['duration_time'], p['data_hash']) for p in json.loads(result.stdout)['packets']]


def test_absent_is_byte_identical_no_normalization(media, tmp_path, monkeypatch):
    source, _ = media
    monkeypatch.setattr(audio, '_filters', lambda: pytest.fail('absent needs no optional filters'))
    monkeypatch.setattr(render, '_normalize', lambda *a: pytest.fail('already normalized'))
    output = tmp_path/'absent.mp4'
    receipt = audio.apply_audio_policy(source, output, dialogue_intervals=[(0, 1)])
    assert output.read_bytes() == source.read_bytes()
    assert receipt['status'] == 'absent' and receipt['limitations']
    assert receipt['music_sha256'] is None
    json.dumps(receipt, allow_nan=False)


def test_real_mix_duck_loudness_peak_duration_frames_and_intro(media, tmp_path):
    source, music = media
    output = tmp_path/'mixed.mp4'
    receipt = audio.apply_audio_policy(source, output, music_path=music, authorized=True,
        dialogue_intervals=[(0,1), (2,3), (8,9)])
    assert receipt['status'] == 'mixed'
    assert receipt['music_sha256'] == hashlib.sha256(music.read_bytes()).hexdigest()
    public = json.dumps(receipt, allow_nan=False)
    assert str(music) not in public and music.name not in public
    stats = audio._measure(output)
    assert -15 <= stats['integrated_lufs'] <= -13
    assert stats['true_peak_dbtp'] <= -1
    assert abs(receipt['output_video_duration_seconds']-10) <= 1/30
    assert video_packets(output) == video_packets(source)  # visuals and timestamps unchanged
    ducked = amplitude(output, 2000, 2.2, .5)
    broll = amplitude(output, 2000, 5, .5)
    short_gap = amplitude(output, 2000, 1.2, .5)
    assert 20*math.log10(broll/ducked) == pytest.approx(10, abs=1)
    assert 20*math.log10(short_gap/ducked) == pytest.approx(0, abs=1)
    # Both source dialogue and cached VO remain, with no deletion of the intro.
    assert amplitude(output, 440, .2, .5) > .01
    assert amplitude(output, 880, .2, .5) > .01
    assert amplitude(output, 880, .2, .5)/amplitude(output, 440, .2, .5) == pytest.approx(
        amplitude(source, 880, .2, .5)/amplitude(source, 440, .2, .5), rel=.08)
    for row in receipt['regions']:
        if row['kind'] != 'broll':
            assert row['max_relative_gain_db'] == pytest.approx(-10)
            assert row['measured_gain_db']-receipt['bed_normalization_gain_db'] == pytest.approx(-10, abs=.001)
    # Independent EBU R128 measurement of the shaped sine verifies absolute
    # pre-mix -24/-14 targets, not just declared gain parameters.
    root = tmp_path/'proof'; root.mkdir()
    samples = 10*audio.RATE
    bed = root/'bed.wav'; raw = root/'bed.f32'; shaped = root/'shaped.f32'
    run('-stream_loop', '-1', '-i', music, '-t', '10', '-ac', '2', '-ar', audio.RATE, '-c:a', 'pcm_f32le', bed)
    measurement = audio._measure(bed)
    run('-i', bed, '-f', 'f32le', '-c:a', 'pcm_f32le', raw)
    regions = audio._regions([(0,audio.RATE),(2*audio.RATE,3*audio.RATE),(8*audio.RATE,9*audio.RATE)], samples, audio.DEFAULT_POLICY)
    audio._envelope(raw, shaped, regions, 10**((-14-measurement['integrated_lufs'])/20), audio.DEFAULT_POLICY)
    wav = root/'shaped.wav'
    run('-f', 'f32le', '-ar', audio.RATE, '-ac', '2', '-i', shaped, '-c:a', 'pcm_f32le', wav)
    for start, target in [(2.1,-24), (5,-14)]:
        section = root/f'{start}.wav'
        run('-i', wav, '-ss', start, '-t', '0.8', '-c:a', 'pcm_f32le', section)
        assert audio._measure(section)['integrated_lufs'] == pytest.approx(target, abs=.3)


@pytest.mark.parametrize('intervals', [[(-1,1)],[(1,1)],[(0,11)],[(0,float('nan'))],
    [(0,float('inf'))],[(False,1)],[(0,'1')],[(0,1,2)],None,[(0,1)]*513])
def test_bad_intervals_fail_before_output(media, tmp_path, intervals):
    output = tmp_path/'bad.mp4'
    with pytest.raises(audio.AudioPolicyError) as error:
        audio.apply_audio_policy(media[0], output, dialogue_intervals=intervals)
    assert error.value.code == 'AUDIO_INTERVALS'
    assert error.value.stage == 'assembler' and not output.exists()


@pytest.mark.parametrize('authorized', [False, None, 1, 'true'])
def test_unlicensed_fails(media, tmp_path, authorized):
    with pytest.raises(audio.AudioPolicyError, match='authorization'):
        audio.apply_audio_policy(media[0], tmp_path/'out.mp4', music_path=media[1], authorized=authorized)


@pytest.mark.parametrize('path', ['https://example.com/a.wav', 'file:///tmp/x.wav', 'missing.wav', 'bad\x00name'])
def test_unsafe_music_paths(media, tmp_path, path):
    with pytest.raises(audio.AudioPolicyError) as error:
        audio.apply_audio_policy(media[0], tmp_path/'out.mp4', music_path=path, authorized=True)
    assert error.value.code == 'AUDIO_PATH'


def test_non_audio_and_silent_music_rejected(media, tmp_path):
    silent = tmp_path/'silent.wav'
    run('-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-t', 1, silent)
    with pytest.raises(audio.AudioPolicyError) as error:
        audio.apply_audio_policy(media[0], tmp_path/'out.mp4', music_path=silent, authorized=True)
    assert error.value.code == 'AUDIO_LOUDNESS'
    no_audio = tmp_path/'no-audio.mp4'
    run('-i', media[0], '-an', '-c:v', 'copy', no_audio)
    with pytest.raises(audio.AudioPolicyError) as error:
        audio.apply_audio_policy(media[0], tmp_path/'out.mp4', music_path=no_audio, authorized=True)
    assert error.value.code == 'AUDIO_MUSIC'


@pytest.mark.parametrize('flag', ['...', '..C', 'T.C'])
def test_plain_and_homebrew_filter_flags(monkeypatch, flag):
    text = '\n'.join(f' {flag} {name} A->A Some filter' for name in audio.REQUIRED_FILTERS)
    monkeypatch.setattr(audio, '_run', lambda _: subprocess.CompletedProcess([], 0, text, ''))
    audio._filters()


def test_missing_filters_typed_error(monkeypatch):
    monkeypatch.setattr(audio, '_run', lambda _: subprocess.CompletedProcess([], 0, ' ... loudnorm A->A', ''))
    with pytest.raises(audio.AudioPolicyError) as error:
        audio._filters()
    assert error.value.code == 'AUDIO_FILTERS' and error.value.stage == 'assembler'


def test_policy_bounded(media, tmp_path):
    for policy in [{'dialogue_lufs':-20}, {'transition_seconds':float('nan')}, {'transition_seconds':1}, {'unknown':1}]:
        with pytest.raises(audio.AudioPolicyError) as error:
            audio.apply_audio_policy(media[0], tmp_path/'out.mp4', policy=policy)
        assert error.value.code == 'AUDIO_POLICY'


def test_interval_merge_grid_and_strict_broll_threshold():
    merged = audio._intervals([(1.000001,2),(0,1.5),(4.5,5)], 8, 8*audio.RATE)
    assert merged == [(0,2*audio.RATE),(int(4.5*audio.RATE),5*audio.RATE)]
    rows = audio._regions(merged, 8*audio.RATE, audio.DEFAULT_POLICY)
    assert [r['kind'] for r in rows] == ['dialogue','short_gap','dialogue','broll']
