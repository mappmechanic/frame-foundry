"""Local, opt-in music bed on the output sample grid; no timeline ownership."""
from __future__ import annotations

import array
import hashlib
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .contracts import StageError
from .media_tools import media_binary
from . import render as renderer

RATE = 48000
MAX_SECONDS = 3600
MAX_INTERVALS = 512
DEFAULT_POLICY = dict(bed_lufs=-14.0, dialogue_lufs=-24.0,
                      broll_min_seconds=2.5, transition_seconds=0.15)
REQUIRED_FILTERS = frozenset({'loudnorm', 'aresample', 'aformat', 'apad',
                              'atrim', 'amix', 'volume', 'alimiter'})


class AudioPolicyError(StageError):
    """Assembler-compatible, path-redacted local audio policy failure."""
    def __init__(self, code, message):
        super().__init__(code, 'assembler', message)


def _fail(code, message):
    raise AudioPolicyError(code, message)


def _run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        _fail('AUDIO_PROCESS', 'Local media process could not complete')
    if result.returncode:
        _fail('AUDIO_PROCESS', 'Local media process failed; verify decodable files and available codecs (paths redacted)')
    return result


def _ffmpeg():
    return [media_binary('ffmpeg'), '-hide_banner', '-nostdin', '-y',
            '-threads', '2', '-filter_threads', '1', '-filter_complex_threads', '1']


def _input(path):
    return ['-protocol_whitelist', 'file,pipe', '-i', str(path)]


def _filters():
    # Parse names, not flags: FFmpeg versions use both "..." and "..C", etc.
    text = _run([media_binary('ffmpeg'), '-hide_banner', '-filters']).stdout
    names = set(re.findall(r'^\s*[TSC.]{3}\s+(\w+)\s+', text, re.M))
    missing = sorted(REQUIRED_FILTERS - names)
    if missing:
        _fail('AUDIO_FILTERS', 'Required local audio filters unavailable: ' + ', '.join(missing))


def _probe(path):
    try:
        return json.loads(_run([media_binary('ffprobe'), '-v', 'error',
            '-protocol_whitelist', 'file,pipe', '-show_streams', '-show_format',
            '-of', 'json', str(path)]).stdout)
    except (AudioPolicyError, ValueError, KeyError, TypeError):
        _fail('AUDIO_PROBE', 'Cannot probe local media; supply a readable, decodable media file')


def _local(value, *, output=False):
    try:
        text = os.fspath(value)
        if not isinstance(text, str) or not text or '\x00' in text or '://' in text:
            raise ValueError()
        path = Path(text).expanduser().resolve()
        if (not output and (not path.is_file() or path.stat().st_size == 0)) or (output and path.is_dir()):
            raise ValueError()
        return path
    except (TypeError, ValueError, OSError):
        _fail('AUDIO_PATH', 'Expected a nonempty local media file and a local output file')


def _policy(value):
    if value is None:
        return DEFAULT_POLICY.copy()
    if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY):
        _fail('AUDIO_POLICY', 'Unsupported audio policy keys')
    result = DEFAULT_POLICY | value
    for key, number in result.items():
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
            _fail('AUDIO_POLICY', 'Audio policy numbers must be finite')
        if key != 'transition_seconds' and number != DEFAULT_POLICY[key]:
            _fail('AUDIO_POLICY', 'Loudness targets and B-roll threshold are fixed')
    if not 0 <= result['transition_seconds'] <= 0.5:
        _fail('AUDIO_POLICY', 'Transition must be between zero and 0.5 seconds')
    return result


def _intervals(values, duration, samples):
    try:
        iterator = iter(values)
        rows = []
        for index, row in enumerate(iterator):
            if index >= MAX_INTERVALS or not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError()
            start, end = row
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
                raise ValueError()
            if not 0 <= start < end <= duration:
                raise ValueError()
            # Outward rounding protects even sub-sample speech.
            rows.append((max(0, min(samples-1, math.floor(start * RATE))),
                         max(1, min(samples, math.ceil(end * RATE)))))
    except (TypeError, ValueError, OverflowError):
        _fail('AUDIO_INTERVALS', 'Expected at most 512 finite output-relative (start, end) seconds within video')
    merged = []
    for start, end in sorted(rows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _regions(intervals, samples, policy):
    rows, cursor = [], 0
    threshold = round(policy['broll_min_seconds'] * RATE)
    for start, end in intervals + [(samples, samples)]:
        if start > cursor:
            rows.append(dict(start_sample=cursor, end_sample=start,
                kind='broll' if start-cursor > threshold else 'short_gap'))
        if end > start:
            rows.append(dict(start_sample=start, end_sample=end, kind='dialogue'))
        cursor = end
    return rows


def _measure(path, extra=()):
    stderr = _run(_ffmpeg() + list(extra) + _input(path) + ['-vn', '-af',
        'loudnorm=I=-14:TP=-2:LRA=11:print_format=json', '-f', 'null', '-']).stderr
    try:
        stats = json.loads(stderr[stderr.rfind('{'):stderr.rfind('}')+1])
        result = {key: float(stats[field]) for key, field in
                  [('integrated_lufs', 'input_i'), ('true_peak_dbtp', 'input_tp')]}
        if not all(math.isfinite(x) for x in result.values()):
            raise ValueError()
        return result
    except (ValueError, KeyError, TypeError):
        _fail('AUDIO_LOUDNESS', 'Audio is silent or has no finite EBU R128 measurement')


def _envelope(raw, shaped, regions, base_gain, policy):
    """Stream bounded PCM chunks; fades lie entirely outside protected speech."""
    ramp = round(policy['transition_seconds'] * RATE)
    with raw.open('rb') as source, shaped.open('wb') as dest:
        for region in regions:
            count = region['end_sample'] - region['start_sample']
            low = 10 ** ((policy['dialogue_lufs'] - policy['bed_lufs']) / 20)
            broll = region['kind'] == 'broll'
            in_energy = out_energy = 0.0
            minimum, maximum = math.inf, 0.0
            offset = 0
            while offset < count:
                length = min(4096, count-offset)
                data = source.read(length * 2 * 4)
                if len(data) != length * 2 * 4:
                    _fail('AUDIO_DURATION', 'Decoded bed did not fill the output sample grid')
                values = array.array('f'); values.frombytes(data)
                for i in range(length):
                    pos = offset + i
                    fraction = min(1.0, pos/ramp, (count-1-pos)/ramp) if broll and ramp else float(broll)
                    # Linear amplitude ramps, exactly -10 dB during speech/short gaps.
                    gain = base_gain * (low + (1-low) * fraction)
                    relative = gain / base_gain
                    minimum = min(minimum, relative); maximum = max(maximum, relative)
                    for channel in (0, 1):
                        j = 2*i + channel
                        in_energy += values[j] ** 2
                        values[j] *= gain
                        out_energy += values[j] ** 2
                dest.write(values.tobytes()); offset += length
            region.update(premix_target_lufs=policy['bed_lufs'] if broll else policy['dialogue_lufs'],
                min_relative_gain_db=20*math.log10(minimum),
                max_relative_gain_db=20*math.log10(maximum),
                measured_gain_db=10*math.log10(out_energy/in_energy) if in_energy > 0 else None,
                input_rms=math.sqrt(in_energy/(count*2)), output_rms=math.sqrt(out_energy/(count*2)))


def _apply_audio_policy(video, output, *, music_path=None, authorized=False,
                       dialogue_intervals=(), policy=None) -> dict:
    """Copy normalized video unchanged, or locally mix explicitly authorized music.

    Intervals are (start, end) seconds relative to OUTPUT, including source ASR
    protected speech and cached VO. Parent owns their timeline mapping.
    """
    policy = _policy(policy)
    video, output = _local(video), _local(output, output=True)
    if video == output:
        _fail('AUDIO_PATH', 'Input and output must be distinct local files')
    info = _probe(video)
    try:
        streams = info['streams']
        visual = next(s for s in streams if s['codec_type'] == 'video')
        duration = float(visual.get('duration', info['format']['duration']))
        if not math.isfinite(duration) or not 0 < duration <= MAX_SECONDS:
            raise ValueError()
        samples = round(duration * RATE)
        frame_rate = float(Fraction(visual['avg_frame_rate']))
        if samples < 1 or not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError()
    except (ValueError, KeyError, StopIteration, TypeError, ZeroDivisionError):
        _fail('AUDIO_DURATION', 'Expected video with a finite duration of at most 3600 seconds')
    intervals = _intervals(dialogue_intervals, duration, samples)
    receipt = dict(version=1, status='absent', music_sha256=None, authorized=False,
        duration_seconds=duration, sample_rate=RATE, sample_count=samples,
        policy=policy, dialogue_intervals=[[a/RATE, b/RATE] for a,b in intervals],
        regions=[], limitations=[])
    if music_path is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(video, output)
        except OSError:
            _fail('AUDIO_PATH', 'Could not copy normalized source to output')
        receipt['limitations'] = ['No authorized music supplied; normalized source copied byte-for-byte without re-normalization.']
        return receipt
    if authorized is not True:
        _fail('AUDIO_AUTHORIZATION', 'Supplied music requires explicit operator authorization (authorized=True)')
    music = _local(music_path)
    if output == music:
        _fail('AUDIO_PATH', 'Output must not overwrite the supplied music')
    if not any(s['codec_type'] == 'audio' for s in streams):
        _fail('AUDIO_SOURCE', 'Music mixing requires existing rendered source audio including any intro')
    music_info = _probe(music)
    try:
        music_audio = next(s for s in music_info['streams'] if s.get('codec_type') == 'audio')
        music_duration = float(music_audio.get('duration', music_info['format']['duration']))
        if not math.isfinite(music_duration) or music_duration <= 0:
            raise ValueError()
    except (KeyError, ValueError, TypeError, StopIteration):
        _fail('AUDIO_MUSIC', 'Supplied music needs an audio stream with finite positive duration')
    _filters()
    digest = hashlib.sha256()
    with music.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    output.parent.mkdir(parents=True, exist_ok=True)
    regions = _regions(intervals, samples, policy)
    with tempfile.TemporaryDirectory(prefix='.audio-policy-', dir=output.parent) as directory:
        root = Path(directory)
        bed, raw, shaped = root/'bed.wav', root/'bed.f32', root/'shaped.f32'
        # Decode a local loop to the exact output grid before measuring: no assets
        # are generated or fetched. This is looping/resampling the supplied bed.
        _run(_ffmpeg() + ['-stream_loop', '-1'] + _input(music) + ['-map', '0:a:0',
            '-af', f'aresample={RATE},aformat=channel_layouts=stereo,apad,atrim=end_sample={samples}',
            '-c:a', 'pcm_f32le', '-ar', str(RATE), '-ac', '2', str(bed)])
        stats = _measure(bed)
        base_gain = 10 ** ((policy['bed_lufs']-stats['integrated_lufs'])/20)
        _run(_ffmpeg() + _input(bed) + ['-map', '0:a:0', '-f', 'f32le', '-c:a', 'pcm_f32le', str(raw)])
        _envelope(raw, shaped, regions, base_gain, policy)
        joined, candidate = root/'mixed.mov', root/'candidate.mp4'
        graph = (f'[0:a:0]aresample={RATE}:async=1:first_pts=0,aformat=channel_layouts=stereo,'
                 f'apad,atrim=end_sample={samples}[source];'
                 f'[source][1:a:0]amix=inputs=2:duration=first:normalize=0,'
                 f'atrim=end_sample={samples}[mix]')
        _run(_ffmpeg() + _input(video) + ['-f', 'f32le', '-ar', str(RATE), '-ac', '2'] +
            _input(shaped) + ['-filter_complex', graph, '-map', '0:v:0', '-map', '[mix]',
            '-c:v', 'copy', '-c:a', 'pcm_f32le', str(joined)])
        try:
            renderer._normalize(joined, candidate)
        except StageError as exc:
            raise AudioPolicyError('AUDIO_NORMALIZE', 'Authoritative renderer normalization failed') from exc
        final = _measure(candidate)
        if not -15 <= final['integrated_lufs'] <= -13 or final['true_peak_dbtp'] > -1:
            _fail('AUDIO_QA', 'Encoded mix failed authoritative -14 +/-1 LUFS / <=-1 dBTP gates')
        delivered = _probe(candidate)
        out_video = next(s for s in delivered['streams'] if s['codec_type'] == 'video')
        frame_duration = 1 / frame_rate
        try:
            out_duration = float(out_video['duration'])
            container_duration = float(delivered['format']['duration'])
            if not math.isfinite(out_duration) or not math.isfinite(container_duration):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            _fail('AUDIO_DURATION', 'Cannot verify delivered video duration')
        if (abs(out_duration-duration) > frame_duration or
                abs(container_duration-duration) > frame_duration or
                out_video.get('nb_frames') != visual.get('nb_frames')):
            _fail('AUDIO_DURATION', 'Mix changed video frame count or duration beyond one frame')
        os.replace(candidate, output)
    receipt.update(status='mixed', authorized=True, music_sha256=digest.hexdigest(),
        regions=regions, music_input_measurement=stats,
        bed_normalization_gain_db=20*math.log10(base_gain), final_measurement=final,
        video_stream_copy=True, output_video_duration_seconds=out_duration,
        output_container_duration_seconds=container_duration,
        limitations=['Authorization is operator attestation, not license verification.',
          'Bed targets are absolute pre-mix EBU R128-reference levels; final overall normalization changes all audio globally.',
          'Regional music content can vary in loudness; measured PCM gain/RMS proves ducking, not uniform regional LUFS.',
          'Music loops locally; no download, generation, paid calls, or automatic speech detection.'])
    return receipt


def apply_audio_policy(video, output, *, music_path=None, authorized=False,
                       dialogue_intervals=(), policy=None) -> dict:
    """Apply optional local authorized music; return a path-free JSON-safe receipt.

    Output-relative speech/VO intervals are finite (start, end) seconds.
    Errors are AudioPolicyError/StageError with stage='assembler'. Output is
    replaced only after encoded loudness/peak and video-duration verification.
    """
    try:
        return _apply_audio_policy(video, output, music_path=music_path,
            authorized=authorized, dialogue_intervals=dialogue_intervals, policy=policy)
    except OSError:
        _fail('AUDIO_PATH', 'Cannot access local media or temporary/output storage; check permissions and disk space')
