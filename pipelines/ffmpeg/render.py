"""Render original-source V1/A1 OTIO ranges using bounded FFmpeg processes."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

from .contracts import StageError

_TIMEOUT = 600


def _run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageError('RENDER_PROCESS', 'render', f'{args[0]} could not complete: {exc}') from exc
    if result.returncode:
        raise StageError('RENDER_PROCESS', 'render', f'{args[0]} failed: {result.stderr[-8000:]}')
    return result


def probe_media(path):
    result = _run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)])
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise StageError('RENDER_PROBE', 'render', 'Invalid ffprobe JSON') from exc


def _ffmpeg():
    return ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-loglevel', 'info',
            '-threads', '2', '-filter_threads', '1', '-filter_complex_threads', '1']


def _path(url, episode_dir):
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme != 'file':
        raise StageError('SOURCE_REFERENCE', 'render', 'Only original local source references are supported')
    if parsed.scheme == 'file' and parsed.netloc not in ('', 'localhost'):
        raise StageError('SOURCE_REFERENCE', 'render', 'Remote file references are not supported')
    path = Path(unquote(parsed.path) if parsed.scheme else url)
    if not path.is_absolute():
        path = Path(episode_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise StageError('SOURCE_MISSING', 'render', f'Original source does not exist: {path}')
    return path


def timeline_ranges(timeline, *, episode_dir):
    """Read ordered, original-source ranges exclusively from serialized OTIO data."""
    tracks = list(timeline.tracks)
    if len(tracks) != 2 or [t.kind for t in tracks] != ['Video', 'Audio']:
        raise StageError('TIMELINE_INVALID', 'render', 'Expected exactly V1/A1 tracks')
    if len(tracks[0]) != len(tracks[1]) or not len(tracks[0]):
        raise StageError('TIMELINE_INVALID', 'render', 'Empty or unaligned V1/A1 tracks')
    ranges = []
    for clip, audio in zip(*tracks):
        try:
            if (clip.schema_name() != 'Clip' or audio.schema_name() != 'Clip'
                    or clip.media_reference.schema_name() != 'ExternalReference'
                    or audio.media_reference.schema_name() != 'ExternalReference'
                    or clip.media_reference.target_url != audio.media_reference.target_url
                    or clip.source_range != audio.source_range):
                raise ValueError('Unaligned original-source clips')
            start = clip.source_range.start_time.to_seconds()
            duration = clip.source_range.duration.to_seconds()
            if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
                raise ValueError('Invalid source range')
            available = clip.media_reference.available_range
            if available is None or start + duration > available.end_time_exclusive().to_seconds() + 1e-6:
                raise ValueError('Source range exceeds available range')
            ranges.append(dict(path=_path(clip.media_reference.target_url, episode_dir),
                               start=start, duration=duration))
        except (AttributeError, TypeError, ValueError) as exc:
            raise StageError('TIMELINE_INVALID', 'render', f'Invalid OTIO clip: {exc}') from exc
    return ranges


def _normalize(joined, candidate):
    measured = _run(_ffmpeg() + ['-i', str(joined), '-vn', '-af',
        'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-']).stderr
    try:
        # loudnorm JSON is emitted on stderr after normal FFmpeg progress.
        stats = json.loads(measured[measured.rfind('{'):measured.rfind('}') + 1])
        finite = all(math.isfinite(float(stats[k])) for k in
                     ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset'))
    except (ValueError, KeyError) as exc:
        raise StageError('LOUDNESS_INVALID', 'render', f'Could not read loudnorm measurement: {measured[-4000:]}') from exc
    if not finite:
        # Digital silence has -inf LUFS. Keep it silent; do not synthesize content.
        _run(_ffmpeg() + ['-i', str(joined), '-map', '0:v:0', '-map', '0:a:0',
                          '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                          '-ac', '2', '-threads', '2', '-movflags', '+faststart', str(candidate)])
        return
    filter_ = ('loudnorm=I=-14:TP=-1:LRA=11:linear=true:print_format=json:'
               f'measured_I={stats["input_i"]}:measured_TP={stats["input_tp"]}:'
               f'measured_LRA={stats["input_lra"]}:measured_thresh={stats["input_thresh"]}:'
               f'offset={stats["target_offset"]}')
    _run(_ffmpeg() + ['-i', str(joined), '-map', '0:v:0', '-map', '0:a:0',
                      '-c:v', 'copy', '-af', filter_, '-c:a', 'aac', '-b:a', '192k',
                      '-ar', '48000', '-ac', '2', '-threads', '2',
                      '-movflags', '+faststart', str(candidate)])


def render_timeline(timeline, candidate_path, *, episode_dir):
    """Render a saved/readback timeline; return measured duration and output cuts.

    Source A/V retains its common container timestamp origin. Audio timestamps
    are NOT independently reset to their first sample: aresample fills any real
    leading gap, and apad only maintains stream continuity up to the video end.
    """
    aspect = timeline.metadata.get('aspect_ratio')
    if aspect not in ('9:16', '16:9'):
        raise StageError('TIMELINE_INVALID', 'render', 'Missing/invalid aspect ratio')
    width, height = (1080, 1920) if aspect == '9:16' else (1920, 1080)
    ranges = timeline_ranges(timeline, episode_dir=episode_dir)
    candidate = Path(candidate_path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    pieces, cut_times, cursor = [], [], 0
    for index, interval in enumerate(ranges):
        path = interval['path']
        info = probe_media(path)
        if not any(s['codec_type'] == 'video' for s in info['streams']):
            raise StageError('SOURCE_VIDEO_MISSING', 'render', f'No original video stream: {path}')
        has_audio = any(s['codec_type'] == 'audio' for s in info['streams'])
        origin = float(info['format'].get('start_time', 0))
        start = origin + interval['start']
        end = start + interval['duration']
        # Round each clip to the fixed 30fps output grid, then use that same
        # duration for audio and output mapping; do not accumulate AAC priming.
        frames = max(1, int(math.floor(interval['duration'] * 30 + .5)))
        duration = frames / 30
        video = (f'[0:v:0]trim=start={start:.9f}:end={end:.9f},'
                 f'setpts=PTS-({start:.9f})/TB,'
                 f'scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                 f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,'
                 f'fps=30,tpad=stop_mode=clone:stop_duration={duration:.9f},'
                 f'trim=duration={duration:.9f},format=yuv420p[v]')
        args = _ffmpeg() + ['-ss', f'{interval["start"]:.9f}', '-copyts', '-i', str(path)]
        if has_audio:
            audio = (f'[0:a:0]atrim=start={start:.9f}:end={end:.9f},'
                     f'asetpts=PTS-({start:.9f})/TB,aresample=48000:async=1:first_pts=0,'
                     f'aformat=channel_layouts=stereo,apad,atrim=duration={duration:.9f}[a]')
        else:
            args += ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo']
            audio = f'[1:a:0]atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[a]'
        # Lossless PCM in intermediate files avoids AAC encoder delay at cuts.
        piece = candidate.parent / f'cut-{index + 1:03d}.mkv'
        _run(args + ['-filter_complex', video + ';' + audio, '-map', '[v]', '-map', '[a]',
                    '-t', f'{duration:.9f}', '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                    '-c:a', 'pcm_s16le', '-ar', '48000', '-ac', '2', str(piece)])
        pieces.append(piece)
        if index:
            cut_times.append(round(cursor * 1000))
        cursor += duration
    manifest = candidate.parent / 'concat.txt'
    # Generated basenames only: no source filename is interpolated into concat syntax.
    manifest.write_text(''.join(f"file '{p.name}'\n" for p in pieces))
    joined = candidate.parent / 'joined.mkv'
    _run(_ffmpeg() + ['-f', 'concat', '-safe', '1', '-i', str(manifest),
                      '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy', str(joined)])
    _normalize(joined, candidate)
    info = probe_media(candidate)
    return dict(cut_times_ms=cut_times, duration_ms=round(float(info['format']['duration']) * 1000))
