"""Render original-source V1/A1 OTIO ranges using bounded FFmpeg processes."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

from .contracts import StageError

_TIMEOUT = 600

def output_frames(duration):
    """Nearest nonempty 30fps clip length, shared with independent QA mapping."""
    return max(1, int(math.floor(duration * 30 + .5)))


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


def _normalize(joined, candidate, *, target_i=-14):
    # AAC can create intersample overshoot; 1 dB encoding headroom keeps the
    # delivered <= -1 dBTP gate independent from normalization metadata.
    measured = _run(_ffmpeg() + ['-i', str(joined), '-vn', '-af',
        f'loudnorm=I={target_i}:TP=-2:LRA=11:print_format=json', '-f', 'null', '-']).stderr
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
    filter_ = (f'loudnorm=I={target_i}:TP=-2:LRA=11:linear=true:print_format=json:'
               f'measured_I={stats["input_i"]}:measured_TP={stats["input_tp"]}:'
               f'measured_LRA={stats["input_lra"]}:measured_thresh={stats["input_thresh"]}:'
               f'offset={stats["target_offset"]}')
    _run(_ffmpeg() + ['-i', str(joined), '-map', '0:v:0', '-map', '0:a:0',
                      '-c:v', 'copy', '-af', filter_, '-c:a', 'aac', '-b:a', '192k',
                      '-ar', '48000', '-ac', '2', '-threads', '2',
                      '-movflags', '+faststart', str(candidate)])
    # Always measure encoded AAC, not loudnorm's predicted output. Two-pass
    # dynamic normalization can undershoot on high-crest phone audio. First try
    # the existing bounded -13 LUFS normalization; if it still misses, perform
    # one measured gain correction with an oversampled post-normalization peak
    # limiter. Re-encode from lossless joined audio, never compound AAC losses.
    measured = _run(_ffmpeg() + ['-i',str(candidate),'-vn','-af',
        'loudnorm=I=-14:TP=-1:LRA=11:print_format=json','-f','null','-']).stderr
    try:
        encoded=json.loads(measured[measured.rfind('{'):measured.rfind('}')+1])
        i,p=float(encoded['input_i']),float(encoded['input_tp'])
    except (ValueError,KeyError):
        return  # independent validator will reject missing evidence
    if not (math.isfinite(i) and math.isfinite(p)) or (-15<=i<=-13 and p<=-1):
        return
    if target_i == -14:
        _normalize(joined,candidate,target_i=-13)
        return
    gain=max(-12.0,min(12.0,-14.0-i))
    limited=(filter_+f',aresample=192000,volume={gain:.6f}dB,'
             'alimiter=limit=0.7943282347:attack=5:release=50:level=false:latency=true,'
             'aresample=48000')  # -2 dBTP before AAC, 4x sample-rate peak limiting.
    _run(_ffmpeg() + ['-i',str(joined),'-map','0:v:0','-map','0:a:0',
                      '-c:v','copy','-af',limited,'-c:a','aac','-b:a','192k',
                      '-ar','48000','-ac','2','-threads','2',
                      '-movflags','+faststart',str(candidate)])
    # No further attempts here. Independent LUFS/true-peak QA decides delivery;
    # truly sparse/silent/incompatible content is retained only as diagnostic.


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
    framing_mode=timeline.metadata.get('framing_mode','fit_pad')
    if framing_mode not in ('center_crop','fit_pad'):
        raise StageError('TIMELINE_INVALID','render','Unsupported saved framing mode')
    ranges = timeline_ranges(timeline, episode_dir=episode_dir)
    candidate = Path(candidate_path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    pieces, durations_us, cut_times, cursor_frames = [], [], [], 0
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
        frames = output_frames(interval['duration'])
        duration = frames / 30
        # Portrait teasers fill the frame with a proportional center crop.
        # Explicit horizontal output retains the existing proportional fit.
        framing = (
            f'scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,'
            f'crop={width}:{height}:(iw-ow)/2:(ih-oh)/2'
            if aspect == '9:16' and framing_mode=='center_crop' else
            f'scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
            f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black')
        video = (f'[0:v:0]trim=start={start:.9f}:end={end:.9f},'
                 f'setpts=PTS-({start:.9f})/TB,'
                 f'{framing},setsar=1,'
                 f'fps=30:start_time=0,tpad=stop_mode=clone:stop_duration={duration:.9f},'
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
        # MOV preserves the exact 30fps video time base alongside PCM audio.
        # Matroska rounds PTS to milliseconds; stream-copying those into MP4
        # produces alternating 33/34ms frames and a fractional average rate.
        piece = candidate.parent / f'cut-{index + 1:03d}.mov'
        _run(args + ['-filter_complex', video + ';' + audio, '-map', '[v]', '-map', '[a]',
                    '-t', f'{duration:.9f}', '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                    '-c:a', 'pcm_s16le', '-ar', '48000', '-ac', '2', str(piece)])
        pieces.append(piece)
        durations_us.append(round((cursor_frames+frames)*1000000/30)-round(cursor_frames*1000000/30))
        if index:
            cut_times.append(round(cursor_frames * 1000 / 30))
        cursor_frames += frames
    manifest = candidate.parent / 'concat.txt'
    # Generated basenames only: no source filename is interpolated into concat syntax.
    # Override container duration rounding at each cut; rounded cumulative
    # microsecond boundaries prevent concat offset drift over many clips.
    manifest.write_text(''.join(f"file '{p.name}'\nduration {us/1000000:.6f}\n"
                                for p,us in zip(pieces,durations_us)))
    joined = candidate.parent / 'joined.mov'
    _run(_ffmpeg() + ['-f', 'concat', '-safe', '1', '-i', str(manifest),
                      '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy', str(joined)])
    if timeline.metadata.get('voiceover'):
        from .voiceover import mix_intro
        joined=mix_intro(joined,timeline.metadata['voiceover'],episode_dir,cursor_frames/30)
    _normalize(joined, candidate)
    info = probe_media(candidate)
    return dict(cut_times_ms=cut_times, duration_ms=round(float(info['format']['duration']) * 1000))
