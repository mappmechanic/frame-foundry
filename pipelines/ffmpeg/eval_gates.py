"""Zero-token, independent evaluation gates measured on the actual delivered media.

Nothing here trusts qa-report.json numbers: loudness, geometry, frame timing,
duration, source cuts, black/frozen content and repeated shots are re-measured
from finished.mp4, timeline.otio and the original sources. Missing evidence is
reported as ``not_run`` (which fails G), never as a fabricated pass.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import json
import math
import re
import subprocess

from jsonschema import Draft202012Validator
from PIL import Image, ImageChops, ImageFilter, ImageStat

GATES_VERSION = 'eval-gates-1.0'
REPO = Path(__file__).resolve().parents[2]
THRESHOLDS = dict(
    lufs_min=-15.0, lufs_max=-13.0, true_peak_max_dbtp=-1.0, fps=30,
    frame_interval_tolerance_ms=1.0, duration_tolerance_fraction=0.10,
    pre_roll_ms=80, post_roll_ms=120, edge_tolerance_ms=1,
    black_min_seconds=0.5, black_pixel_luma=38, black_pixel_fraction=0.98,
    freeze_min_seconds=1.0, freeze_mean_abs_diff=0.15,
    dhash_bits=64, dhash_max_distance=4, dhash_sample_frames=15, dhash_flat_luma_std=6.0,
    letterbox_margin_fraction=0.01, analysis_scale_divisor=4)
CHECK_NAMES = ['artifacts_present', 'artifact_schemas', 'timeline_sources', 'readable_media',
               'dimensions', 'audio_stream', 'constant_frame_rate', 'loudness', 'true_peak',
               'duration_target', 'word_cut_integrity', 'black_frames', 'frozen_frames',
               'repeated_shots', 'contact_sheet']
# Known JSON artifacts and their contracts; anything else is declared, not inferred.
KNOWN_SCHEMAS = {'narrative-spec.json':'narrative-spec.schema.json','qa-report.json':'qa-report.schema.json','eval-report.json':'eval-report.schema.json',
    'analysis.json':'internal:analysis','transcript.json':'internal:transcript','state.json':'internal:state','proxy-map.json':'internal:proxy_map',
    'contact-sheet-index.json':'internal:contact_sheet_index','llm-usage.json':'internal:llm_usage','tts-usage.json':'internal:tts_usage','eval-judge-guard.json':'internal:eval_judge_guard'}
UNSCHEMATIZED = []
_TIMEOUT = 900


class GateError(RuntimeError):
    pass


def check(name, status, reason=None, measurement=None):
    item = dict(name=name, status=status, reason=reason)
    if measurement is not None:
        item['measurement'] = measurement
    return item


def _run(args, timeout=_TIMEOUT):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateError(f'{args[0]} could not run') from exc
    if result.returncode:
        raise GateError(f'{args[0]} failed: {result.stderr[-400:]}')
    return result


def read_strict_json(path):
    def no_constant(value):
        raise ValueError('Non-finite JSON value')

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), parse_constant=no_constant, object_pairs_hook=unique)


def optional_json(path):
    try:
        return read_strict_json(path) if Path(path).is_file() else None
    except (OSError, ValueError):
        return None


def ffprobe(path):
    data = json.loads(_run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                            '-of', 'json', str(path)]).stdout)
    video = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
    audio = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if video is None:
        raise GateError('No video stream')
    rotation = float(video.get('tags', {}).get('rotate', 0) or 0)
    for side in video.get('side_data_list', []):
        rotation = float(side.get('rotation', rotation))
    duration = float(data['format'].get('duration') or video.get('duration'))
    return dict(data=data, video=video, audio=audio, width=int(video['width']),
                height=int(video['height']), duration_ms=round(duration * 1000), rotation=rotation)


def measure_ebur128(path):
    """Integrated loudness and true peak from ffmpeg ebur128=peak=true."""
    err = _run(['ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-v', 'info', '-i', str(path),
                '-map', '0:a:0', '-af', 'ebur128=peak=true:framelog=verbose', '-f', 'null', '-']).stderr
    summary = err[err.rfind('Summary:'):]

    def value(pattern):
        match = re.search(pattern, summary, re.S)
        if not match:
            return None
        try:
            number = float(match.group(1))
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return dict(integrated_lufs=value(r'Integrated loudness:\s*I:\s*(\S+)\s*LUFS'),
                true_peak_dbtp=value(r'True peak:\s*Peak:\s*(\S+)\s*dBFS'))


def frame_timing(path):
    out = _run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'packet=pts_time', '-of', 'csv=p=0', str(path)]).stdout
    pts = sorted(float(x.split(',')[0]) for x in out.split() if x.strip() and x.strip() != 'N/A')
    deltas = [round((b - a) * 1000, 4) for a, b in zip(pts, pts[1:])]
    return dict(frames=len(pts), min_interval_ms=min(deltas) if deltas else None,
                max_interval_ms=max(deltas) if deltas else None)


def read_shots(episode_dir):
    """Ordered V1 shots from the saved OTIO, in source and output milliseconds."""
    import opentimelineio as otio
    from .render import timeline_ranges
    timeline = otio.adapters.read_from_file(str(Path(episode_dir) / 'timeline.otio'))
    timeline_ranges(timeline, episode_dir=episode_dir)  # reuse render's strict V1/A1 contract
    video = next(t for t in timeline.tracks if t.kind == 'Video')
    shots, cursor = [], 0.0
    for index, clip in enumerate(video, 1):
        start = clip.source_range.start_time.to_seconds() * 1000
        duration = clip.source_range.duration.to_seconds() * 1000
        available = clip.media_reference.available_range
        shots.append(dict(
            shot_id=f'shot-{index:02d}', source_id=clip.metadata.get('source_id'),
            source_path=clip.media_reference.target_url,
            source_start_ms=round(start), source_end_ms=round(start + duration),
            source_duration_ms=round(available.duration.to_seconds() * 1000),
            timeline_start_ms=round(cursor), timeline_end_ms=round(cursor + duration),
            selection_ids=list(clip.metadata.get('selection_ids') or [])))
        cursor += duration
    return shots, timeline.metadata.get('aspect_ratio')


def word_cut_violations(shots, words):
    """Cut edges must not fall inside (onset-80ms, decay+120ms) of any source word.

    The only exemption is an edge on the source's first/last sample where the
    protected interval was clamped by available media; that clamp is recorded.
    """
    pre, post, tol = THRESHOLDS['pre_roll_ms'], THRESHOLDS['post_roll_ms'], THRESHOLDS['edge_tolerance_ms']
    violations, clamps = [], []
    for shot in shots:
        duration = shot['source_duration_ms']
        mine = [w for w in words if w.get('source_id') == shot['source_id']]
        for edge_name, edge in (('start', shot['source_start_ms']), ('end', shot['source_end_ms'])):
            for word in mine:
                a, b = word['start_ms'] - pre, word['end_ms'] + post
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) or not (math.isfinite(a) and math.isfinite(b)):
                    raise TypeError('Non-numeric word interval')
                ca, cb = max(0, a), min(duration, b)
                if ca + tol < edge < cb - tol:
                    violations.append(dict(shot_id=shot['shot_id'], edge=edge_name, edge_ms=edge,
                                           word=str(word.get('text', ''))[:40],
                                           word_start_ms=word['start_ms'], word_end_ms=word['end_ms']))
                elif ((edge_name == 'start' and edge <= tol and a < 0 <= word['end_ms'])
                      or (edge_name == 'end' and edge >= duration - tol and b > duration >= word['start_ms'])):
                    clamps.append(dict(shot_id=shot['shot_id'], edge=edge_name, edge_ms=edge,
                                       word_start_ms=word['start_ms'], word_end_ms=word['end_ms'],
                                       note='protected padding clamped to available source media'))
    return violations, clamps


def content_box(render_w, render_h, source_w, source_h, rotation=0):
    """Render-space rectangle holding real picture (render.py fit+pad letterboxing)."""
    if source_w is None or source_h is None:
        return (0, 0, render_w, render_h)
    if round(abs(rotation)) % 180 == 90:
        source_w, source_h = source_h, source_w
    ratio = min(render_w / source_w, render_h / source_h)
    cw, ch = max(2, int(source_w * ratio) // 2 * 2), max(2, int(source_h * ratio) // 2 * 2)
    x, y = (render_w - cw) // 2, (render_h - ch) // 2
    mx, my = int(cw * THRESHOLDS['letterbox_margin_fraction']), int(ch * THRESHOLDS['letterbox_margin_fraction'])
    return (x + mx, y + my, x + cw - mx, y + ch - my)


def dhash(image):
    small = image.resize((9, 8), Image.Resampling.BOX)
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f'{bits:016x}'


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count('1')


_LAPLACE = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def _best_shift(prev, cur, radius=3):
    """Coarse global translation (integer px on a 64-px-wide copy) by exhaustive SAD."""
    w, h = cur.size
    best = (float('inf'), 0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            a = prev.crop((max(0, dx), max(0, dy), w + min(0, dx), h + min(0, dy)))
            b = cur.crop((max(0, -dx), max(0, -dy), w + min(0, -dx), h + min(0, -dy)))
            score = ImageStat.Stat(ImageChops.difference(a, b)).mean[0]
            if score < best[0] - 1e-9:
                best = (score, dx, dy)
    return best[1], best[2]


def analyze_frames(video_path, shots, render_w, render_h, source_dims, fps=30):
    """One grayscale decode of the final render: black, freeze, dHash and D evidence."""
    div = THRESHOLDS['analysis_scale_divisor']
    aw, ah = max(2, render_w // div // 2 * 2), max(2, render_h // div // 2 * 2)
    sx, sy = aw / render_w, ah / render_h
    boxes = []
    for shot in shots:
        dims = source_dims.get(shot['source_path'])
        box = content_box(render_w, render_h, *(dims or (None, None, 0)))
        x0, y0 = int(round(box[0] * sx)), int(round(box[1] * sy))
        boxes.append((x0, y0, max(x0 + 2, int(round(box[2] * sx))), max(y0 + 2, int(round(box[3] * sy)))))
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video_path), '-map', '0:v:0',
                             '-vf', f'scale={aw}:{ah}:flags=area,format=gray', '-f', 'rawvideo', '-'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = aw * ah
    frames = []
    per_shot = [dict(frames=0, luma_sum=0.0, clip_sum=0.0, diffs=[], shifts=[], mid=None, hashes=[]) for _ in shots]
    mids = [((s['timeline_start_ms'] + s['timeline_end_ms']) / 2) for s in shots]
    prev = prev_small = None
    prev_shot, index, shot_index = -1, 0, 0
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            t = index * 1000 / fps
            while shot_index + 1 < len(shots) and t >= shots[shot_index]['timeline_end_ms']:
                shot_index += 1
            image = Image.frombytes('L', (aw, ah), raw)
            crop = image.crop(boxes[shot_index]) if shots else image
            hist = crop.histogram()
            count = crop.size[0] * crop.size[1]
            black = sum(hist[:THRESHOLDS['black_pixel_luma'] + 1]) >= THRESHOLDS['black_pixel_fraction'] * count
            small = crop.resize((64, max(8, round(64 * crop.size[1] / crop.size[0]))), Image.Resampling.BOX)
            diff = None
            if prev is not None and prev_shot == shot_index:
                diff = ImageStat.Stat(ImageChops.difference(prev, crop)).mean[0]
            if shots:
                stats = per_shot[shot_index]
                stats['frames'] += 1
                stats['luma_sum'] += sum(i * v for i, v in enumerate(hist)) / count
                stats['clip_sum'] += (sum(hist[:17]) + sum(hist[236:])) / count
                if diff is not None:
                    stats['diffs'].append(diff)
                    stats['shifts'].append(_best_shift(prev_small, small))
                if stats['mid'] is None and t + 1000 / fps > mids[shot_index]:
                    stats['mid'] = crop.copy()
                if (stats['frames'] - 1) % THRESHOLDS['dhash_sample_frames'] == THRESHOLDS['dhash_sample_frames'] // 2:
                    if ImageStat.Stat(crop).stddev[0] >= THRESHOLDS['dhash_flat_luma_std']:
                        stats['hashes'].append(dhash(crop))
            frames.append(dict(t_ms=t, black=black, diff=diff))
            prev, prev_small, prev_shot = crop, small, shot_index
            index += 1
    finally:
        proc.stdout.close()
        proc.wait(timeout=60)
    if proc.returncode:
        raise GateError('Frame decode failed')
    if not frames:
        raise GateError('No decodable frames')

    def runs(flags, min_seconds, transitions=False):
        found, start = [], None
        for i, flag in enumerate(flags + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                length = (i - start) / fps
                if length > min_seconds:
                    first = start - 1 if transitions else start
                    found.append(dict(start_ms=round(max(0, first) * 1000 / fps),
                                      end_ms=round(i * 1000 / fps), seconds=round(length, 3)))
                start = None
        return found
    black_runs = runs([f['black'] for f in frames], THRESHOLDS['black_min_seconds'])
    freeze_runs = runs([f['diff'] is not None and f['diff'] <= THRESHOLDS['freeze_mean_abs_diff'] for f in frames],
                       THRESHOLDS['freeze_min_seconds'], transitions=True)
    shot_stats = []
    for shot, stats in zip(shots, per_shot):
        mid = stats['mid']
        entry = dict(shot_id=shot['shot_id'], frames=stats['frames'])
        if mid is not None:
            std = ImageStat.Stat(mid).stddev[0]
            hashes = stats['hashes'] + ([dhash(mid)] if std >= THRESHOLDS['dhash_flat_luma_std'] else [])
            entry.update(dhash=dhash(mid), dhashes=sorted(set(hashes)), luma_std=round(std, 3),
                         flat=std < THRESHOLDS['dhash_flat_luma_std'],
                         laplacian_variance=round(ImageStat.Stat(mid.filter(_LAPLACE)).var[0], 3))
        if stats['frames']:
            entry.update(mean_luma=round(stats['luma_sum'] / stats['frames'], 3),
                         clipped_fraction=round(stats['clip_sum'] / stats['frames'], 5))
        entry['mean_abs_diff'] = round(sum(stats['diffs']) / len(stats['diffs']), 4) if stats['diffs'] else None
        shifts = stats['shifts']
        accel = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(shifts, shifts[1:])]
        entry['jitter_px64'] = round(sum(accel) / len(accel), 4) if accel else None
        shot_stats.append(entry)
    hook = [f['diff'] for f in frames if f['diff'] is not None and f['t_ms'] < 2000]
    every = [f['diff'] for f in frames if f['diff'] is not None]
    return dict(frames=len(frames), analysis_size=[aw, ah], black_runs=black_runs, freeze_runs=freeze_runs,
                shots=shot_stats, hook_mean_abs_diff=sum(hook) / len(hook) if hook else None,
                overall_mean_abs_diff=sum(every) / len(every) if every else None)


def ffmpeg_content_defects(video_path,duration_ms):
    """Corroborate content-aware local measurements with named FFmpeg detectors.

    Full-frame blackdetect does not classify ordinary letterbox borders as black
    because 98% of the complete picture must be dark. The separate cropped
    analysis catches black source content hidden by mattes. freezedetect uses a
    conservative -80 dB noise floor; cropped jitter analysis remains stricter.
    """
    err=_run(['ffmpeg','-hide_banner','-nostats','-nostdin','-v','info','-i',str(video_path),'-an','-vf',
        'scale=270:-2,blackdetect=d=0.5:pix_th=0.15:pic_th=0.98,freezedetect=n=-80dB:d=1','-f','null','-']).stderr
    black=[dict(start_ms=round(float(a)*1000),end_ms=round(float(b)*1000),seconds=float(d))
        for a,b,d in re.findall(r'black_start:([0-9.]+) black_end:([0-9.]+) black_duration:([0-9.]+)',err) if float(d)>0.5]
    frozen=[];start=None
    for kind,value in re.findall(r'freeze_(start|end):\s*([0-9.]+)',err):
        if kind=='start':start=float(value)
        elif start is not None:
            end=float(value)
            if end-start>1:frozen.append(dict(start_ms=round(start*1000),end_ms=round(end*1000),seconds=end-start))
            start=None
    if start is not None and duration_ms/1000-start>1:
        frozen.append(dict(start_ms=round(start*1000),end_ms=duration_ms,seconds=duration_ms/1000-start))
    return black,frozen


def repeated_shot_pairs(shot_stats):
    pairs, flat = [], [s['shot_id'] for s in shot_stats if not s.get('dhashes')]
    hashed = [s for s in shot_stats if s.get('dhashes')]
    for i, a in enumerate(hashed):
        for b in hashed[i + 1:]:
            distance = min(hamming(x, y) for x in a['dhashes'] for y in b['dhashes'])
            if distance <= THRESHOLDS['dhash_max_distance']:
                pairs.append(dict(a=a['shot_id'], b=b['shot_id'], distance=distance))
    return pairs, flat


def _schema_errors(payload, schema_name):
    if schema_name.startswith('internal:'):
        schema=read_strict_json(REPO/'schemas'/'episode-artifacts.schema.json')
        schema['$ref']='#/$defs/'+schema_name.split(':',1)[1]
    else:
        folder='contracts' if schema_name=='eval-report.schema.json' else 'schemas'
        schema=read_strict_json(REPO/folder/schema_name)
    return [e.message[:200] for e in list(Draft202012Validator(schema).iter_errors(payload))[:3]]


def run_gates(episode_dir):
    """Return dict(passed, checks, context, limitations). Never raises for bad artifacts."""
    root = Path(episode_dir)
    checks, limitations = {}, []
    ctx = dict(spec=optional_json(root / 'narrative-spec.json'), analysis=optional_json(root / 'analysis.json'),
               qa=optional_json(root / 'qa-report.json'), shots=None, frames=None, render=None, loudness=None)
    spec, analysis = ctx['spec'], ctx['analysis']
    required = ['finished.mp4', 'timeline.otio', 'narrative-spec.json', 'analysis.json', 'contact-sheet.jpg']
    missing = [n for n in required if not (root / n).is_file() or not (root / n).stat().st_size]
    checks['artifacts_present'] = check('artifacts_present', 'fail' if missing else 'pass',
                                        f'Missing artifacts: {", ".join(missing)}' if missing else None,
                                        dict(required=required, missing=missing))
    # Schemas: validate artifacts that have a contract; declare the rest.
    validated, invalid, unparsable, declared = [], {}, [], []
    for name, schema in KNOWN_SCHEMAS.items():
        if (root / name).is_file():
            try:
                errors = _schema_errors(read_strict_json(root / name), schema)
            except (OSError, ValueError):
                errors = ['Unparsable or non-finite JSON']
            if errors:
                invalid[name] = errors
            else:
                validated.append(name)
    # Every emitted top-level JSON must have a declared contract. Nested saved
    # narrative plans share the same contract as the final narrative.
    for path in root.rglob('*.json'):
        if path.parent==root and path.name in KNOWN_SCHEMAS:continue
        name=str(path.relative_to(root))
        schema=KNOWN_SCHEMAS.get(path.name)
        if schema is None:
            invalid[name]=['No registered schema for emitted Episode JSON']
            continue
        try:
            errors=_schema_errors(read_strict_json(path),schema)
        except Exception:errors=['Unparsable or schema-invalid JSON']
        if errors:invalid[name]=errors
        else:validated.append(name)
    bad = bool(invalid or unparsable or spec is None)
    reason = None
    if bad:
        reason = ('Invalid contracts: ' + json.dumps(dict(invalid=invalid, unparsable=unparsable))[:400]
                  if invalid or unparsable else 'narrative-spec.json unavailable')
    checks['artifact_schemas'] = check('artifact_schemas', 'fail' if bad else 'pass', reason,
                                       dict(validated=validated, invalid=sorted(invalid), unparsable=unparsable,
                                            unschematized=declared))
    shots, aspect, source_dims = None, None, {}
    try:
        shots, aspect = read_shots(root)
        for shot in shots:
            path = shot['source_path']
            if path not in source_dims:
                info = ffprobe(path)
                source_dims[path] = (info['width'], info['height'], info['rotation'])
            if not 0 <= shot['source_start_ms'] < shot['source_end_ms'] <= shot['source_duration_ms'] + 1:
                raise GateError('Source range outside available media')
        ctx['shots'] = shots
        checks['timeline_sources'] = check('timeline_sources', 'pass', None,
                                           dict(shots=len(shots), sources=len(source_dims)))
    except Exception as exc:
        shots = None
        checks['timeline_sources'] = check('timeline_sources', 'fail',
                                           f'Timeline/source evidence invalid: {str(exc)[:200]}')
    requested = (spec or {}).get('aspect_ratio') or aspect
    target = (spec or {}).get('target_duration_ms')
    render = None
    if (root / 'finished.mp4').is_file():
        try:
            render = ffprobe(root / 'finished.mp4')
            ctx['render'] = render
            checks['readable_media'] = check('readable_media', 'pass', None, dict(duration_ms=render['duration_ms']))
        except Exception:
            checks['readable_media'] = check('readable_media', 'fail', 'finished.mp4 cannot be probed')
    else:
        checks['readable_media'] = check('readable_media', 'not_run', 'No finished render')
    if render is None:
        for name in ['dimensions', 'audio_stream', 'constant_frame_rate', 'loudness', 'true_peak',
                     'duration_target', 'black_frames', 'frozen_frames', 'repeated_shots']:
            checks[name] = check(name, 'not_run', 'Render unavailable or unreadable')
    else:
        _media_checks(root, render, requested, target, shots, source_dims, checks, ctx, limitations)
    _word_check(root, analysis, shots, checks, limitations)
    sheet = root / 'contact-sheet.jpg'
    try:
        with Image.open(sheet) as image:
            image.verify()
        with Image.open(sheet) as image:
            dims = list(image.size)
        checks['contact_sheet'] = check('contact_sheet', 'pass', None, dict(size=dims, bytes=sheet.stat().st_size))
    except Exception:
        checks['contact_sheet'] = check('contact_sheet', 'fail', 'Frame-boundary contact-sheet.jpg missing or undecodable')
    ordered = [checks[name] for name in CHECK_NAMES]
    return dict(passed=all(c['status'] == 'pass' for c in ordered), checks=ordered, context=ctx,
                limitations=limitations)


def _media_checks(root, render, requested, target, shots, source_dims, checks, ctx, limitations):
    video_path = root / 'finished.mp4'
    size = {'9:16': (1080, 1920), '16:9': (1920, 1080)}.get(requested)
    actual = (render['width'], render['height'])
    if size is None:
        checks['dimensions'] = check('dimensions', 'fail', 'Requested aspect unavailable', dict(actual=list(actual)))
    else:
        checks['dimensions'] = check('dimensions', 'pass' if actual == size else 'fail',
                                     None if actual == size else f'Expected {size[0]}x{size[1]} for {requested}',
                                     dict(requested=requested, expected=list(size), actual=list(actual)))
    checks['audio_stream'] = check('audio_stream', 'pass' if render['audio'] else 'fail',
                                   None if render['audio'] else 'No audio stream',
                                   dict(codec=(render['audio'] or {}).get('codec_name')))
    try:
        timing = frame_timing(video_path)
        v = render['video']
        r, avg = Fraction(v.get('r_frame_rate', '0/1')), Fraction(v.get('avg_frame_rate', '0/1'))
        ideal, tol = 1000 / THRESHOLDS['fps'], THRESHOLDS['frame_interval_tolerance_ms']
        ok = (r == avg == THRESHOLDS['fps'] and timing['frames'] > 1
              and abs(timing['min_interval_ms'] - ideal) <= tol and abs(timing['max_interval_ms'] - ideal) <= tol)
        checks['constant_frame_rate'] = check('constant_frame_rate', 'pass' if ok else 'fail',
                                              None if ok else 'Frame timestamps are not constant 30 fps',
                                              dict(r_frame_rate=str(r), avg_frame_rate=str(avg), **timing))
    except Exception:
        checks['constant_frame_rate'] = check('constant_frame_rate', 'fail', 'Frame timestamps unavailable')
    if render['audio']:
        try:
            loud = measure_ebur128(video_path)
            ctx['loudness'] = loud
            lufs, peak = loud['integrated_lufs'], loud['true_peak_dbtp']
            lok = lufs is not None and THRESHOLDS['lufs_min'] <= lufs <= THRESHOLDS['lufs_max']
            pok = peak is not None and peak <= THRESHOLDS['true_peak_max_dbtp']
            checks['loudness'] = check('loudness', 'pass' if lok else 'fail',
                                       None if lok else 'Integrated loudness must be finite within [-15,-13] LUFS',
                                       dict(integrated_lufs=lufs, method='ffmpeg ebur128=peak=true'))
            checks['true_peak'] = check('true_peak', 'pass' if pok else 'fail',
                                        None if pok else 'True peak must be finite and <= -1 dBTP',
                                        dict(true_peak_dbtp=peak, method='ffmpeg ebur128=peak=true'))
        except Exception:
            checks['loudness'] = check('loudness', 'fail', 'Loudness measurement failed')
            checks['true_peak'] = check('true_peak', 'fail', 'True-peak measurement failed')
    else:
        checks['loudness'] = check('loudness', 'fail', 'No audio stream to measure')
        checks['true_peak'] = check('true_peak', 'fail', 'No audio stream to measure')
    if type(target) is int and target > 0:
        deviation = abs(render['duration_ms'] - target) / target
        ok = deviation <= THRESHOLDS['duration_tolerance_fraction'] + 1e-9
        checks['duration_target'] = check(
            'duration_target', 'pass' if ok else 'fail',
            None if ok else 'Duration outside target +/-10% (strict; partial coverage is not exempt)',
            dict(target_ms=target, actual_ms=render['duration_ms'], deviation_fraction=round(deviation, 4)))
    else:
        checks['duration_target'] = check('duration_target', 'fail', 'Target duration unavailable')
    if not shots:
        for name in ['black_frames', 'frozen_frames', 'repeated_shots']:
            checks[name] = check(name, 'not_run', 'Timeline shots unavailable')
        return
    try:
        frames = analyze_frames(video_path, shots, render['width'], render['height'], source_dims)
    except Exception as exc:
        for name in ['black_frames', 'frozen_frames', 'repeated_shots']:
            checks[name] = check(name, 'fail', f'Frame analysis failed: {str(exc)[:120]}')
        return
    ctx['frames'] = frames
    try:
        detected_black,detected_freeze=ffmpeg_content_defects(video_path,render['duration_ms'])
    except GateError:
        checks['black_frames']=check('black_frames','fail','FFmpeg blackdetect unavailable')
        checks['frozen_frames']=check('frozen_frames','fail','FFmpeg freezedetect unavailable')
        checks['repeated_shots']=check('repeated_shots','not_run','Required detector failed')
        return
    frames['black_runs']+=detected_black
    frames['freeze_runs']+=detected_freeze
    checks['black_frames'] = check('black_frames', 'fail' if frames['black_runs'] else 'pass',
                                   'Black content > 0.5 s' if frames['black_runs'] else None,
                                   dict(runs=frames['black_runs'], letterbox_excluded=True,method='content-aware analysis + ffmpeg blackdetect'))
    checks['frozen_frames'] = check('frozen_frames', 'fail' if frames['freeze_runs'] else 'pass',
                                    'Frozen content > 1 s' if frames['freeze_runs'] else None,
                                    dict(runs=frames['freeze_runs'], letterbox_excluded=True,method='content-aware analysis + ffmpeg freezedetect',
                                         mean_abs_diff_threshold=THRESHOLDS['freeze_mean_abs_diff']))
    pairs, flat = repeated_shot_pairs(frames['shots'])
    checks['repeated_shots'] = check('repeated_shots', 'fail' if pairs else 'pass',
                                     'Perceptually repeated shots' if pairs else None,
                                     dict(method='min dHash-64 distance over content frames sampled every 0.5 s (luma-guarded)',
                                          max_distance=THRESHOLDS['dhash_max_distance'],
                                          pairs=pairs, flat_excluded=flat))
    if flat:
        limitations.append('Low-texture shots excluded from duplicate hashing by luma guard: ' + ', '.join(flat))


def _word_check(root, analysis, shots, checks, limitations):
    status = (analysis or {}).get('transcription_status')
    if status == 'visual_only_unavailable':
        checks['word_cut_integrity'] = check('word_cut_integrity', 'not_run',
                                             'Visual-only diagnostic: no speech evidence; word-safe cuts unverified',
                                             dict(transcription_status=status))
        return
    if status not in ('fixture', 'local_asr'):
        checks['word_cut_integrity'] = check('word_cut_integrity', 'not_run', 'Transcription provenance unknown',
                                             dict(transcription_status=status))
        return
    if not shots:
        checks['word_cut_integrity'] = check('word_cut_integrity', 'not_run', 'Timeline shots unavailable')
        return
    transcript = optional_json(root / 'transcript.json')
    words = transcript.get('words') if isinstance(transcript, dict) else None
    if not isinstance(words, list):
        checks['word_cut_integrity'] = check('word_cut_integrity', 'not_run', 'Word evidence (transcript.json) unavailable')
        return
    try:
        violations, clamps = word_cut_violations(shots, words)
    except (KeyError, TypeError, AttributeError):
        checks['word_cut_integrity'] = check('word_cut_integrity', 'fail', 'Malformed word evidence')
        return
    checks['word_cut_integrity'] = check(
        'word_cut_integrity', 'fail' if violations else 'pass',
        'Cut inside protected word interval (-80 ms/+120 ms)' if violations else None,
        dict(transcription_status=status, words=len(words), violations=violations[:20], source_clamps=clamps[:20]))
    if clamps:
        limitations.append(f'{len(clamps)} cut edge(s) rely on source-boundary clamped word padding')
    if status == 'fixture':
        limitations.append('Word evidence is an explicit transcript fixture, not local ASR')


# --------------------------------------------------------------------------
# Seeded-defect self-test: proves detection of KNOWN injected defects only.
# --------------------------------------------------------------------------
SEEDED_DEFECTS = {'loud': 'loudness', 'black': 'black_frames', 'aspect': 'dimensions',
                  'midword': 'word_cut_integrity', 'duplicate': 'repeated_shots', 'freeze': 'frozen_frames'}
_DISTINCT_SOURCES = ['testsrc2=size=320x180:rate=30', 'mandelbrot=size=320x180:rate=30',
                     'testsrc2=size=320x180:rate=30,hflip,hue=h=120:s=2']


def generate_distinct_footage(out_dir, *, seconds=4):
    """Visually distinct moving synthetic sources plus an explicit fixture transcript."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for index, source in enumerate(_DISTINCT_SOURCES, 1):
        path = out_dir / f'{index:02d}-trip.mp4'
        _run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', source, '-f', 'lavfi', '-i',
              f'sine=frequency={330 + 110 * index}:sample_rate=48000', '-t', str(seconds), '-c:v', 'libx264',
              '-preset', 'ultrafast', '-threads', '2', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(path)])
        sources.append(dict(source_id=f'source-{index:04d}', filename=path.name, duration_ms=seconds * 1000))
    # Synthetic word timing only (tones are not speech); labelled fixture evidence.
    words = [dict(source_id='source-0001', text='hello', start_ms=1000, end_ms=1400, confidence=1.0)]
    fixture = out_dir.parent / f'{out_dir.name}-transcript-fixture.json'
    fixture.write_text(json.dumps(dict(schema_version='1.0', sources=sources, words=words), indent=2))
    return fixture


def _reencode_video(src, dst, vf=None, filter_complex=None, af=None):
    args = ['ffmpeg', '-y', '-v', 'error', '-nostdin', '-i', str(src)]
    if filter_complex:
        args += ['-filter_complex', filter_complex, '-map', '[v]', '-map', '0:a:0']
    else:
        args += ['-map', '0:v:0', '-map', '0:a:0']
        if vf:
            args += ['-vf', vf]
    if vf is None and filter_complex is None:
        args += ['-c:v', 'copy']
    else:
        args += ['-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p']
    args += (['-af', af, '-c:a', 'aac', '-ar', '48000'] if af else ['-c:a', 'copy'])
    _run(args + ['-movflags', '+faststart', str(dst)])
    Path(dst).replace(src)


def seed_defect(episode_dir, defect):
    """Mutate a copied passing Episode in place to contain exactly one known defect."""
    root = Path(episode_dir)
    video, tmp = root / 'finished.mp4', root / 'seeded.tmp.mp4'
    shots, _ = read_shots(root)
    if defect == 'loud':
        _reencode_video(video, tmp, af='volume=14dB')
    elif defect == 'black':
        mid = shots[len(shots) // 2]
        t0 = (mid['timeline_start_ms'] + 200) / 1000
        _reencode_video(video, tmp, vf=f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{t0},{t0 + 0.9})'")
    elif defect == 'aspect':
        _reencode_video(video, tmp, vf='scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1')
    elif defect == 'freeze':
        shot = max(shots, key=lambda s: s['timeline_end_ms'] - s['timeline_start_ms'])
        first = shot['timeline_start_ms'] * 30 // 1000 + 5
        _reencode_video(video, tmp, filter_complex=f'[0:v]split[x][y];[x][y]freezeframes=first={first}:last={first + 45}:replace={first}[v]')
    elif defect == 'duplicate':
        a, b = sorted(shots[:2], key=lambda s: s['timeline_end_ms'] - s['timeline_start_ms'], reverse=True)
        segments, labels = [], []
        for i, shot in enumerate(shots):
            start, end = shot['timeline_start_ms'] / 1000, shot['timeline_end_ms'] / 1000
            if shot is b:  # replace the shorter shot's frames with the longer shot's frames
                start = a['timeline_start_ms'] / 1000
                end = start + (b['timeline_end_ms'] - b['timeline_start_ms']) / 1000
            segments.append(f'[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[s{i}]')
            labels.append(f'[s{i}]')
        graph = ';'.join(segments) + ';' + ''.join(labels) + f'concat=n={len(labels)}:v=1:a=0[v]'
        _reencode_video(video, tmp, filter_complex=graph)
    elif defect == 'midword':
        transcript = read_strict_json(root / 'transcript.json')
        shot = next(s for s in shots if s['source_end_ms'] < s['source_duration_ms'] - 300)
        edge = shot['source_end_ms']
        transcript['words'].append(dict(source_id=shot['source_id'], text='seeded', start_ms=edge - 60,
                                        end_ms=edge + 180, confidence=1.0))
        (root / 'transcript.json').write_text(json.dumps(transcript, indent=2))
    else:
        raise ValueError(f'Unknown seeded defect {defect}')


def seeded_self_test(work_dir, defects=tuple(SEEDED_DEFECTS)):
    """Render one real offline Episode, seed each defect into a copy, re-run gates.

    Returns counts for the seeded set only; this is not a claim about all defects.
    """
    import shutil
    from .cli import run_episode  # lazy: avoid import cycles with the runner
    work = Path(work_dir)
    fixture = generate_distinct_footage(work / 'footage')
    report = run_episode(work / 'footage', 'Create a 10 second family-trip teaser', episodes_root=work / 'episodes',
                         episode_id='selftest-base', transcript_fixture=fixture, provider=None)
    base = work / 'episodes' / 'selftest-base'
    control = run_gates(base)
    cases = []
    for defect in defects:
        target = work / 'seeded' / defect
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(base, target, ignore=shutil.ignore_patterns('work', 'proxies', 'frames'))
        seed_defect(target, defect)
        result = run_gates(target)
        failed = [c['name'] for c in result['checks'] if c['status'] != 'pass']
        expected = SEEDED_DEFECTS[defect]
        cases.append(dict(defect=defect, expected_check=expected, detected=expected in failed and not result['passed'],
                          failed_checks=failed))
    return dict(pipeline_passed=report['passed'], control_passed=control['passed'],
                control_failed=[c['name'] for c in control['checks'] if c['status'] != 'pass'],
                seeded=len(cases), detected=sum(c['detected'] for c in cases), cases=cases)
