"""Deterministic local candidate evidence; no external models or token calls.

Candidates are evidence windows, NOT word-boundary-safe edit decisions. The
assembler must still apply audio pre/post roll. Image grouping permits preview
reuse only: a perceptual image match never removes a distinct audio moment.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageFilter, ImageStat

from .audio_analysis import analyze_audio
from .contracts import StageError

MAX_WINDOW_MS = 10000
SNIPPET_CHARS = 160


def _bounded(value, maximum=100.0):
    try:
        number = float(value)
        return min(maximum, max(0.0, number)) if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _local_path(root, relative):
    if not relative or Path(relative).is_absolute():
        return None
    path = (root / relative).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def _image_info(path):
    try:
        with Image.open(path) as image:
            gray = image.convert('L')
            gray.thumbnail((320, 320))
            edges = gray.filter(ImageFilter.Kernel((3, 3), (0, 1, 0, 1, -4, 1, 0, 1, 0), scale=1))
            if min(gray.size) > 2:
                edges = edges.crop((1, 1, gray.width - 1, gray.height - 1))
            sharpness = _bounded(ImageStat.Stat(edges).mean[0] * 4)
            histogram = gray.histogram()
            clipped = (sum(histogram[:12]) + sum(histogram[244:])) / (gray.width * gray.height)
            mean = ImageStat.Stat(gray).mean[0]
            exposure = _bounded((1 - clipped) * (1 - abs(mean - 127.5) / 127.5) * 100)
            small = gray.resize((32, 32), Image.Resampling.BILINEAR)
            hashed = list(gray.resize((9, 8), Image.Resampling.BILINEAR).getdata())
            bits = 0
            for y in range(8):
                for x in range(8):
                    bits = (bits << 1) | (hashed[y * 9 + x] > hashed[y * 9 + x + 1])
            return dict(sharpness=round(sharpness, 3), exposure=round(exposure, 3),
                        frame_hash=hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest(),
                        dhash=f'{bits:016x}', pixels=tuple(small.getdata()))
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _stability(first, second):
    """Sparse-frame luma-change surrogate, not calibrated camera-shake detection."""
    if first is None or second is None:
        return 0.0
    # A fixed bounded difference measures observed visual change. Sparse samples
    # cannot separate camera motion from moving subjects, cuts or lighting.
    change = sum(abs(a - b) for a, b in zip(first['pixels'], second['pixels'])) / 1024
    return round(_bounded(100 * (1 - change / 64)), 3)


def _near(first, second):
    distance = (int(first['dhash'], 16) ^ int(second['dhash'], 16)).bit_count()
    # Luma distance prevents uniform white/black dHash collisions from merging.
    luma_distance = sum(abs(a - b) for a, b in
                        zip(first['pixels'], second['pixels'])) / 1024
    # Tiny compression changes can flip many equal-intensity dHash ties.
    return luma_distance <= 2 or (distance <= 4 and luma_distance <= 12)


def _file_hash(path):
    try:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, TypeError):
        return None


def _audio_window(metrics, words, start, end, offset=0):
    duration = end - start
    bins = metrics.get('bins') or []
    weighted = covered = 0
    for item in bins:
        overlap = max(0, min(end, item['end_ms'] + offset) - max(start, item['start_ms'] + offset))
        weighted += _bounded(item.get('rms'), 1) ** 2 * overlap
        covered += overlap
    rms = math.sqrt(weighted / covered) if covered else _bounded(metrics.get('rms'), 1)
    intervals = metrics.get('speech_intervals_ms')
    method = str(metrics.get('method') or 'energy_only')
    if intervals is not None:
        ranges = [(a + offset, b + offset) for a, b in intervals]
    elif words:
        ranges = [(w['start_ms'], w['end_ms']) for w in words]
        method = 'asr_word_coverage'
    else:
        ranges = None
    if ranges is not None:
        # Union overlaps; duplicate/aligned words never inflate speech coverage.
        merged = []
        for a, b in sorted((max(start, a), min(end, b)) for a, b in ranges):
            if b <= a:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        speech = _bounded(sum(b - a for a, b in merged) / duration, 1)
    else:
        # A source-wide fraction cannot honestly locate speech in this window.
        speech = None
        if method not in ('no_audio', 'unavailable'):
            method = 'energy_only'
    return dict(rms=round(rms, 6), speech_fraction=speech, method=method)


def _base_score(candidate):
    quality = candidate.get('quality', {})
    audio = candidate.get('audio', {})
    return round(20*min(2,(quality.get('face_count') or 0)+(quality.get('person_count') or 0)) + 30*(quality.get('motion') or 0) + _bounded(quality.get('visual_score')) * .7 +
                 _bounded(audio.get('rms'), 1) * 10 +
                 _bounded(audio.get('speech_fraction'), 1) * 10, 6)


def extract_candidates(analysis: dict, episode_dir: Path, *, window_ms: int = MAX_WINDOW_MS) -> list[dict]:
    """Extract scene-split, source-bounded evidence windows of at most 10 seconds.

    ``frame_hash`` is a grayscale pixel SHA-256 (not collision-prone dHash);
    canonical_frame_* pointers identify near-image groups for preview reuse.
    Optional ``audio_metrics`` is a source-ID map of analyze_audio results. With
    no cached entry, a local proxy is decoded once per audio source. Extra fields
    include canonical image reuse, source provenance and honest quality methods.
    Missing scenes default to the full source; labeled keyframes never cross
    source identities. ``proxy_time_offset_ms`` maps proxy time onto source time.
    """
    if isinstance(window_ms, bool) or not isinstance(window_ms, int) or window_ms <= 0:
        raise ValueError('window_ms must be a positive integer')
    window_ms = min(window_ms, MAX_WINDOW_MS)
    root = Path(episode_dir).resolve()
    sources = sorted(analysis.get('sources', []), key=lambda s: s['source_id'])
    images = {}
    frames_by_source = {}
    for source in sources:
        frames = []
        for frame in source.get('keyframes', []):
            time = frame.get('source_time_ms')
            if time is None or not 0 <= time < source['duration_ms']:
                continue
            path = _local_path(root, frame.get('path'))
            if path is None:
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in images:
                images[relative] = _image_info(path)
            if images[relative] is not None:
                frames.append(dict(path=relative, time=time))
        frames_by_source[source['source_id']] = sorted(frames, key=lambda f: (f['time'], f['path']))
    canonical = {}
    representatives = []
    for path in sorted(images):
        if images[path] is None:
            continue
        match = next((p for p in representatives if _near(images[path], images[p])), None)
        canonical[path] = match or path
        if match is None:
            representatives.append(path)
    candidates = []
    duplicates = {}
    source_hashes = {}
    for source in sources:
        sid = source['source_id']
        duration = source['duration_ms']
        words = sorted((w for w in analysis.get('words', []) if w.get('source_id') == sid),
                       key=lambda w: (w['start_ms'], w['end_ms'], w['text']))
        metrics = (analysis.get('audio_metrics') or {}).get(sid)
        if metrics is None:
            proxy = _local_path(root, source.get('proxy_path'))
            if not source.get('has_audio', True):
                metrics = dict(rms=0, method='no_audio', speech_intervals_ms=[])
            elif proxy:
                offset = source.get('proxy_time_offset_ms', 0)
                proxy_words = [dict(w, start_ms=w['start_ms'] - offset, end_ms=w['end_ms'] - offset)
                               for w in words]
                try:
                    metrics = analyze_audio(proxy, words=proxy_words)
                except StageError:
                    metrics = dict(rms=0, method='unavailable')
            else:
                metrics = dict(rms=0, method='unavailable')
        frames = frames_by_source[sid]
        scenes = [s for s in analysis.get('scenes', []) if s.get('source_id') == sid]
        # Every scene boundary splits windows, even for overlapping scene records.
        boundaries = {0, duration}
        for scene in scenes:
            boundaries.update((max(0, min(duration, scene['start_ms'])),
                               max(0, min(duration, scene['end_ms']))))
        bounds = sorted(boundaries)
        intervals = []
        for a, b in zip(bounds, bounds[1:]):
            if scenes and not any(s['start_ms'] <= a and b <= s['end_ms'] for s in scenes):
                continue
            intervals.extend((start, min(start + window_ms, b)) for start in range(a, b, window_ms))
        original = source.get('path')
        # Hash a local original once; no file identity is inferred from thumbnails.
        if original not in source_hashes:
            source_hashes[original] = _file_hash(original) if original else None
        digest = source_hashes[original]
        for start, end in intervals:
            moment_words = [w for w in words if w['start_ms'] < end and w['end_ms'] > start]
            signature = tuple((w['start_ms'], w['end_ms'], w['text']) for w in moment_words)
            key = (digest, start, end, signature)
            provenance = dict(source_id=sid, start_ms=start, end_ms=end)
            if digest and key in duplicates:
                duplicates[key]['provenance'].append(provenance)
                continue
            in_window = [f for f in frames if start <= f['time'] < end]
            frame = min(in_window or frames,
                        key=lambda f: (abs(f['time'] - (start + end) / 2), f['time'], f['path'])) if frames else None
            info = images[frame['path']] if frame else None
            stability = 0.0
            stability_method = 'unavailable_single_frame'
            if frame:
                neighbors = [f for f in frames if f['time'] != frame['time']]
                if neighbors:
                    neighbor = min(neighbors, key=lambda f: (abs(f['time'] - frame['time']), f['time'], f['path']))
                    stability = _stability(info, images[neighbor['path']])
                    stability_method = 'sampled_luma_change_surrogate'
            sharpness = info['sharpness'] if info else 0.0
            exposure = info['exposure'] if info else 0.0
            quality = dict(sharpness=sharpness, stability=stability, exposure=exposure,
                           visual_score=round(.45 * sharpness + .25 * stability + .30 * exposure, 3))
            identity = json.dumps([sid, start, end], separators=(',', ':'))
            candidate = dict(candidate_id='candidate-' + hashlib.sha256(identity.encode()).hexdigest()[:20],
                             source_id=sid, start_ms=start, end_ms=end,
                             transcript_snippet=' '.join(w['text'] for w in moment_words)[:SNIPPET_CHARS],
                             quality=quality,
                             audio=_audio_window(metrics, words, start, end, source.get('proxy_time_offset_ms', 0)),
                             frame_path=frame['path'] if frame else None,
                             frame_hash=info['frame_hash'] if info else None,
                             canonical_frame_path=canonical[frame['path']] if frame else None,
                             canonical_frame_hash=images[canonical[frame['path']]]['frame_hash'] if frame else None,
                             frame_time_ms=frame['time'] if frame else None,
                             frame_in_window=start <= frame['time'] < end if frame else False,
                             quality_method=dict(sharpness='positive_laplacian_mean', exposure='luma_mean_and_clipping',
                                                 stability=stability_method) if info else dict(status='missing_frame'),
                             provenance=[provenance])
            candidate['pre_rank'] = _base_score(candidate)
            candidates.append(candidate)
            if digest:
                duplicates[key] = candidate
    from .teaser_edit import enabled, short_candidates
    return short_candidates(analysis,candidates) if enabled(analysis) else candidates


def rank_candidates(candidates: list[dict], goal: str, *, top_n: int = 8) -> list[dict]:
    """Fixed lexical bonus (0..10), then deterministic source-diverse round robin.

    At most one candidate per source per round; scores are independently bounded
    to 0..100, not normalized over a changing input pool. Input is not mutated.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError('top_n must be a nonnegative integer')
    tokens = set(re.findall(r'\w+', goal.casefold()))
    ranked = []
    for candidate in candidates:
        evidence = set(re.findall(r'\w+', candidate.get('transcript_snippet', '').casefold()))
        bonus = min(10, 2 * len(tokens & evidence))
        ranked.append(dict(candidate, pre_rank=round(_bounded(_base_score(candidate) + bonus), 6)))
    ranked.sort(key=lambda c: (-c['pre_rank'], c['source_id'], c['start_ms'], c['end_ms'], c['candidate_id']))
    selected = []
    while ranked and len(selected) < top_n:
        seen = set()
        rest = []
        for candidate in ranked:
            if candidate['source_id'] not in seen and len(selected) < top_n:
                selected.append(candidate)
                seen.add(candidate['source_id'])
            else:
                rest.append(candidate)
        ranked = rest
    return selected


def compact_candidates(candidates: list[dict]) -> list[dict]:
    """Keep the required planner evidence and canonical preview reuse pointers."""
    keys = ('candidate_id', 'source_id', 'start_ms', 'end_ms', 'transcript_snippet', 'quality',
            'audio', 'frame_path', 'frame_hash', 'pre_rank', 'canonical_frame_path', 'canonical_frame_hash',
            'quality_method', 'frame_time_ms', 'frame_in_window')
    return [{key: c[key] for key in keys if key in c} for c in candidates]
