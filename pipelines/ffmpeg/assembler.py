"""Word-safe source selections compiled to a minimal, source-backed V1/A1 OTIO."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
import json
from numbers import Integral, Real
from pathlib import Path
import math

import opentimelineio as otio

from .contracts import AssemblyResult, EpisodeContext, StageError, atomic_write_json, validate_narrative
from .render import render_timeline


def pad_word_interval(start_ms: int, end_ms: int, duration_ms: int,
                      words=()) -> tuple[int, int]:
    """Protect onset -80ms and decay +120ms, clamped to the original source.

    Extend to a fixed point if the new edges intersect neighbouring protected
    words. This avoids moving a cut out of one word and into its neighbour.
    Words must already be filtered to this source (source-time milliseconds).
    """
    if not 0 <= start_ms < end_ms <= duration_ms:
        raise StageError('INTERVAL_INVALID', 'assembler', 'Invalid word/source interval')
    start, end = max(0, start_ms - 80), min(duration_ms, end_ms + 120)
    protected = []
    for word in words:
        a, b = word['start_ms'], word['end_ms']
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration_ms):
            raise StageError('WORDS_INVALID', 'assembler', 'Invalid source-time word interval')
        protected.append((max(0, math.floor(a - 80)), min(duration_ms, math.ceil(b + 120))))
    while True:
        previous = start, end
        for a, b in protected:
            if a < end and start < b:
                start, end = min(start, a), max(end, b)
        if previous == (start, end):
            return start, end


def _range(start, end):
    return otio.opentime.TimeRange(otio.opentime.RationalTime(start, 1000),
                                 otio.opentime.RationalTime(end - start, 1000))


def _plain_metadata(value):
    """Keep only JSON primitives before crossing the OTIO native boundary.

    bool must precede Integral because it is an int subclass. Nested mappings
    and tuples become plain dict/list objects, never OTIO/Python proxy values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Integral):
        result = int(value)
        if not -(2**63) <= result < 2**63:
            raise ValueError('Metadata integer exceeds OTIO int64 range')
        return result
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Metadata numbers must be finite')
        return result
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('Metadata keys must be strings')
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_plain_metadata(item) for item in value]
    raise TypeError('Unsupported metadata value type')


def _with_metadata(obj, metadata):
    """Attach metadata with OTIO's native JSON reader, not constructor PyAny casts.

    Some OTIO 0.18.1 Mac/Python builds raise ``bad any cast`` converting a Python
    mapping passed as constructor metadata. Normalizing its scalar values alone
    cannot fix a mapping-conversion failure. Construct without metadata, then
    let OTIO deserialize its own canonical schema with plain JSON metadata.
    This also handles Clip's nested selection mappings through the same boundary.
    """
    try:
        clean = _plain_metadata(metadata)
        if not isinstance(clean, dict):
            raise TypeError('Metadata must be a mapping')
        payload = json.loads(otio.adapters.write_to_string(obj, 'otio_json'))
        payload['metadata'] = clean
        return otio.adapters.read_from_string(json.dumps(payload, allow_nan=False), 'otio_json')
    except Exception as exc:
        raise StageError('METADATA_INVALID', 'assembler',
                         'OTIO metadata normalization or native JSON construction failed') from exc


def build_timeline(spec: dict, analysis: dict) -> tuple[otio.schema.Timeline, dict]:
    """Return (timeline, amended deep-copy spec); raw narrative is never edited.

    Render intervals are SOURCE intervals, not output positions. Sequential,
    forward-ordered selections with overlapping protection coalesce; reorderings
    and intervening sources stay separate. Clip metadata retains every selection
    and its mapping into the output timeline.
    """
    validate_narrative(spec, analysis)
    amended = deepcopy(spec)
    sources = {s['source_id']: s for s in amended['sources']}
    words_by_source = {sid: [] for sid in sources}
    for word in analysis.get('words', []):
        sid = word.get('source_id')
        if sid not in sources:
            raise StageError('WORDS_INVALID', 'assembler', 'Word references unknown source', sid)
        # Validate even words outside the selected ranges.
        pad_word_interval(word['start_ms'], word['end_ms'], sources[sid]['duration_ms'])
        words_by_source[sid].append(word)
    groups = []
    for segment in amended['segments']:
        sid = segment['source_id']
        a, b = segment['raw_start_ms'], segment['raw_end_ms']
        words = words_by_source[sid]
        selected = [w for w in words if w['start_ms'] <= b and a <= w['end_ms']]
        if selected:
            # Pad WORD boundaries, not unrelated visual edges. Keep the raw
            # selection when it already extends beyond the protected speech.
            word_start, word_end = pad_word_interval(
                min(w['start_ms'] for w in selected), max(w['end_ms'] for w in selected),
                sources[sid]['duration_ms'], words)
            a, b = min(a, word_start), max(b, word_end)
        segment['render_start_ms'], segment['render_end_ms'] = a, b
        previous = groups[-1] if groups else None
        if (previous and previous['source_id'] == sid
                and segment['raw_start_ms'] >= previous['selections'][-1]['raw_end_ms']
                and a <= previous['end_ms']):
            previous['end_ms'] = max(previous['end_ms'], b)
            previous['start_ms'] = min(previous['start_ms'], a)
            previous['selections'].append(segment)
        else:
            groups.append(dict(source_id=sid, start_ms=a, end_ms=b, selections=[segment]))
    timeline = _with_metadata(otio.schema.Timeline(name=amended['episode_id']), {
        'schema_version': '1.0', 'aspect_ratio': amended['aspect_ratio'],
        'target_duration_ms': amended['target_duration_ms'],
        'framing_mode': 'center_crop' if amended['aspect_ratio']=='9:16' else 'fit_pad'})
    if spec.get('voiceover'):
        metadata=_plain_metadata(timeline.metadata)
        metadata['voiceover']=_plain_metadata(spec['voiceover'])
        timeline=_with_metadata(timeline,metadata)
    video = otio.schema.Track(name='V1', kind=otio.schema.TrackKind.Video)
    audio = otio.schema.Track(name='A1', kind=otio.schema.TrackKind.Audio)
    timeline.tracks.extend([video, audio])
    cursor = 0
    for index, group in enumerate(groups):
        source = sources[group['source_id']]
        a, b = group['start_ms'], group['end_ms']
        mappings = [dict(segment_id=s['segment_id'], raw_start_ms=s['raw_start_ms'],
                         raw_end_ms=s['raw_end_ms'], render_start_ms=s['render_start_ms'],
                         render_end_ms=s['render_end_ms'],
                         timeline_start_ms=cursor + s['render_start_ms'] - a,
                         timeline_end_ms=cursor + s['render_end_ms'] - a)
                    for s in group['selections']]
        metadata = dict(source_id=source['source_id'], has_audio=source['has_audio'],
                        selection_ids=[s['segment_id'] for s in group['selections']],
                        selection_mappings=mappings)
        reference = otio.schema.ExternalReference(target_url=source['path'],
                        available_range=_range(0, source['duration_ms']))
        clip = _with_metadata(otio.schema.Clip(name=f'cut-{index + 1:03d}', media_reference=reference,
                               source_range=_range(a, b)), metadata)
        video.append(clip)
        audio.append(clip.deepcopy())
        cursor += b - a
    if abs(cursor - amended['target_duration_ms']) > 1000:
        amended['coverage']['status'] = 'partial'
        amended['coverage']['limitations'].append(
            f'Word-safe source selections total {cursor}ms; requested target '
            f'{amended["target_duration_ms"]}ms cannot be achieved within ±1000ms without changing the narrative')
    validate_narrative(amended, analysis)
    return timeline, amended


def assemble(ctx: EpisodeContext, spec: dict, analysis: dict, *, attempt: int = 1) -> AssemblyResult:
    """Persist the OTIO, render its readback, and return canonical artifact paths."""
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise StageError('ATTEMPT_INVALID', 'assembler', 'Attempt must be a positive integer')
    if spec.get('episode_id') != ctx.episode_id or spec.get('goal') != ctx.goal:
        raise StageError('PLAN_INVALID', 'assembler', 'Plan identity differs from context')
    from .disk_safety import require_free_disk
    require_free_disk(ctx.episode_dir,stage='assembler')
    timeline, amended = build_timeline(spec, analysis)
    from .overlays import compile_cards,burn,preflight
    if spec.get('overlays'):preflight()
    amended=compile_cards(amended,analysis,timeline,ctx.episode_dir)
    if amended.get("overlays"):
        metadata=_plain_metadata(timeline.metadata);metadata["overlays"]=_plain_metadata(amended["overlays"])
        timeline=_with_metadata(timeline,metadata)
    directory = Path(ctx.episode_dir) / 'work' / f'attempt-{attempt}'
    directory.mkdir(parents=True, exist_ok=True)
    timeline_path = directory / 'timeline.otio'
    spec_path = directory / 'narrative-spec.json'
    candidate_path = directory / 'candidate.mp4'
    try:
        otio.adapters.write_to_file(timeline, str(timeline_path))
        # The saved artifact, never the in-memory compiler output, is render SSOT.
        saved = otio.adapters.read_from_file(str(timeline_path))
    except Exception as exc:
        raise StageError('TIMELINE_IO', 'assembler', f'OTIO persistence failed: {exc}') from exc
    atomic_write_json(spec_path, amended)
    metadata = render_timeline(saved, candidate_path, episode_dir=ctx.episode_dir)
    if amended.get('overlays'):
        overlaid=directory/'overlaid.mp4'
        burn(candidate_path,overlaid,amended['overlays']);overlaid.replace(candidate_path)
        atomic_write_json(spec_path,amended)
    if abs(metadata['duration_ms'] - amended['target_duration_ms']) > 1000:
        limitation = (f'Rendered duration {metadata["duration_ms"]}ms is outside '
                      f'target {amended["target_duration_ms"]}ms ±1000ms')
        amended['coverage']['status'] = 'partial'
        amended['coverage']['limitations'].append(limitation)
        atomic_write_json(spec_path, amended)
    return AssemblyResult(ctx.episode_id, candidate_path, timeline_path, spec_path,
                          metadata['cut_times_ms'], metadata['duration_ms'],
                          list(amended['coverage']['limitations']))
