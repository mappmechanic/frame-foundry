"""Independent canonical-contract tests for word protection and OTIO assembly."""
from copy import deepcopy
import json
from pathlib import Path

import opentimelineio as otio
import pytest

from pipelines.ffmpeg import assembler
from pipelines.ffmpeg.contracts import EpisodeContext, StageError, validate_narrative


def narrative(tmp_path):
    sources = [dict(source_id=sid, path=str(tmp_path / f'{sid}.mp4'), duration_ms=10000,
                    fps_num=30, fps_den=1, width=160, height=90, has_audio=True,
                    creation_time=None, proxy_path=f'proxies/{sid}.mp4', proxy_duration_ms=10000,
                    proxy_time_offset_ms=0, rotation_degrees=0, keyframes=[])
               for sid in ('a', 'b')]
    spec = dict(schema_version='1.0', episode_id='episode', goal='teaser', planner_mode='deterministic',
                interpretation='Source highlights', assumptions=[], target_duration_ms=5000,
                aspect_ratio='9:16', sources=sources, segments=[],
                coverage=dict(status='full', limitations=[]), exclusions=[])
    return spec, dict(episode_id='episode', goal='teaser', sources=deepcopy(sources), words=[])


def selection(sid, start, end, identity):
    return dict(segment_id=identity, source_id=sid, raw_start_ms=start, raw_end_ms=end,
                render_start_ms=None, render_end_ms=None, reason='Source-backed highlight')


def test_exact_word_padding_and_source_bounds():
    assert assembler.pad_word_interval(1000, 1300, 2000) == (920, 1420)
    assert assembler.pad_word_interval(30, 1980, 2000) == (0, 2000)
    with pytest.raises(StageError):
        assembler.pad_word_interval(10, 2001, 2000)


def test_adjacent_word_closure_is_order_independent():
    words = [dict(start_ms=900, end_ms=1000), dict(start_ms=1060, end_ms=1200),
             dict(start_ms=1270, end_ms=1400), dict(start_ms=2000, end_ms=2200)]
    assert assembler.pad_word_interval(1060, 1200, 3000, words) == (820, 1520)
    assert assembler.pad_word_interval(1060, 1200, 3000, words[::-1]) == (820, 1520)


def test_coalesce_preserves_narrative_and_mapping(tmp_path):
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 1000, 1300, 'one'), selection('a', 1360, 1600, 'two'),
                        selection('b', 1000, 2000, 'three')]
    analysis['words'] = [dict(source_id='a', start_ms=1000, end_ms=1300, text='first'),
                         dict(source_id='a', start_ms=1360, end_ms=1600, text='second')]
    original = deepcopy(spec)
    timeline, amended = assembler.build_timeline(spec, analysis)
    assert spec == original
    validate_narrative(amended, analysis)
    assert [track.name for track in timeline.tracks] == ['V1', 'A1']
    assert len(timeline.tracks[0]) == 2
    clip = timeline.tracks[0][0]
    assert list(clip.metadata['selection_ids']) == ['one', 'two']
    assert clip.source_range.start_time.to_seconds() == pytest.approx(.92)
    assert clip.source_range.duration.to_seconds() == pytest.approx(.8)
    assert clip.media_reference.target_url == spec['sources'][0]['path']
    assert len(clip.metadata['selection_mappings']) == 2
    assert clip.metadata['selection_mappings'][1]['raw_start_ms'] == 1360
    assert amended['segments'][0]['raw_start_ms'] == 1000
    assert amended['segments'][0]['render_end_ms'] == 1720
    assert amended['coverage']['status'] == 'partial'
    assert '±1000ms' in amended['coverage']['limitations'][-1]


def test_timeline_order_readback_multiple_cuts_and_no_reverse_coalescing(tmp_path):
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 5000, 6000, 'one'), selection('b', 0, 2000, 'two'),
                        selection('a', 0, 1000, 'three'), selection('a', 2000, 3000, 'four')]
    timeline, amended = assembler.build_timeline(spec, analysis)
    path = tmp_path / 'timeline.otio'
    otio.adapters.write_to_file(timeline, str(path))
    saved = otio.adapters.read_from_file(str(path))
    assert [c.metadata['source_id'] for c in saved.tracks[0]] == ['a', 'b', 'a', 'a']
    assert [c.source_range.start_time.to_seconds() for c in saved.tracks[0]] == [5, 0, 0, 2]
    assert amended['coverage']['status'] == 'full'
    spec['segments'] = [selection('a', 1100, 1300, 'late'), selection('a', 900, 1000, 'early')]
    analysis['words'] = [dict(source_id='a', start_ms=900, end_ms=1300, text='word')]
    timeline, _ = assembler.build_timeline(spec, analysis)
    assert len(timeline.tracks[0]) == 2


@pytest.mark.parametrize('damage', ['source', 'word', 'interval'])
def test_invalid_references_and_intervals(tmp_path, damage):
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 1000, 2000, 'one')]
    if damage == 'source':
        spec['segments'][0]['source_id'] = 'missing'
    elif damage == 'word':
        analysis['words'] = [dict(source_id='missing', start_ms=1, end_ms=2, text='bad')]
    else:
        spec['segments'][0]['raw_end_ms'] = 10001
    with pytest.raises(StageError):
        assembler.build_timeline(spec, analysis)


def test_assemble_reads_saved_timeline_as_render_ssot(tmp_path, monkeypatch):
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 1000, 6000, 'one')]
    write = otio.adapters.write_to_file
    def changed_write(timeline, path):
        for track in timeline.tracks:
            track[0].name = 'disk-only-marker'
        return write(timeline, path)
    monkeypatch.setattr(otio.adapters, 'write_to_file', changed_write)
    def render(saved, path, *, episode_dir):
        assert saved.tracks[0][0].name == 'disk-only-marker'
        assert Path(path) == tmp_path / 'work/attempt-2/candidate.mp4'
        return dict(cut_times_ms=[], duration_ms=5000)
    monkeypatch.setattr(assembler, 'render_timeline', render)
    result = assembler.assemble(EpisodeContext('episode', tmp_path, 'teaser'), spec, analysis, attempt=2)
    assert result.duration_ms == 5000
    assert result.timeline_path.is_file()
    assert json.loads(result.spec_path.read_text())['segments'][0]['render_start_ms'] == 1000


def test_visual_edges_outside_speech_are_not_unnecessarily_padded(tmp_path):
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 0, 5000, 'one')]
    analysis['words'] = [dict(source_id='a', start_ms=1000, end_ms=2000, text='inside')]
    timeline, amended = assembler.build_timeline(spec, analysis)
    assert amended['segments'][0]['render_start_ms'] == 0
    assert amended['segments'][0]['render_end_ms'] == 5000
    assert timeline.duration().to_seconds() == 5


def test_exact_word_edges_are_protected_even_at_selection_boundary(tmp_path):
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 2000, 4000, 'one')]
    analysis['words'] = [dict(source_id='a', start_ms=1900, end_ms=2000, text='preceding'),
                         dict(source_id='a', start_ms=4000, end_ms=4100, text='following')]
    _, amended = assembler.build_timeline(spec, analysis)
    assert amended['segments'][0]['render_start_ms'] == 1820
    assert amended['segments'][0]['render_end_ms'] == 4220


def test_metadata_avoids_constructor_mapping_cast_and_roundtrips(tmp_path, monkeypatch):
    """Emulate the Mac 0.18.1 binding failure, including Clip's nested mappings."""
    spec, analysis = narrative(tmp_path)
    spec['segments'] = [selection('a', 1000, 4000, 'one'), selection('b', 0, 2000, 'two')]
    original = deepcopy(spec)
    constructors = {name: getattr(otio.schema, name) for name in ('Timeline', 'Clip')}
    calls = []
    for name, constructor in constructors.items():
        def reject_metadata(*args, _name=name, _constructor=constructor, **kwargs):
            calls.append((_name, kwargs.copy()))
            if 'metadata' in kwargs:
                raise RuntimeError('bad any cast')
            return _constructor(*args, **kwargs)
        monkeypatch.setattr(otio.schema, name, reject_metadata)
    timeline, amended = assembler.build_timeline(spec, analysis)
    path = tmp_path / 'roundtrip.otio'
    otio.adapters.write_to_file(timeline, str(path))
    saved = otio.adapters.read_from_file(str(path))
    assert spec == original and calls and all('metadata' not in kw for _, kw in calls)
    assert dict(saved.metadata) == dict(schema_version='1.0', aspect_ratio='9:16', target_duration_ms=5000)
    assert type(saved.metadata['target_duration_ms']) is int
    assert saved.metadata['aspect_ratio'] == amended['aspect_ratio']
    for video, audio in zip(saved.tracks[0], saved.tracks[1]):
        assert type(video.metadata['has_audio']) is bool and video.metadata['has_audio'] is True
        assert assembler._plain_metadata(video.metadata) == assembler._plain_metadata(audio.metadata)
        mappings = video.metadata['selection_mappings']
        assert len(mappings) == 1
        assert type(mappings[0]['timeline_start_ms']) is int
        assert mappings[0]['segment_id'] == list(video.metadata['selection_ids'])[0]
        assert video.source_range == audio.source_range
        assert video.media_reference.target_url == audio.media_reference.target_url


def test_metadata_normalizes_nested_subclasses_without_mutating_input():
    from collections import UserDict
    class Label(str): pass
    class Count(int): pass
    class Value(float): pass
    metadata = UserDict({Label('name'): Label('clip'), 'count': Count(5), 'active': True,
                         'nested': (UserDict({'fraction': Value(.5), 'missing': None}),)})
    clean = assembler._plain_metadata(metadata)
    assert type(clean) is dict and all(type(k) is str for k in clean)
    assert type(clean['name']) is str and type(clean['count']) is int and type(clean['active']) is bool
    assert type(clean['nested']) is list and type(clean['nested'][0]) is dict
    assert type(clean['nested'][0]['fraction']) is float and clean['nested'][0]['missing'] is None
    assert isinstance(metadata['nested'], tuple) and isinstance(metadata['count'], Count)
    # Exercise the actual native reader for every supported scalar kind.
    timeline = assembler._with_metadata(otio.schema.Timeline(name='plain'), clean)
    parsed = json.loads(otio.adapters.write_to_string(timeline, 'otio_json'))
    assert parsed['metadata'] == clean


@pytest.mark.parametrize('metadata', [{'x': float('nan')}, {'x': float('inf')},
                                       {'x': 2**63}, {1: 'invalid key'}, {'x': {1, 2}}])
def test_metadata_rejects_non_json_or_non_otio_values(metadata):
    with pytest.raises(StageError) as failure:
        assembler._with_metadata(otio.schema.Timeline(), metadata)
    assert failure.value.code == 'METADATA_INVALID'
