"""Offline regression coverage for explicit visual-only duration completion."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from pipelines.ffmpeg.contracts import EpisodeContext, validate_narrative
from pipelines.ffmpeg.planner import plan_teaser
from pipelines.ffmpeg.llm_provider import LLMResult


class FakeProvider:
    provider = 'fake'
    model = 'offline'

    def __init__(self, data):
        self.data, self.calls = data, []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResult(deepcopy(self.data), {})


def analysis_for(goal='family trip highlights', count=2, duration=30000):
    sources, scenes = [], []
    for i in range(count):
        sid = f'source-{i}'
        sources.append(dict(source_id=sid, path=f'raw/{sid}.mp4', duration_ms=duration,
            fps_num=30, fps_den=1, width=1920, height=1080, has_audio=False,
            creation_time=None, proxy_path=f'proxies/{sid}.mp4', proxy_duration_ms=duration,
            proxy_time_offset_ms=0, rotation_degrees=0, keyframes=[]))
        scenes.append(dict(scene_id=f'scene-{i}', source_id=sid, start_ms=0, end_ms=duration,
            thumbnail_path=None, brightness=.5, sharpness=.5, motion=.5, tokens=[]))
    return dict(episode_id='test', goal=goal, sources=sources, scenes=scenes, words=[], exclusions=[])


def context(tmp_path, analysis):
    return EpisodeContext('test', tmp_path, analysis['goal'])


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv('FRAMEFOUNDRY_CACHE_DIR', str(tmp_path / 'cache'))
    with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No live HTTP')):
        yield


def visual_analysis(duration=60000, count=1):
    analysis = analysis_for('45 seconds highlights', count=count, duration=duration)
    analysis['transcription_status'] = 'visual_only_unavailable'
    analysis['candidates'] = [dict(candidate_id=f'{source["source_id"]}-{start}',
        source_id=source['source_id'], start_ms=start, end_ms=min(start + 4000, duration),
        quality=dict(visual_score=100 - start / duration), audio={}, transcript_snippet='')
        for source in analysis['sources'] for start in range(0, duration, 4000)]
    return analysis


def response_for(candidates, duration):
    segments = []
    for candidate in candidates:
        length = min(duration, candidate['end_ms'] - candidate['start_ms'])
        segments.append(dict(source_id=candidate['source_id'], raw_start_ms=candidate['start_ms'],
            raw_end_ms=candidate['start_ms'] + length, reason='Chosen visual cut'))
        duration -= length
        if not duration:
            break
    assert duration == 0
    return dict(interpretation='Visual highlights', assumptions=['Sparse visual samples'],
        segments=segments, coverage=dict(status='full', limitations=[]))


def actual(spec):
    return sum(s['raw_end_ms'] - s['raw_start_ms'] for s in spec['segments'])


def assert_preserved(spec, response):
    assert [{k: s[k] for k in ('source_id', 'raw_start_ms', 'raw_end_ms', 'reason')}
            for s in spec['segments'][:len(response['segments'])]] == response['segments']


def test_29_second_provider_top_eight_only_fills_45_stably(tmp_path):
    analysis = visual_analysis()
    before = deepcopy(analysis)
    response = response_for(analysis['candidates'][:8], 29000)
    provider = FakeProvider(response)
    ctx = context(tmp_path, analysis)
    spec = plan_teaser(ctx, analysis, provider=provider)
    assert actual(spec) == 45000
    assert spec['planner_mode'] == 'llm'
    assert_preserved(spec, response)
    assert len(provider.calls) == 1
    payload = json.loads(provider.calls[0]['prompt'])
    assert len(payload['candidates']) == 8
    assert sum(c['end_ms'] - c['start_ms'] for c in payload['candidates']) == 32000
    assert 'transcript' not in payload and 'words' not in payload and 'sources' not in payload
    assert 'beyond the top-eight' in ' '.join(spec['assumptions'])
    assert 'not inspected by the LLM' in ' '.join(spec['assumptions'])
    assert analysis == before and provider.data == response
    validate_narrative(spec, analysis)
    assert spec == plan_teaser(ctx, analysis, provider=FakeProvider(response))


def test_fill_prefers_unused_sources_and_preserves_chosen_order(tmp_path):
    analysis = visual_analysis(count=3)
    response = response_for([analysis['candidates'][15], analysis['candidates'][0]], 8000)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=FakeProvider(response))
    assert_preserved(spec, response)
    assert spec['segments'][2]['source_id'] == 'source-2'
    assert actual(spec) == 45000
    validate_narrative(spec, analysis)


def test_overlapping_candidates_and_selected_cuts_never_duplicate(tmp_path):
    analysis = visual_analysis(duration=20000)
    analysis['candidates'] += [dict(analysis['candidates'][0], candidate_id='overlap',
        start_ms=1000, end_ms=7000, quality=dict(visual_score=100))]
    response = response_for(analysis['candidates'][:1], 2000)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=FakeProvider(response))
    assert_preserved(spec, response)
    assert actual(spec) == 20000
    assert spec['coverage']['status'] == 'partial'
    assert '25000ms' in ' '.join(spec['coverage']['limitations'])
    validate_narrative(spec, analysis)


def test_insufficient_usable_windows_do_not_extend_into_gaps_or_out_of_bounds(tmp_path):
    analysis = visual_analysis()
    analysis['candidates'] = [analysis['candidates'][0], analysis['candidates'][3]]
    response = response_for(analysis['candidates'][:1], 1000)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=FakeProvider(response))
    assert actual(spec) == 8000
    assert spec['coverage']['status'] == 'partial'
    assert '37000ms' in ' '.join(spec['coverage']['limitations'])
    for segment in spec['segments']:
        assert any(c['start_ms'] <= segment['raw_start_ms'] < segment['raw_end_ms'] <= c['end_ms']
                   for c in analysis['candidates'])
    validate_narrative(spec, analysis)


@pytest.mark.parametrize('status', ['local_asr', 'fixture', None, 'visual_only_unavailable'])
def test_genuine_words_and_transcript_fixtures_stay_short(tmp_path, status):
    analysis = visual_analysis()
    analysis['transcription_status'] = status
    if status in ('local_asr', 'fixture', 'visual_only_unavailable'):
        analysis['words'] = [dict(source_id='source-0', text='hello', start_ms=500,
                                  end_ms=900, confidence=1)]
    response = response_for(analysis['candidates'][:8], 29000)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=FakeProvider(response))
    assert actual(spec) == 29000
    assert len(spec['segments']) == len(response['segments'])
    assert_preserved(spec, response)
    assert not any('duration completion' in a for a in spec['assumptions'])


@pytest.mark.parametrize('status', ['local_asr', 'fixture'])
def test_empty_asr_or_explicit_fixture_not_inferred_visual_only(tmp_path, status):
    analysis = visual_analysis()
    analysis['transcription_status'] = status
    response = response_for(analysis['candidates'][:1], 1000)
    assert actual(plan_teaser(context(tmp_path, analysis), analysis,
                             provider=FakeProvider(response))) == 1000


@pytest.mark.parametrize('persisted', [None, []])
@pytest.mark.parametrize('with_provider', [False, True])
def test_missing_candidates_and_scenes_extract_source_bounded_windows(tmp_path, persisted, with_provider):
    analysis = visual_analysis(duration=50000)
    analysis['candidates'] = persisted
    analysis['scenes'] = []
    response = response_for([dict(source_id='source-0', start_ms=0, end_ms=10000)], 5000)
    provider = FakeProvider(response) if with_provider else None
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    assert actual(spec) == 45000
    validate_narrative(spec, analysis)
    if provider:
        assert len(provider.calls) == 1
        assert spec['planner_mode'] == 'llm'
        assert_preserved(spec, response)


def test_cache_separates_transcription_status_and_recompletes_locally(tmp_path):
    analysis = visual_analysis()
    response = response_for(analysis['candidates'][:8], 29000)
    ctx = context(tmp_path, analysis)
    provider = FakeProvider(response)
    # No raw media is needed: isolate only cache eligibility from file hashing.
    with patch('pipelines.ffmpeg.evaluator.footage_fingerprint', return_value='offline-footage'):
        assert actual(plan_teaser(ctx, analysis, provider=provider)) == 45000
        cached = plan_teaser(ctx, analysis, provider=provider)
        assert actual(cached) == 45000 and len(provider.calls) == 1
        assert any('cache hit' in a for a in cached['assumptions'])
        analysis['transcription_status'] = 'fixture'
        assert actual(plan_teaser(ctx, analysis, provider=provider)) == 29000
        assert len(provider.calls) == 2
        analysis['transcription_status'] = 'visual_only_unavailable'
        assert actual(plan_teaser(ctx, analysis, provider=provider)) == 45000
        assert len(provider.calls) == 2
    entries = list((tmp_path / 'cache' / 'planner').glob('*.json'))
    assert len(entries) == 2
    assert all(json.loads(entry.read_text())['data'] == response for entry in entries)
