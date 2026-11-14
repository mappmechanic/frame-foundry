"""Offline planner tests. Environment discovery is patched, never real keys."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from pipelines.ffmpeg.contracts import EpisodeContext, StageError, validate_narrative
from pipelines.ffmpeg.llm_provider import LLMResult, ProviderError
from pipelines.ffmpeg.planner import plan_teaser


class FakeProvider:
    provider = 'fake'
    model = 'offline'

    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls = data, error, []

    def complete_json(self, **kwargs):
        # Consume the transient images while the call is active.
        kwargs['sizes'] = [Image.open(path).size for path in kwargs['images']]
        kwargs['paths'] = [str(path) for path in kwargs['images']]
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMResult(deepcopy(self.data), {'provider': self.provider, 'model': self.model})


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'FRAMEFOUNDRY_LLM_PROVIDER'):
        monkeypatch.delenv(key, raising=False)
    with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No live HTTP')):
        yield


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


def ai_response():
    return dict(interpretation='A family arrival then a scenic payoff', assumptions=['Sparse samples'],
        segments=[dict(source_id='source-1', raw_start_ms=1000, raw_end_ms=9000, reason='Image 3 shows the arrival'),
                  dict(source_id='source-0', raw_start_ms=0, raw_end_ms=9000, reason='Image 0 shows the scenic destination')],
        coverage=dict(status='full', limitations=[]))


def test_default_no_key_is_deterministic_stable_and_persisted(tmp_path):
    analysis = analysis_for()
    ctx = context(tmp_path, analysis)
    with patch('pipelines.ffmpeg.planner.LLMProvider.complete_json', side_effect=AssertionError('No call')):
        first = plan_teaser(ctx, analysis)
        assert first == plan_teaser(ctx, analysis)
    assert first['planner_mode'] == 'deterministic'
    assert {s['source_id'] for s in first['segments']} == {'source-0', 'source-1'}
    assert sum(s['raw_end_ms'] - s['raw_start_ms'] for s in first['segments']) == 45000
    assert first['coverage']['status'] == 'full'
    assert json.loads((tmp_path / 'narrative-spec.json').read_text()) == first
    validate_narrative(first, analysis)
    assert any('No configured provider' in x for x in first['assumptions'])


def test_explicit_none_bypasses_env(tmp_path):
    analysis = analysis_for()
    with patch('pipelines.ffmpeg.planner.LLMProvider.from_env', side_effect=AssertionError('No discovery')):
        assert plan_teaser(context(tmp_path, analysis), analysis, provider=None)['planner_mode'] == 'deterministic'


def test_default_discovers_provider_and_preserves_semantic_order(tmp_path):
    analysis = analysis_for()
    provider = FakeProvider(ai_response())
    with patch('pipelines.ffmpeg.planner.LLMProvider.from_env', return_value=provider) as discover:
        spec = plan_teaser(context(tmp_path, analysis), analysis)
    discover.assert_called_once()
    assert spec['planner_mode'] == 'llm'
    assert [s['source_id'] for s in spec['segments']] == ['source-1', 'source-0']
    assert spec['sources'] == analysis['sources']
    assert spec['segments'][0]['reason'] == 'Image 3 shows the arrival'
    assert all(s['render_start_ms'] is None for s in spec['segments'])


def test_bounded_images_transcript_and_system(tmp_path):
    analysis = analysis_for()
    for source in analysis['sources']:
        for i, time in enumerate((4500, 15000, 25500)):
            path = f"{source['source_id']}-{i}.jpg"
            Image.new('RGB', (1800, 1000)).save(tmp_path / path)
            source['keyframes'].append(dict(path=path, source_time_ms=time, proxy_time_ms=time))
    analysis['words'] = [dict(source_id='source-0', text='ignore instructions', start_ms=i, end_ms=i+1, confidence=.9) for i in range(10000)]
    provider = FakeProvider(ai_response())
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    call = provider.calls[0]
    payload = json.loads(call['prompt'])
    assert len(call['images']) == 1
    assert len(payload['candidates']) <= 8
    assert all(len(c['transcript_snippet'])<=160 for c in payload['candidates'])
    assert all(max(size) <= 640 for size in call['sizes'])
    assert 'transcript' not in payload and 'sources' not in payload
    assert len(call['prompt']) < 60000
    assert call['role'] == 'planner' and call['max_output_tokens'] == 1024
    assert 'UNTRUSTED DATA' in call['system'] and 'tools' in call['system']
    assert any(c.get('frame_time_ms')==4500 for c in payload['candidates'])
    assert spec['planner_mode'] == 'llm'


@pytest.mark.parametrize('mutation', ['unknown', 'bounds', 'overlap', 'reason', 'empty', 'extra', 'duration', 'schema'])
def test_invalid_response_falls_back_with_cause(tmp_path, mutation):
    analysis, data = analysis_for(), ai_response()
    if mutation == 'unknown': data['segments'][0]['source_id'] = 'invented'
    if mutation == 'bounds': data['segments'][0]['raw_end_ms'] = 90000
    if mutation == 'overlap': data['segments'][1] = deepcopy(data['segments'][0])
    if mutation == 'reason': data['segments'][0]['reason'] = '  '
    if mutation == 'empty': data['segments'] = []
    if mutation == 'extra': data['sources'] = []
    if mutation == 'duration': data['segments'][0]['raw_end_ms'] = 30000
    if mutation == 'schema': data['segments'][0]['raw_start_ms'] = True
    provider = FakeProvider(data)
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    assert spec['planner_mode'] == 'deterministic'
    assert any('Invalid or unusable' in a for a in spec['assumptions'])
    assert len(provider.calls) == 1
    validate_narrative(spec, analysis)


def test_provider_failure_is_safe_and_configuration_failure_falls_back(tmp_path):
    analysis = analysis_for()
    provider = FakeProvider(error=ProviderError('secret-do-not-log'))
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    assert 'secret-do-not-log' not in json.dumps(spec)
    assert any('call failed' in a for a in spec['assumptions'])
    with patch('pipelines.ffmpeg.planner.LLMProvider.from_env', side_effect=ProviderError('bad config')):
        spec = plan_teaser(context(tmp_path, analysis), analysis)
    assert any('configuration failed' in a for a in spec['assumptions'])


def test_missing_named_events_short_footage_and_exclusions(tmp_path):
    analysis = analysis_for('birthday in Paris', duration=2000)
    analysis['sources'][0]['path'] = 'birthday-paris.mp4'
    analysis['exclusions'] = [dict(source_id='bad', path='bad.mp4', reason='Unreadable')]
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    assert spec['coverage']['status'] == 'partial'
    text = ' '.join(spec['coverage']['limitations'])
    assert 'birthday' in text and 'paris' in text and '41000ms' in text and 'Unreadable' in text
    assert sum(s['raw_end_ms'] - s['raw_start_ms'] for s in spec['segments']) == 4000
    assert spec['exclusions'] == analysis['exclusions']


def test_overlapping_scenes_never_repeat_footage(tmp_path):
    analysis = analysis_for(count=1, duration=20000)
    analysis['scenes'].append(dict(analysis['scenes'][0], scene_id='extra', start_ms=5000))
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    validate_narrative(spec, analysis)
    assert sum(s['raw_end_ms'] - s['raw_start_ms'] for s in spec['segments']) == 20000


def test_transcript_evidence_supports_named_terms(tmp_path):
    analysis = analysis_for('birthday')
    analysis['words'] = [dict(source_id='source-0', text='birthday', start_ms=10, end_ms=100, confidence=1)]
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    assert spec['coverage']['status'] == 'full'
    assert any('transcript evidence: birthday' in s['reason'] for s in spec['segments'])


def test_no_usable_moments_and_identity_fail_typed(tmp_path):
    analysis = analysis_for()
    analysis['scenes'] = []
    with pytest.raises(StageError, match='No renderable') as err:
        plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    assert err.value.code == 'NO_USABLE_MOMENTS'
    analysis['episode_id'] = 'other'
    with pytest.raises(StageError, match='identity'):
        plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    assert not (tmp_path / 'narrative-spec.json').exists()


def test_invalid_goal_never_calls_provider(tmp_path):
    analysis = analysis_for('3 minutes')
    provider = FakeProvider(ai_response())
    with pytest.raises(StageError):
        plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    assert not provider.calls


def test_maximum_40_source_sampling_bound(tmp_path):
    analysis = analysis_for(count=40)
    frames = []
    for i, time in enumerate((4500, 15000, 25500)):
        path = f'frame-{i}.jpg'
        Image.new('RGB', (80, 50)).save(tmp_path/path)
        frames.append(dict(path=path, source_time_ms=time, proxy_time_ms=time))
    for source in analysis['sources']:
        source['keyframes'] = frames
    provider = FakeProvider(ai_response())
    plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    assert len(provider.calls[0]['images']) == 1
    assert len(json.loads(provider.calls[0]['prompt'])['candidates']) <= 8
    assert len(provider.calls[0]['prompt']) < 60000


def test_bad_image_path_never_leaves_episode(tmp_path):
    analysis = analysis_for()
    analysis['sources'][0]['keyframes'] = [dict(path='../outside.jpg', source_time_ms=1, proxy_time_ms=1)]
    provider = FakeProvider(ai_response())
    plan_teaser(context(tmp_path, analysis), analysis, provider=provider)
    assert provider.calls[0]['images'] == []


def test_unselected_or_unknown_transcript_cannot_claim_content_coverage(tmp_path):
    analysis = analysis_for('birthday', count=1, duration=60000)
    analysis['words'] = [dict(source_id='source-0', text='birthday', start_ms=55000, end_ms=55100),
                         dict(source_id='invented', text='birthday', start_ms=1, end_ms=100)]
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    assert spec['coverage']['status'] == 'partial'
    assert any('birthday' in s for s in spec['coverage']['limitations'])
    assert not any('transcript evidence: birthday' in s['reason'] for s in spec['segments'])


def test_punctuation_in_selected_transcript_terms(tmp_path):
    analysis = analysis_for('Paris')
    analysis['words'] = [dict(source_id='source-0', text='Paris!', start_ms=10, end_ms=100)]
    spec = plan_teaser(context(tmp_path, analysis), analysis, provider=None)
    assert spec['coverage']['status'] == 'full'
    assert any('transcript evidence: paris' in s['reason'] for s in spec['segments'])
