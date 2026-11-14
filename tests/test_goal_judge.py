"""Judge tests use offline provider fakes and temporary synthetic rendered video."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from PIL import Image
import pytest

from pipelines.ffmpeg.contracts import AssemblyResult, EpisodeContext, unavailable_judge
from pipelines.ffmpeg.goal_judge import judge_goal
from pipelines.ffmpeg.llm_provider import LLMResult, ProviderError
from tests.test_planner import analysis_for


class JudgeProvider:
    provider = 'fake'
    model = 'offline'

    def __init__(self, data=None, error=None):
        self.data = data if data is not None else dict(score=82, rationale='Rendered frames show the blue sea', evidence=['Rendered image 0: blue sea', 'Selected transcript: beach'])
        self.error = error
        self.calls = []

    def complete_json(self, **kwargs):
        kwargs['sizes'], kwargs['pixels'] = [], []
        for path in kwargs['images']:
            with Image.open(path) as image:
                kwargs['sizes'].append(image.size)
                kwargs['pixels'].append(image.getpixel((0, 0)))
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMResult(deepcopy(self.data), dict(provider='fake', model='offline'))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'FRAMEFOUNDRY_LLM_PROVIDER'):
        monkeypatch.delenv(key, raising=False)
    with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No live HTTP')):
        yield


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    path = tmp_path_factory.mktemp('rendered') / 'candidate.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'lavfi', '-i', 'color=c=blue:s=1280x720:r=1:d=10', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True, capture_output=True, timeout=30)
    return path


def inputs(tmp_path, candidate=None):
    analysis = analysis_for('family beach highlights')
    spec = dict(episode_id='test', goal=analysis['goal'], sources=analysis['sources'],
        segments=[dict(segment_id='seg-001', source_id='source-0', raw_start_ms=1000, raw_end_ms=3000,
                       render_start_ms=920, render_end_ms=3120, reason='Beach arrival')])
    ctx = EpisodeContext('test', tmp_path, analysis['goal'])
    result = AssemblyResult('test', candidate or tmp_path/'missing.mp4', tmp_path/'timeline.otio', tmp_path/'narrative-spec.json', [], 10000)
    return ctx, spec, result


def test_no_key_no_call_no_frame_sampling(tmp_path):
    ctx, spec, result = inputs(tmp_path)
    with patch('pipelines.ffmpeg.goal_judge._rendered_frames', side_effect=AssertionError('No decoding')):
        assert judge_goal(ctx, spec, result) == unavailable_judge()
    with patch('pipelines.ffmpeg.goal_judge.LLMProvider.from_env', side_effect=AssertionError('No env discovery')):
        assert judge_goal(ctx, spec, result, provider=None) == unavailable_judge()


def test_configuration_error_is_failed_null_score(tmp_path):
    ctx, spec, result = inputs(tmp_path)
    with patch('pipelines.ffmpeg.goal_judge.LLMProvider.from_env', side_effect=ProviderError('secret')):
        data = judge_goal(ctx, spec, result)
    assert data['status'] == 'failed' and data['score'] is None
    assert 'secret' not in json.dumps(data)


def test_samples_rendered_output_not_original_or_contact_sheet(tmp_path, rendered):
    ctx, spec, result = inputs(tmp_path, rendered)
    # Excluded/source evidence is red. The judge must see only the BLUE rendered output.
    source_image = tmp_path/'excluded-original.jpg'
    contact_sheet = tmp_path/'contact-sheet.jpg'
    Image.new('RGB', (1000, 1000), 'red').save(source_image)
    Image.new('RGB', (4000, 4000), 'red').save(contact_sheet)
    spec['sources'][0]['keyframes'] = [dict(path=source_image.name, source_time_ms=0, proxy_time_ms=0)]
    words = [dict(source_id='source-0', text='beach', start_ms=1000, end_ms=1100),
             dict(source_id='source-0', text='excluded-before', start_ms=0, end_ms=100),
             dict(source_id='source-0', text='excluded-after', start_ms=4000, end_ms=4100),
             dict(source_id='source-1', text='excluded-source', start_ms=1000, end_ms=1100)]
    provider = JudgeProvider()
    original = deepcopy(spec)
    with patch('pipelines.ffmpeg.goal_judge.LLMProvider.from_env', return_value=provider):
        data = judge_goal(ctx, spec, result, words=words)
    assert data == dict(status='scored', score=82, rationale=provider.data['rationale'], evidence=provider.data['evidence'], provider='fake', model='offline')
    assert spec == original  # Advisory only: never mutates hard gates/coverage.
    call = provider.calls[0]
    assert len(call['images']) == 8
    assert all(max(size) <= 640 for size in call['sizes'])
    assert all(pixel[2] > 200 and pixel[0] < 10 for pixel in call['pixels'])
    payload = json.loads(call['prompt'])
    assert [w['text'] for w in payload['selected_transcript']] == ['beach']
    assert payload['goal'] == ctx.goal and payload['plan_segments'] == spec['segments']
    assert payload['images'][0]['render_time_ms'] == 625
    assert call['role'] == 'judge' and call['max_output_tokens'] == 2048
    assert 'UNTRUSTED DATA' in call['system']


@pytest.mark.parametrize('data', [dict(score=-1, rationale='x', evidence=['x']),
    dict(score=101, rationale='x', evidence=['x']), dict(score=True, rationale='x', evidence=['x']),
    dict(score=None, rationale='x', evidence=['x']), dict(score=float('nan'), rationale='x', evidence=['x']),
    dict(score=70, rationale=' ', evidence=['x']), dict(score=70, rationale='x', evidence=[]),
    dict(score=70, rationale='x', evidence=[' ']), dict(score=70, rationale='x', evidence=['x'], extra=1), {}])
def test_invalid_response_null_failed(tmp_path, data):
    ctx, spec, result = inputs(tmp_path)
    provider = JudgeProvider(data)
    with patch('pipelines.ffmpeg.goal_judge._rendered_frames', return_value=([], [])):
        judged = judge_goal(ctx, spec, result, provider=provider)
    assert judged['status'] == 'failed' and judged['score'] is None
    assert len(provider.calls) == 1


@pytest.mark.parametrize('score', [0, 59, 60, 100])
def test_scores_never_change_plan_coverage_or_hard_gates(tmp_path, score):
    ctx, spec, result = inputs(tmp_path)
    spec['coverage'] = dict(status='full', limitations=[])
    provider = JudgeProvider(dict(score=score, rationale='Evidence-based opinion', evidence=['Rendered image 0']))
    with patch('pipelines.ffmpeg.goal_judge._rendered_frames', return_value=([], [])):
        data = judge_goal(ctx, spec, result, provider=provider)
    assert data['score'] == score and data['status'] == 'scored'
    assert spec['coverage']['status'] == 'full'


def test_provider_error_safe_and_usage_not_aggregated_here(tmp_path):
    ctx, spec, result = inputs(tmp_path)
    provider = JudgeProvider(error=ProviderError('private-key'))
    with patch('pipelines.ffmpeg.goal_judge._rendered_frames', return_value=([], [])):
        data = judge_goal(ctx, spec, result, provider=provider)
    assert data['status'] == 'failed' and data['score'] is None
    assert 'private-key' not in json.dumps(data)
    assert not (tmp_path/'llm-usage.json').exists()  # Fake provider: caller owns aggregation.


def test_missing_output_does_not_call_provider(tmp_path):
    ctx, spec, result = inputs(tmp_path)
    provider = JudgeProvider()
    data = judge_goal(ctx, spec, result, provider=provider)
    assert data['status'] == 'failed' and data['score'] is None
    assert provider.calls == []


def test_source_raw_ranges_used_if_render_ranges_not_yet_recorded(tmp_path):
    ctx, spec, result = inputs(tmp_path)
    spec['segments'][0].update(render_start_ms=None, render_end_ms=None)
    words = [dict(source_id='source-0', text='selected', start_ms=1000, end_ms=1100)]
    provider = JudgeProvider()
    with patch('pipelines.ffmpeg.goal_judge._rendered_frames', return_value=([], [])):
        judge_goal(ctx, spec, result, provider=provider, words=words)
    assert json.loads(provider.calls[0]['prompt'])['selected_transcript'][0]['text'] == 'selected'
