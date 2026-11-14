"""Eval judge: reasoning-free projection, strict rubric, single call, cache, guard."""
import json

import pytest
from PIL import Image

from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.eval_gates import generate_distinct_footage, read_shots
from pipelines.ffmpeg import eval_judge
from pipelines.ffmpeg.eval_judge import CRITERIA, GUARD_NAME, build_judge_sheet, normalize, projection, run_judge
from pipelines.ffmpeg.llm_provider import DEFAULT_RATES, LLMProvider, ProviderError

GOAL = 'Create a 10 second family-trip teaser'


@pytest.fixture(scope='module')
def episode(tmp_path_factory):
    root = tmp_path_factory.mktemp('judge')
    fixture = generate_distinct_footage(root / 'footage')
    assert run_episode(root / 'footage', GOAL, episodes_root=root / 'episodes', episode_id='judged',
                       transcript_fixture=fixture, provider=None)['passed']
    return root / 'episodes' / 'judged'


def _response(ids, **over):
    data = dict(scores={c: 4 for c in CRITERIA}, evidence={c: [ids[0]] for c in CRITERIA}, weakest_shot=ids[-1],
                fix_instruction='Swap the weakest shot for a wider establishing view', verdict='pass',
                unsupported_goal=False, limitations=['stills only'])
    data.update(over)
    return data


class Transport:
    def __init__(self):
        self.bodies = []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(json.loads(body))
        prompt = json.loads(self.bodies[-1]['messages'][0]['content'][-1]['text'])
        ids = [s['shot_id'] for s in prompt['edit']['shots']]
        return 200, json.dumps(dict(content=[dict(type='text', text=json.dumps(_response(ids)))], stop_reason='end_turn',
                                    usage=dict(input_tokens=1500, output_tokens=300))).encode()


def _args(episode, tmp_path, provider, render_hash='a' * 64):
    shots, aspect = read_shots(episode)
    return dict(episode_dir=episode, evals_root=tmp_path / 'evals', provider=provider, goal=GOAL, shots=shots,
                duration_ms=10000, aspect_ratio=aspect, render_path=episode / 'finished.mp4', render_hash=render_hash,
                footage_hash='b' * 64, render_size=(1080, 1920))


@pytest.fixture
def clean(episode):
    (episode / GUARD_NAME).unlink(missing_ok=True)
    for name in ['llm-usage.json']:
        (episode / name).unlink(missing_ok=True)
    yield episode
    (episode / GUARD_NAME).unlink(missing_ok=True)
    (episode / 'llm-usage.json').unlink(missing_ok=True)


def test_projection_is_reasoning_free(episode):
    shots, aspect = read_shots(episode)
    text = json.dumps(projection(GOAL, shots, 10000, aspect))
    spec = json.loads((episode / 'narrative-spec.json').read_text())
    for forbidden in ['reason', 'assumption', 'interpretation', 'transcript', 'words', 'tokens', 'selection_ids']:
        assert forbidden not in text
    assert spec['segments'][0]['reason'] not in text
    assert set(json.loads(text)) == {'goal', 'aspect_ratio', 'duration_ms', 'shots'}


def test_sheet_is_single_labelled_tiled_render_under_640(episode, tmp_path):
    shots, _ = read_shots(episode)
    many = [dict(s, shot_id=f'shot-{i + 1:02d}') for i, s in enumerate(shots * 9)]
    for subset in (shots, many):
        info = build_judge_sheet(episode / 'finished.mp4', subset, tmp_path / 'sheet.jpg', render_w=1080, render_h=1920)
        with Image.open(tmp_path / 'sheet.jpg') as image:
            assert max(image.size) <= 640 and image.format == 'JPEG'
        assert info['tiles'] == len(subset)


def test_rubric_normalization_and_strictness():
    ids = ['shot-01', 'shot-02']
    result, j = normalize(_response(ids), ids)
    assert j == pytest.approx(0.75) and result['verdict'] == 'pass'
    result, j = normalize(_response(ids, unsupported_goal=True), ids)
    assert result['scores']['goal_fit'] == 2 and result['verdict'] == 'fail'
    assert any('clamped' in x for x in result['limitations'])
    low = _response(ids, scores={c: 2 for c in CRITERIA}, verdict='pass')
    assert normalize(low, ids)[0]['verdict'] == 'fail'
    assert normalize(_response(ids, scores={c: 1 for c in CRITERIA}), ids)[1] == 0
    for bad in [_response(ids, weakest_shot='shot-99'), _response(ids, fix_instruction='x' * 161),
                _response(ids, evidence={c: ['shot-77'] for c in CRITERIA}),
                _response(ids, scores=dict({c: 4 for c in CRITERIA}, goal_fit=6)), dict(_response(ids), extra=1)]:
        with pytest.raises(Exception):
            normalize(bad, ids)


def test_single_call_ledger_cache_and_guard(clean, tmp_path):
    transport = Transport()
    provider = LLMProvider('anthropic', 'sk-test', 'claude-sonnet-4-5', DEFAULT_RATES['claude-sonnet-4-5'],
                           transport=transport)
    first = run_judge(**_args(clean, tmp_path, provider))
    assert first['status'] == 'scored' and not first['cache_hit'] and first['J'] == pytest.approx(0.75)
    body = transport.bodies[0]
    assert body['max_tokens'] == 1024 and len([c for c in body['messages'][0]['content'] if c['type'] == 'image']) == 1
    ledger = json.loads((clean / 'llm-usage.json').read_text())
    assert [c['role'] for c in ledger['calls']] == ['judge'] and ledger['calls'][0]['status'] == 'success'
    second = run_judge(**_args(clean, tmp_path, provider))
    assert second['cache_hit'] and second['J'] == first['J'] and len(transport.bodies) == 1
    # Cache gone (or corrupt): the durable guard still prevents a second dispatch.
    for path in (tmp_path / 'evals' / '.inference-cache' / 'judge').iterdir():
        path.write_text('{corrupt')
    third = run_judge(**_args(clean, tmp_path, provider))
    assert third['status'] == 'unavailable' and third['scores'] is None and len(transport.bodies) == 1
    changed = run_judge(**_args(clean, tmp_path, provider, render_hash='c' * 64))
    assert changed['status'] == 'unavailable' and 'different render' in changed['reason'] and len(transport.bodies) == 1


def test_corrupt_cache_is_a_miss_not_a_score(clean, tmp_path):
    key = eval_judge.cache_key(footage_hash='b' * 64, goal=GOAL, render_hash='a' * 64, provider='anthropic',
                               model='claude-sonnet-4-5')
    cache = tmp_path / 'evals' / '.inference-cache' / 'judge' / f'{key}.json'
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(dict(cache_key=key, prompt_version=eval_judge.PROMPT_VERSION, response={'scores': 'x'})))
    transport = Transport()
    provider = LLMProvider('anthropic', 'sk-test', 'claude-sonnet-4-5', DEFAULT_RATES['claude-sonnet-4-5'],
                           transport=transport)
    result = run_judge(**_args(clean, tmp_path, provider))
    assert result['status'] == 'scored' and not result['cache_hit'] and len(transport.bodies) == 1


class ProviderBudgetError(ProviderError):
    pass


class Blocked:
    provider, model = 'fake', 'blocked'

    def __init__(self):
        self.calls = 0

    def complete_json(self, **kwargs):
        self.calls += 1
        raise ProviderBudgetError('Episode LLM budget would exceed $5')


def test_budget_error_and_missing_provider(clean, tmp_path):
    assert run_judge(**_args(clean, tmp_path, None))['status'] == 'unavailable'
    assert not (clean / GUARD_NAME).exists()
    blocked = Blocked()
    result = run_judge(**_args(clean, tmp_path, blocked))
    assert result['status'] == 'unavailable' and 'budget' in result['reason'] and result['scores'] is None
    assert run_judge(**_args(clean, tmp_path, blocked))['status'] == 'unavailable' and blocked.calls == 1


def test_ledgerless_fake_provider_still_limited_by_guard(clean, tmp_path):
    class Fake:
        provider, model, calls = 'fake', 'invalid', 0

        def complete_json(self, **kwargs):
            Fake.calls += 1
            assert kwargs['role'] == 'judge' and kwargs['max_output_tokens'] == 1024
            return type('R', (), dict(data={'not': 'rubric'}, usage={}))()
    assert run_judge(**_args(clean, tmp_path, Fake()))['status'] == 'failed'
    assert run_judge(**_args(clean, tmp_path, Fake()))['status'] == 'unavailable' and Fake.calls == 1
