"""EpisodeEvaluator end-to-end on real FFmpeg renders with mocked judge only."""
import json
import shutil
import subprocess

import pytest

from pipelines.ffmpeg import goal_judge
from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.eval_gates import generate_distinct_footage, seed_defect
from pipelines.ffmpeg.eval_judge import CRITERIA
from pipelines.ffmpeg.evaluator import (append_scoreboard, composite, evaluate_episode, model_cost,
                                        validate_eval_report)
from pipelines.ffmpeg.llm_provider import LLMResult

GOAL = 'Create a 10 second family-trip teaser'
TOP = {'schema_version', 'evaluation_id', 'episode_id', 'goal', 'created_at', 'render_hash', 'footage_hash', 'gates',
       'scores', 'judge', 'model_cost', 'tokens', 'timing', 'provenance', 'first_try_passed', 'limitations'}


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('evaluator')
    fixture = generate_distinct_footage(root / 'footage')
    report = run_episode(root / 'footage', GOAL, episodes_root=root / 'episodes', episode_id='base',
                         transcript_fixture=fixture, provider=None)
    assert report['passed']
    return root


@pytest.fixture
def episode(world, tmp_path):
    target = tmp_path / 'ep'
    shutil.copytree(world / 'episodes' / 'base', target, ignore=shutil.ignore_patterns('work', 'proxies', 'frames'))
    return target


class FakeJudge:
    provider, model = 'fake', 'fake-judge-1'

    def __init__(self, goal_fit=4):
        self.calls, self.goal_fit = 0, goal_fit

    def complete_json(self, **kwargs):
        self.calls += 1
        ids = [s['shot_id'] for s in json.loads(kwargs['prompt'])['edit']['shots']]
        scores = {c: 4 for c in CRITERIA}
        scores['goal_fit'] = self.goal_fit
        return LLMResult(data=dict(scores=scores, evidence={c: ids[:1] for c in CRITERIA}, weakest_shot=ids[-1],
                                   fix_instruction='Tighten the ending', verdict='pass', unsupported_goal=False,
                                   limitations=[]), usage={})


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_passing_episode_full_report_and_scoreboard(episode, tmp_path, monkeypatch):
    monkeypatch.setattr(goal_judge, 'judge_goal', lambda *a, **k: (_ for _ in ()).throw(AssertionError('legacy judge')))
    judge = FakeJudge()
    report = evaluate_episode(episode, provider=judge, evals_root=tmp_path / 'evals')
    assert set(report) == TOP and judge.calls == 1
    s = report['scores']
    assert s['G'] == 1 and s['J'] == pytest.approx(0.75) and 0 < s['D'] <= 1 and s['H'] is None
    assert s['weights'] == dict(J=0.625, D=0.375, H=0.0)
    assert s['Q'] == pytest.approx(0.625 * s['J'] + 0.375 * s['D'], abs=1e-5) and s['Q_status'] == 'computed'
    assert set(s['D_components']) == {'sharpness', 'stability', 'exposure', 'pacing', 'coverage', 'variety', 'hook'}
    assert report['judge']['verdict'] == 'pass' and report['judge']['scores']['goal_fit'] == 4
    assert report['model_cost'] == dict(report['model_cost'], usd=0.0, status='measured', quality_per_dollar=None)
    assert report['provenance']['fixture_mode'] and report['provenance']['transcription_status'] == 'fixture'
    assert report['first_try_passed'] is True and report['render_hash'] and report['footage_hash']
    assert json.loads((episode / 'eval-report.json').read_text()) == report
    assert _lines(tmp_path / 'evals' / 'scoreboard.jsonl') == [report]
    # Persistence retry: same evaluation_id is never duplicated.
    assert append_scoreboard(report, tmp_path / 'evals') is False
    again = evaluate_episode(episode, provider=judge, evals_root=tmp_path / 'evals')
    assert again['evaluation_id'] != report['evaluation_id'] and again['judge']['cache_hit'] and judge.calls == 1
    assert len(_lines(tmp_path / 'evals' / 'scoreboard.jsonl')) == 2


def test_changed_render_after_judgment_has_no_stale_score(episode, tmp_path):
    judge = FakeJudge()
    assert evaluate_episode(episode, provider=judge, evals_root=tmp_path / 'evals')['scores']['J'] is not None
    video = episode / 'finished.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-c', 'copy', '-metadata', 'comment=rerender',
                    str(episode / 'x.mp4')], check=True)
    (episode / 'x.mp4').replace(video)
    report = evaluate_episode(episode, provider=judge, evals_root=tmp_path / 'evals')
    assert report['gates']['passed'] and judge.calls == 1
    assert report['scores']['J'] is None and report['scores']['Q'] is None and report['judge']['status'] == 'unavailable'


def test_gate_failure_forces_q_zero_and_no_judge_call(episode, tmp_path):
    seed_defect(episode, 'loud')
    judge = FakeJudge()
    report = evaluate_episode(episode, provider=judge, evals_root=tmp_path / 'evals')
    assert report['scores']['G'] == 0 and report['scores']['Q'] == 0 and judge.calls == 0
    assert report['judge']['status'] == 'unavailable' and report['scores']['J'] is None
    assert report['scores']['Q_status'] == 'forced_zero_gate_failed'


def test_prerender_failure_yields_valid_zero_report(world, tmp_path):
    failed = run_episode(world / 'footage', GOAL, episodes_root=tmp_path / 'eps', episode_id='missing', provider=None)
    assert not failed['passed'] and failed['error']['code'] == 'DEPENDENCY_MISSING'
    report = evaluate_episode(tmp_path / 'eps' / 'missing', provider=FakeJudge(), evals_root=tmp_path / 'evals')
    s = report['scores']
    assert s['G'] == 0 and s['Q'] == 0 and s['D'] is None and s['J'] is None and s['H'] is None
    assert report['render_hash'] is None and report['judge']['status'] == 'unavailable'
    assert report['goal'] == GOAL and report['first_try_passed'] is False
    assert _lines(tmp_path / 'evals' / 'scoreboard.jsonl')[0]['evaluation_id'] == report['evaluation_id']


def test_visual_only_episode_is_g0_without_fabricated_word_pass(world, tmp_path):
    report = run_episode(world / 'footage', GOAL, episodes_root=tmp_path / 'eps', episode_id='visual',
                         provider=None, visual_only=True)
    evaluation = evaluate_episode(tmp_path / 'eps' / 'visual', evals_root=tmp_path / 'evals', persist=False)
    word = next(c for c in evaluation['gates']['checks'] if c['name'] == 'word_cut_integrity')
    assert word['status'] == 'not_run' and evaluation['scores']['G'] == 0 and evaluation['scores']['Q'] == 0
    assert evaluation['provenance']['transcription_status'] == 'visual_only_unavailable'
    assert not (tmp_path / 'evals').exists() and report['episode_id'] == 'visual'


def test_labels_drive_h_and_weights(episode, tmp_path):
    shots = json.loads((episode / 'narrative-spec.json').read_text())['sources']
    name = shots[0]['path'].rsplit('/', 1)[-1]
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps(dict(labels=[dict(filename=name, start_ms=100, end_ms=900),
                                              dict(filename='not-in-footage.mp4', start_ms=0, end_ms=500)])))
    report = evaluate_episode(episode, provider=FakeJudge(), evals_root=tmp_path / 'evals', labels=labels)
    s = report['scores']
    assert s['H'] == 1.0 and s['weights'] == dict(J=0.5, D=0.3, H=0.2)
    assert s['Q'] == pytest.approx(0.5 * s['J'] + 0.3 * s['D'] + 0.2, abs=1e-5)
    assert s['H_detail']['labelled'] == 1 and len(s['H_detail']['ignored']) == 1


def test_composite_rules():
    assert composite(0, 1, 1, None, False)[0:2] == (0.0, 'forced_zero_gate_failed')
    assert composite(1, None, 0.8, None, False)[0] is None
    assert composite(1, 0.8, None, None, False)[0] is None
    assert composite(1, 0.8, 0.8, None, True)[0] is None  # supplied labels never silently renormalized
    assert composite(1, 0.8, 0.4, None, False)[0] == pytest.approx(0.65)
    assert composite(1, 0.8, 0.4, 0.5, True)[0] == pytest.approx(0.62)


def test_cost_accounting_known_estimated_unknown(tmp_path):
    ledger = lambda calls: (tmp_path / 'llm-usage.json').write_text(json.dumps(dict(version=1, budget_usd=5, calls=calls)))
    call = lambda **kw: dict(dict(call_id='c', role='planner', provider='p', model='m', status='success', input_tokens=10,
                                  output_tokens=5, cost_usd=0.02, cost_status='estimated', reserved_usd=0.4, error=None), **kw)
    cost, tokens = model_cost(tmp_path, None)
    assert cost['usd'] == 0 and cost['status'] == 'measured' and tokens == dict(input=0, output=0, complete=True)
    ledger([call(), call(status='failed', cost_usd=None, input_tokens=None, output_tokens=None), call(status='blocked',
            cost_usd=None, reserved_usd=0, input_tokens=None, output_tokens=None)])
    cost, tokens = model_cost(tmp_path, 0.5)
    assert cost['usd'] == pytest.approx(0.42) and cost['status'] == 'estimated'
    assert cost['quality_per_dollar'] == pytest.approx(0.5 / 0.42, rel=1e-4)
    assert tokens == dict(input=10, output=5, complete=False)
    (tmp_path / 'tts-usage.json').write_text(json.dumps(dict(calls=[dict(status='success', cost_usd=0.08, cost_status='measured')])))
    assert model_cost(tmp_path, 0.5)[0]['tts_usd'] == pytest.approx(0.08)
    ledger([call(cost_usd=None, cost_status='unknown')])
    cost, _ = model_cost(tmp_path, 0.5)
    assert cost['usd'] is None and cost['llm_usd'] is None and cost['status'] == 'unknown' and cost['quality_per_dollar'] is None
    assert cost['known_usd'] == pytest.approx(0.08)


def test_scoreboard_recovers_partial_line_and_schema_rejects_forgery(episode, tmp_path):
    report = evaluate_episode(episode, evals_root=tmp_path / 'evals', persist=False)
    board = tmp_path / 'evals' / 'scoreboard.jsonl'
    board.parent.mkdir(parents=True)
    board.write_text('{"evaluation_id": "trunc')
    assert append_scoreboard(report, tmp_path / 'evals') is True
    lines = board.read_text().split('\n')
    assert lines[0] == '{"evaluation_id": "trunc' and json.loads(lines[1])['evaluation_id'] == report['evaluation_id']
    for mutate in [lambda r: r.update(extra=1), lambda r: r['scores'].update(G=0, Q=0.5),
                   lambda r: r['scores'].update(Q=float('nan')), lambda r: r['judge'].update(status='scored')]:
        forged = json.loads(json.dumps(report))
        mutate(forged)
        with pytest.raises(Exception):
            validate_eval_report(forged)
