"""Terminal scoreboard on real pipeline artifacts, with no network/model APIs."""
import copy
import json

import pytest

from pipelines.ffmpeg import eval_cli, eval_tools as tools
from pipelines.ffmpeg.contracts import read_json
from pipelines.ffmpeg.demo_scoreboard import render_table
from pipelines.ffmpeg.evaluator import evaluate_episode, validate_eval_report
from tests.test_eval_tools import write_jsonl, run
from tests.test_evaluator import FakeJudge, world  # Real FFmpeg fixture, not a made-up EvalReport.


@pytest.fixture(scope='module')
def real_report(world):
    return evaluate_episode(world / 'episodes' / 'base', provider=None, persist=False)


@pytest.fixture
def report(real_report):
    def make(eid='base', *, Q=.5, usd=1.0, created='2026-01-01T00:00:00+00:00', **kwargs):
        r = copy.deepcopy(real_report)
        r.update(episode_id=eid, created_at=created, **kwargs)
        r['scores']['Q'] = Q
        r['model_cost'].update(usd=usd, known_usd=usd if usd is not None else 0.0,
                               status='unknown' if usd is None else 'measured',
                               quality_per_dollar=Q / usd if Q is not None and usd else None)
        validate_eval_report(r)
        return r
    return make


def qa(root, eid, passed=True):
    directory = root / eid
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'qa-report.json').write_text(json.dumps({'episode_id': eid, 'passed': passed}))


def cells(out, eid):
    return [c.strip() for c in next(s for s in out.splitlines() if s.startswith(eid + ' ')).split('|')]


def test_latest_history_cost_not_sum_and_ties_use_file_order(report, tmp_path):
    records = [report('a', usd=100), report('a', usd=2, created='2026-01-02T00:00:00+00:00'),
               report('a', usd=3, created='2026-01-02T00:00:00+00:00'),
               report('a', usd=900, created='2025-01-01T00:00:00+00:00'), report('b', usd=4)]
    qa(tmp_path, 'a')
    qa(tmp_path, 'b', False)
    out = render_table(records, episodes_root=tmp_path)
    assert cells(out, 'a')[1:4] == ['3.000000', 'measured', '3.000000']
    assert cells(out, 'a')[7] == '0.167'
    assert 'Latest episodes: 2; valid evaluation lines: 5' in out
    assert 'Cost USD total: 7.000000;' in out
    assert 'Actual QA pass rate: 50.0% (1/2); known: 2; unavailable: 0' in out


def test_missing_scores_unknown_partial_cost_zero_cost_and_wall(report, tmp_path):
    unknown = report('unknown', Q=None, usd=None)
    unknown['scores']['D'] = None
    unknown['timing']['elapsed_seconds'] = None
    unknown['model_cost']['known_usd'] = .04
    zero = report('zero', usd=0)
    estimated = report('estimated', usd=.02)
    estimated['model_cost']['status'] = 'estimated'
    out = render_table([unknown, zero, estimated], episodes_root=tmp_path)
    c = cells(out, 'unknown')
    assert c[1:9] == ['N/A', 'unknown/incomplete', '0.040000', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    assert cells(out, 'zero')[7] == 'N/A'
    assert cells(out, 'estimated')[2] == 'estimated'
    assert cells(out, 'estimated')[7] == '25.000'
    assert 'Cost USD total: N/A; known_usd* subtotal: 0.060000; cost known: 2; cost missing: 1' in out
    assert 'Actual QA pass rate: N/A (0/0); known: 0; unavailable: 3' in out
    assert 'Missing scores: J=3, Q=1, D=1; wall time missing: 1' in out
    assert 'not a complete bill' in out


def test_real_qa_artifact_not_gates_or_first_try(world, report, tmp_path):
    r = report()
    r['scores'].update(G=0, Q=0)
    r['gates']['passed'] = False
    r['first_try_passed'] = False
    directory = tmp_path / 'base'
    directory.mkdir()
    artifact = read_json(world / 'episodes' / 'base' / 'qa-report.json')
    assert artifact['passed'] is True
    (directory / 'qa-report.json').write_text(json.dumps(artifact))
    out = render_table([r, report('missing')], episodes_root=tmp_path)
    assert cells(out, 'base')[-2:] == ['0/FAIL', 'PASS']
    assert 'Actual QA pass rate: 100.0% (1/1); known: 1; unavailable: 1' in out


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'empty', 'wrong-identity', 'nonboolean', 'unreadable'])
def test_unavailable_qa_not_counted(report, tmp_path, kind):
    qa(tmp_path, 'base')
    path = tmp_path / 'base' / 'qa-report.json'
    if kind == 'missing':
        path.unlink()
    elif kind == 'unreadable':
        path.unlink()
        path.mkdir()
    else:
        path.write_text({'corrupt': '{oops', 'empty': '', 'wrong-identity': '{"episode_id":"other","passed":true}',
                         'nonboolean': '{"episode_id":"base","passed":1}'}[kind])
    out = render_table([report()], episodes_root=tmp_path)
    assert cells(out, 'base')[-1] == 'N/A'
    assert 'Actual QA pass rate: N/A (0/0); known: 0; unavailable: 1' in out


@pytest.mark.parametrize('eid', ['../outside', '/tmp/outside', 'UPPER', 'bad\x1b[31m'])
def test_unsafe_id_never_read_or_echoed(report, tmp_path, monkeypatch, eid):
    def forbidden(*args):
        raise AssertionError('unsafe id must not cause a read')
    monkeypatch.setattr('pipelines.ffmpeg.demo_scoreboard.read_json', forbidden)
    out = render_table([report(eid)], episodes_root=tmp_path)
    assert cells(out, '[unsafe-id]')[-1] == 'N/A'
    assert eid not in out


def test_symlink_episode_and_qa_cannot_escape(report, tmp_path):
    outside = tmp_path / 'outside'
    qa(outside, 'base')
    root = tmp_path / 'episodes'
    root.mkdir()
    (root / 'base').symlink_to(outside / 'base', target_is_directory=True)
    out = render_table([report()], episodes_root=root)
    assert cells(out, '[unsafe-id]')[-1] == 'N/A'
    (root / 'base').unlink()
    (root / 'base').mkdir()
    (root / 'base' / 'qa-report.json').symlink_to(outside / 'base' / 'qa-report.json')
    assert cells(render_table([report()], episodes_root=root), 'base')[-1] == 'N/A'


def test_strict_reports_invalid_corrupt_lines_visible(report, tmp_path, capsys):
    evals = tmp_path / 'evals'
    bad = report()
    bad['scores']['Q'] = 'secret-source-filename.mp4'
    write_jsonl(evals / 'scoreboard.jsonl', [report(), bad, {'episode_id': 'base'}, 42])
    with (evals / 'scoreboard.jsonl').open('a') as f:
        f.write('{truncated\n{"value":NaN}\n')
    code, out, err = run(['scoreboard', '--table', '--evals-root', str(evals)], capsys)
    assert code == 0 and not err
    assert 'Latest episodes: 1; valid evaluation lines: 1; invalid/corrupt lines: 5' in out
    assert '(invalid reports: 2, unreadable lines: 3)' in out
    assert 'secret-source-filename' not in out


def test_empty_missing_and_unreadable_scoreboard(tmp_path, capsys, monkeypatch):
    args = ['scoreboard', '--table', '--evals-root', str(tmp_path)]
    for exists in (False, True):
        if exists:
            (tmp_path / 'scoreboard.jsonl').write_text('')
        code, out, err = run(args, capsys)
        assert code == 0 and not err
        assert '(no readable evaluation reports)' in out
        assert 'Actual QA pass rate: N/A (0/0)' in out
        assert 'Cost USD total: N/A;' in out
    def unreadable(*a):
        raise PermissionError('SECRET/private/path')
    monkeypatch.setattr(tools, 'load_reports', unreadable)
    code, out, err = run(args, capsys)
    assert code == 3 and not err
    assert 'Scoreboard input: unavailable' in out and 'SECRET' not in out


def test_json_default_intact_and_no_provider_calls(report, tmp_path, capsys, monkeypatch):
    write_jsonl(tmp_path / 'scoreboard.jsonl', [report()])
    def forbidden(*a, **k):
        raise AssertionError('scoreboard must not initialize a provider')
    monkeypatch.setattr('pipelines.ffmpeg.llm_provider.LLMProvider.from_env', forbidden)
    args = ['scoreboard', '--evals-root', str(tmp_path)]
    for flag in ([], ['--json']):
        code, out, err = run(args + flag, capsys)
        assert code == 0 and not err and json.loads(out)['episodes'] == 1
    code, out, err = run(args + ['--table'], capsys)
    assert code == 0 and not err and 'quality_per_dollar' in out
    for flag in ('--json', '--lines'):
        code, _, err = run(args + ['--table', flag], capsys)
        assert code == 2 and 'cannot be combined' in err


def test_wall_time_is_pipeline_not_evaluation_time(report, tmp_path):
    r = report()
    r['timing'].update(elapsed_seconds=44.93, evaluation_seconds=1234)
    out = render_table([r], episodes_root=tmp_path)
    assert cells(out, 'base')[8] == '44.93s'
    assert '1234' not in out


def test_table_baseline_semantics_and_invalid_reports(report, tmp_path, capsys):
    write_jsonl(tmp_path / 'scoreboard.jsonl', [report(Q=.5), {'garbage': True}])
    baseline = tmp_path / 'baseline.jsonl'
    args = ['scoreboard', '--table', '--evals-root', str(tmp_path), '--baseline', str(baseline)]
    code, out, err = run(args, capsys)
    assert code == 3 and not err and 'Baseline comparison: unavailable' in out
    write_jsonl(baseline, [report(Q=.8), {'garbage': True}])
    code, out, err = run(args, capsys)
    assert code == 1 and not err and 'Baseline comparison: regression' in out
    write_jsonl(baseline, [report(Q=.5)])
    code, out, err = run(args, capsys)
    assert code == 0 and not err and 'Baseline comparison: ok' in out


def test_directory_scoreboard_is_unavailable(tmp_path, capsys):
    (tmp_path / 'scoreboard.jsonl').mkdir()
    code, out, err = run(['scoreboard', '--table', '--evals-root', str(tmp_path)], capsys)
    assert code == 3 and not err and 'Scoreboard input: unavailable' in out


def test_older_valid_report_survives_invalid_newer_history(report, tmp_path):
    good = report(usd=2)
    bad = report(usd=99, created='2027-01-01T00:00:00+00:00')
    bad['scores']['Q'] = 7
    out = render_table([good, bad], episodes_root=tmp_path)
    assert cells(out, 'base')[1] == '2.000000'
    assert 'invalid/corrupt lines: 1' in out


def test_numeric_scores_from_real_mock_judge_episode(world):
    judge = FakeJudge()
    r = evaluate_episode(world / 'episodes' / 'base', provider=judge, persist=False)
    out = render_table([r], episodes_root=world / 'episodes')
    assert cells(out, 'base')[4:7] == [f"{r['scores'][k]:.3f}" for k in ('J', 'Q', 'D')]
    assert cells(out, 'base')[-2:] == ['1/PASS', 'PASS']


def test_total_usd_uses_usd_not_known_estimate(report, tmp_path):
    r = report(usd=3)
    r['model_cost']['known_usd'] = 2
    out = render_table([r], episodes_root=tmp_path)
    assert 'Cost USD total: 3.000000; known_usd* subtotal: 2.000000' in out
