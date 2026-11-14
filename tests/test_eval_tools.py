"""Eval tools: unit tests on synthetic scoreboard records plus real-FFmpeg episodes with a mocked judge.

No network, no API keys, no real model: the only judges are in-process fakes.
"""
import json
import shutil
from pathlib import Path

import pytest

from pipelines.ffmpeg import cli, eval_cli, eval_tools as tools
from pipelines.ffmpeg.eval_gates import generate_distinct_footage
from pipelines.ffmpeg.eval_judge import CRITERIA
from pipelines.ffmpeg.llm_provider import LLMProvider, LLMResult

REPO = Path(__file__).resolve().parents[1]
GOAL = 'Create a 10 second family-trip teaser'


# ----------------------------------------------------------------- helpers
def rep(eid, rh='r1', *, Q=0.5, passed=True, usd=0.0, created='2026-01-01T00:00:00+00:00', pr=None, first=True,
        elapsed=10.0, footage=100.0, scores=None, verdict=None, method='scores-1.0', fh='f1', goal='g', J=None,
        tokens_complete=True):
    judge = dict(status='scored' if scores else 'unavailable', scores=scores, verdict=verdict, prompt_version='p1',
                 provider='fake', model='m', unsupported_goal=False if scores else None)
    return dict(schema_version='1.0', evaluation_id=f'{eid}-{rh}-{created}', episode_id=eid, goal=goal,
                created_at=created, render_hash=rh, footage_hash=fh,
                gates=dict(passed=passed, version='g1', thresholds={}),
                scores=dict(G=int(passed), D=0.5, J=J, H=None, Q=Q, weights=dict(J=.625, D=.375, H=0.0),
                            method_version=method),
                judge=judge, model_cost=dict(usd=usd, status='unknown' if usd is None else 'measured'),
                tokens=dict(input=0, output=0, complete=tokens_complete),
                timing=dict(elapsed_seconds=elapsed, footage_seconds=footage),
                provenance=dict(pr=pr, fixture_mode=False, evaluator_version='e1'),
                first_try_passed=first, limitations=[])


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(i) + '\n' for i in items))


def rating(eid, rh='r1', *, score=4, publish=True, rater='alice', created='2026-01-02T00:00:00+00:00'):
    return dict(episode_id=eid, render_hash=rh, score=score, publish_without_edits=publish, rater=rater,
                created_at=created)


def run(args, capsys, **kw):
    code = eval_cli.main(args, **kw)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------- kappa
def test_kappa_values_and_null_reasons():
    assert tools.cohen_kappa([(True, True), (False, False), (True, True), (False, False)]) == (1.0, None)
    k, why = tools.cohen_kappa([(True, True), (True, True)])
    assert k is None and 'degenerate' in why
    k, why = tools.cohen_kappa([(True, False), (False, False), (True, False)])  # one constant rater: kappa remains defined
    assert k == 0 and why is None
    assert tools.cohen_kappa([(True, True)])[0] is None
    assert tools.cohen_kappa([])[1].startswith('need at least 2')
    k, _ = tools.cohen_kappa([(True, True), (True, False), (False, False), (False, True)])
    assert k == 0.0


# ------------------------------------------------------------- calibration
def scored(eid, rh, mean_score, verdict, **kw):
    scores = {c: mean_score for c in CRITERIA}
    return rep(eid, rh, scores=scores, verdict=verdict, J=0.5, **kw)


def test_calibration_latest_human_within_one_and_kappa():
    reports = [scored('a', 'r1', 4, 'pass'), scored('b', 'r1', 2, 'fail'), scored('c', 'r1', 5, 'pass'),
               scored('d', 'r1', 3, 'pass'), rep('e', 'r1')]
    ratings = [rating('a', score=1, publish=False, created='2026-01-01T00:00:00+00:00'),
               rating('a', score=5, publish=True),                 # latest wins: |4-5| <= 1
               rating('b', score=1, publish=False),                # |2-1| <= 1
               rating('c', score=2, publish=False),                # |5-2| = 3 -> disagree
               rating('d', score=3, publish=True),
               rating('e', score=3, publish=True),                 # judge unscored -> unmatched
               rating('f', score=3, publish=True),                 # no evaluation -> unmatched
               rating('a', 'old-render', score=1, publish=False)]  # other render: unmatched
    out = tools.calibrate(reports, ratings, rater='alice')
    assert out['samples'] == 4 and out['agreement_within_1_pct'] == 75.0 and out['agreement_target_met'] is False
    assert out['kappa'] is not None and out['kappa_target_met'] is (out['kappa'] >= 0.6)
    reasons = {u['episode_id']: u['reason'] for u in out['unmatched']}
    assert set(reasons) == {'e', 'f', 'a'} and 'did not score' in reasons['e']
    assert out['warning']


def test_calibration_requires_defined_rater_and_reports_null_reasons():
    reports = [scored('a', 'r1', 4, 'pass')]
    out = tools.calibrate(reports, [rating('a', rater='x'), rating('a', rater='y')])
    assert out['agreement_within_1_pct'] is None and 'rater' in out['agreement_reason'] and out['kappa'] is None
    out = tools.calibrate(reports, [])
    assert out['agreement_reason'] == 'no human ratings recorded'
    out = tools.calibrate(reports, [rating('a')])  # single rater inferred; n=1 -> kappa null with reason
    assert out['rater'] == 'alice' and out['agreement_within_1_pct'] == 100.0
    assert out['kappa'] is None and 'at least 2' in out['kappa_reason']
    out = tools.calibrate([scored('a', 'r1', 4, 'pass'), scored('b', 'r1', 4, 'pass')], [rating('a'), rating('b')])
    assert out['kappa'] is None and 'degenerate' in out['kappa_reason']


# ----------------------------------------------------------------- summary
def test_summary_dedups_includes_failures_and_never_fabricates():
    reports = [
        rep('a', 'r1', Q=0.2, created='2026-01-01T00:00:00+00:00', usd=1.0, pr=7),
        rep('a', 'r1', Q=0.6, created='2026-01-02T00:00:00+00:00', usd=1.0, pr=7),   # newer duplicate, same render
        rep('a', 'r2', Q=0.8, created='2026-01-03T00:00:00+00:00', usd=2.0, pr=7),   # new render; cumulative cost
        rep('b', 'r1', Q=0.0, passed=False, usd=0.0, first=False, pr=7),             # gate failure stays in
        rep('c', 'r1', Q=None, usd=None, first=None, elapsed=None, footage=None, tokens_complete=False),
        {'garbage': True}]
    s = tools.summarize(reports, unreadable_lines=2)
    assert s['unreadable_lines'] == 2 and s['unique_renders'] == 4 and s['episodes'] == 3
    assert s['quality']['mean_Q'] == pytest.approx((0.6 + 0.8 + 0.0) / 3, abs=1e-5)
    assert s['quality']['scored_renders'] == 3 and s['quality']['unscored_renders'] == 1
    assert s['gate_pass_rate'] == 0.75
    assert s['cost']['total_usd'] is None and s['cost']['known_usd'] == 2.0 and s['cost']['episodes_cost_unknown'] == 1
    assert s['cost']['episodes_zero_known_spend'] == 1       # b: a known zero is not unknown
    assert s['cost']['quality_per_dollar'] == pytest.approx((0.6 + 0.8) / (1.0 + 2.0), abs=1e-5)
    assert s['cost']['mean_cost_per_clip_usd'] == 1.0         # mean over the 2 known episodes (a=2.0, b=0.0)
    assert s['footage_to_clip_elapsed'] == dict(mean_ratio=0.1, median_ratio=0.1, samples=3, missing=1)
    assert s['first_try'] == dict(pass_rate=0.5, final_pass_rate=0.5, first_final_agreement=1.0, samples=2, unknown=1)
    assert s['per_pr']['7']['episodes'] == 2 and s['per_pr']['unattributed']['cost_unknown_episodes'] == 1
    m = s['missing']
    assert m['Q_null'] == 1 and m['cost_unknown'] == 1 and m['tokens_incomplete'] == 1 and m['elapsed_missing'] == 1
    assert m['first_try_unknown'] == 1 and m['pr_missing'] == 1


def test_summary_empty_has_nulls_not_zeros():
    s = tools.summarize([])
    assert s['quality']['mean_Q'] is None and s['gate_pass_rate'] is None
    assert s['cost']['quality_per_dollar'] is None and s['cost']['quality_per_dollar_reason']
    assert s['cost']['episodes_cost_unknown'] == 0 and s['unique_renders'] == 0


def test_scoreboard_cli_text_lines_and_corrupt_lines(tmp_path, capsys):
    evals = tmp_path / 'evals'
    good = rep('a')
    evals.mkdir()
    (evals / 'scoreboard.jsonl').write_text(json.dumps(good) + '\n{truncated\n')
    code, out, _ = run(['scoreboard', '--evals-root', str(evals)], capsys)
    assert code == 0 and json.loads(out)['unreadable_lines'] == 1
    code, out, _ = run(['scoreboard', '--evals-root', str(evals), '--lines'], capsys)
    assert json.loads(out.strip()) == good
    code, out, _ = run(['scoreboard', '--evals-root', str(tmp_path / 'none')], capsys)
    assert code == 0 and json.loads(out)['unique_renders'] == 0


# ---------------------------------------------------------------- baseline
def test_baseline_comparison_rules(tmp_path, capsys):
    base = [rep('a', Q=0.8), rep('b', 'r1', Q=0.6)]
    ok = tools.compare_baseline([rep('a', Q=0.79), rep('b', Q=0.59)], base)
    assert ok['status'] == 'ok' and ok['compared'] == 2 and tools.compare_exit_code(ok) == 0
    reg = tools.compare_baseline([rep('a', Q=0.7), rep('b', Q=0.5)], base)         # -14%
    assert reg['status'] == 'regression' and reg['relative_change'] < -0.05 and tools.compare_exit_code(reg) == 1
    edge = tools.compare_baseline([rep('a', Q=0.76), rep('b', Q=0.57)], base)       # exactly -5% is not a failure
    assert edge['status'] == 'ok'
    for cur, base_, needle in [
            ([rep('a', Q=0.8)], [], 'no readable'),
            ([], base, 'current scoreboard'),
            ([rep('z', Q=0.8)], base, 'case coverage differs'),
            ([rep('a', Q=0.8, method='scores-2.0')], base[:1], 'config mismatch'),
            ([rep('a', Q=0.8, fh='other')], base[:1], 'footage/goal differ'),
            ([rep('a', Q=0.8, goal='other')], base[:1], 'footage/goal differ'),
            ([rep('a', Q=None)], base[:1], 'unscored'),
            ([rep('a', Q=0.8)], [rep('a', Q=0.0)], 'zero')]:
        res = tools.compare_baseline(cur, base_)
        assert res['status'] == 'unavailable' and needle in res['reason'], (needle, res)
        assert tools.compare_exit_code(res) == 3
    # CLI wiring: regression => 1, missing baseline file => 3 (never guessed)
    evals = tmp_path / 'evals'
    write_jsonl(evals / 'scoreboard.jsonl', [rep('a', Q=0.7)])
    write_jsonl(tmp_path / 'main.jsonl', [rep('a', Q=0.8)])
    code, out, _ = run(['scoreboard', '--evals-root', str(evals), '--baseline', str(tmp_path / 'main.jsonl')], capsys)
    assert code == 1 and json.loads(out)['baseline_comparison']['status'] == 'regression'
    code, out, _ = run(['scoreboard', '--evals-root', str(evals), '--baseline', str(tmp_path / 'absent.jsonl')], capsys)
    assert code == 3 and 'not found' in json.loads(out)['baseline_comparison']['reason']
    code, _, _ = run(['scoreboard', '--evals-root', str(evals), '--baseline', str(tmp_path / 'main.jsonl'),
                      '--tolerance', '0.2'], capsys)
    assert code == 0


# ------------------------------------------------------------------ golden
def test_manifest_matches_contract_and_has_no_media():
    manifest = tools.load_manifest()
    asset = manifest['external_assets'][0]
    assert (asset['release_tag'], asset['file'], asset['expected_clips']) == (
        'sample-data-arizona-480p', 'ff-arizona-480p.zip', 26)
    cases = {c['id']: c for c in manifest['cases']}
    assert {c['kind'] for c in cases.values()} == {'synthetic', 'external'}
    goals = ' '.join(c['goal'].lower() for c in cases.values())
    for word in ('30 second', 'vertical', '60 second', 'horizontal', 'kids', 'landscape', 'beach', 'sunset', 'desert'):
        assert word in goals
    assert cases['arizona-impossible-beach-sunset-desert']['expected']['qa_result_in'] == ['partial']
    for synthetic in (c for c in cases.values() if c['kind'] == 'synthetic'):   # footage covers the target
        target = int(synthetic['goal'].split(' second')[0].split()[-1])
        assert 3 * synthetic['synthetic']['clip_seconds'] >= target
    media = ('.mp4', '.mov', '.wav', '.mkv', '.zip')
    assert not [p for p in (REPO / 'evals' / 'golden').rglob('*') if p.suffix.lower() in media]


class Recorder:
    def __init__(self, result='partial', gates=True, unsupported=None):
        self.runs, self.evals, self.result, self.gates, self.unsupported = [], [], result, gates, unsupported

    def runner(self, footage, goal, **kw):
        self.runs.append((Path(footage), goal, kw))
        (kw['episodes_root'] / kw['episode_id']).mkdir(parents=True, exist_ok=True)
        return dict(result=self.result, passed=True)

    def evaluator(self, directory, **kw):
        self.evals.append((directory, kw))
        if self.unsupported is not None:
            judge = dict(status='scored', unsupported_goal=self.unsupported, verdict='fail')
        else:
            judge = dict(status='unavailable', unsupported_goal=None, verdict=None)
        return dict(evaluation_id='x', gates=dict(passed=self.gates), scores=dict(Q=None, H=None), judge=judge)


def mini_manifest(**case):
    base = dict(id='ext', kind='external', asset='arz', goal='Make a teaser', expected={'gates_passed': True})
    base.update(case)
    return dict(schema_version='1.0', external_assets=[dict(id='arz', repo='o/r', release_tag='t', file='f.zip',
                                                            archive_subdir='clips', expected_clips=2)], cases=[base])


def clips(root, n=2):
    folder = root / 'footage' / 'clips'
    folder.mkdir(parents=True)
    for i in range(n):
        (folder / f'{i}.mp4').write_bytes(b'')
    return folder


def golden(tmp_path, manifest, rec, **kw):
    kw.setdefault('asr_available', True)
    return tools.run_golden(manifest, work_dir=tmp_path / 'w', evals_root=tmp_path / 'e', runner=rec.runner,
                            evaluator=rec.evaluator, run_id='t1', **kw)


def test_golden_external_missing_is_unavailable_nonzero_and_never_downloads(tmp_path):
    rec = Recorder()
    called = []
    res = golden(tmp_path, mini_manifest(), rec, fetcher=lambda *a: called.append(a))
    assert res['status'] == 'unavailable' and res['exit_code'] == 3 and not called and not rec.runs
    assert 'opt in' in res['cases'][0]['reason']


def test_golden_download_is_opt_in_and_failures_are_unavailable(tmp_path):
    rec = Recorder()
    folder = clips(tmp_path / 'src')
    res = golden(tmp_path, mini_manifest(), rec, download=True, golden_data=tmp_path / 'data',
                 fetcher=lambda a, d: folder)
    assert res['status'] == 'passed' and res['exit_code'] == 0 and len(rec.runs) == 1

    def boom(a, d):
        raise tools.EvalToolError('no access')
    res = golden(tmp_path, mini_manifest(), Recorder(), download=True, golden_data=tmp_path / 'data2', fetcher=boom)
    assert res['exit_code'] == 3 and 'download failed' in res['cases'][0]['reason']
    res = golden(tmp_path, mini_manifest(), Recorder(), download=True, fetcher=boom)
    assert res['exit_code'] == 3 and '--golden-data' in res['cases'][0]['reason']


def test_golden_asr_gate_visual_only_and_clip_count(tmp_path):
    data = tmp_path / 'data'
    clips(data)
    rec = Recorder()
    res = golden(tmp_path, mini_manifest(), rec, golden_data=data, asr_available=False)
    assert res['exit_code'] == 3 and 'whisperx' in res['cases'][0]['reason'] and not rec.runs
    res = golden(tmp_path, mini_manifest(), rec, golden_data=data, asr_available=False, visual_only=True)
    assert res['exit_code'] == 0 and rec.runs[0][2]['visual_only'] is True
    assert res['cases'][0]['transcription'] == 'visual_only_explicit'
    assert rec.runs[0][2]['provider'] is None and rec.runs[0][2]['transcript_fixture'] is None  # no LLM, no fixture
    wrong = tmp_path / 'wrong'
    clips(wrong, n=3)
    res = golden(tmp_path, mini_manifest(), Recorder(), golden_data=wrong)
    assert res['exit_code'] == 3 and 'expects 2' in res['cases'][0]['reason']


def test_golden_expected_flags_are_never_invented(tmp_path):
    data = tmp_path / 'data'
    clips(data)
    res = golden(tmp_path, mini_manifest(expected=None), Recorder(), golden_data=data)
    assert res['exit_code'] == 3 and 'no expected flags' in res['cases'][0]['reason']
    res = golden(tmp_path, mini_manifest(expected={'unsupported_goal': True}), Recorder(), golden_data=data)
    assert res['exit_code'] == 3 and 'observation unavailable' in res['cases'][0]['reason']   # judge not run
    res = golden(tmp_path, mini_manifest(expected={'unsupported_goal': True}), Recorder(unsupported=True),
                 golden_data=data, judge=True)
    assert res['exit_code'] == 0
    res = golden(tmp_path, mini_manifest(expected={'qa_result_in': ['partial']}), Recorder(result='passed'),
                 golden_data=data)
    assert res['status'] == 'failed' and res['exit_code'] == 1
    res = golden(tmp_path, mini_manifest(expected={'bogus': 1}), Recorder(), golden_data=data)
    assert res['exit_code'] == 3
    res = golden(tmp_path, mini_manifest(), Recorder(gates=False), golden_data=data)
    assert res['exit_code'] == 1
    res = golden(tmp_path, mini_manifest(), Recorder(), golden_data=data, case_ids=['nope'])
    assert res['exit_code'] == 3 and res['unknown_case_ids'] == ['nope']


def test_golden_labels_optional_and_recorded_and_judge_opt_in(tmp_path):
    data = tmp_path / 'data'
    clips(data)
    rec = Recorder()
    res = golden(tmp_path, mini_manifest(), rec, golden_data=data)
    assert res['cases'][0]['labels'].startswith('missing') and rec.evals[0][1]['labels'] is None
    assert rec.evals[0][1]['provider'] is None                                   # judge off by default
    labels = tmp_path / 'labels'
    labels.mkdir()
    (labels / 'ext.json').write_text('{"labels": []}')
    rec = Recorder()
    golden(tmp_path, mini_manifest(), rec, golden_data=data, labels_dir=labels, judge=True)
    assert rec.evals[0][1]['labels'] == str(labels / 'ext.json') and rec.evals[0][1]['provider'] == 'env'


def test_golden_runner_exceptions_and_synth_failures_are_unavailable(tmp_path):
    def bad_synth(*a, **k):
        raise RuntimeError('ffmpeg exploded')
    manifest = mini_manifest(kind='synthetic', synthetic={'clip_seconds': 1})
    res = golden(tmp_path, manifest, Recorder(), synth=bad_synth)
    assert res['exit_code'] == 3 and 'generation failed' in res['cases'][0]['reason']

    class Raiser(Recorder):
        def runner(self, *a, **k):
            raise RuntimeError('stage blew up')
    res = golden(tmp_path, manifest, Raiser(), synth=lambda d, seconds: tmp_path / 'fx.json')
    assert res['exit_code'] == 3 and 'run_episode raised' in res['cases'][0]['reason']


# --------------------------------------------------- real episode, fake judge
class FakeJudge:
    provider, model = 'fake', 'fake-judge-1'

    def __init__(self, goal_fit=4):
        self.calls = 0
        self.goal_fit = goal_fit

    def complete_json(self, **kwargs):
        self.calls += 1
        ids = [s['shot_id'] for s in json.loads(kwargs['prompt'])['edit']['shots']]
        scores = {c: 4 for c in CRITERIA}
        scores['goal_fit'] = self.goal_fit
        return LLMResult(data=dict(scores=scores, evidence={c: ids[:1] for c in CRITERIA}, weakest_shot=ids[-1],
                                   fix_instruction='Tighten the ending', verdict='pass', unsupported_goal=False,
                                   limitations=[]), usage={})


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp('evaltools')
    fixture = generate_distinct_footage(root / 'footage')
    report = cli.run_episode(root / 'footage', GOAL, episodes_root=root / 'episodes', episode_id='base',
                             transcript_fixture=fixture, provider=None)
    assert report['passed']
    return root


@pytest.fixture
def env(world, tmp_path):
    episodes = tmp_path / 'episodes'
    episodes.mkdir()
    for name in ('one', 'two'):
        shutil.copytree(world / 'episodes' / 'base', episodes / name,
                        ignore=shutil.ignore_patterns('work', 'proxies', 'frames'))
    return dict(episodes=episodes, evals=tmp_path / 'evals',
                base=['--episodes-root', str(episodes), '--evals-root', str(tmp_path / 'evals')])


def no_env_provider(monkeypatch):
    def forbidden():
        raise AssertionError('LLMProvider.from_env must not load unless --judge')
    monkeypatch.setattr(LLMProvider, 'from_env', staticmethod(forbidden))


def test_eval_cli_without_judge_never_loads_provider_and_dispatches_from_parent(env, capsys, monkeypatch):
    no_env_provider(monkeypatch)
    code = cli.main(['eval', 'one', *env['base']])                  # parent dispatch path
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report['episode_id'] == 'one' and report['judge']['status'] == 'unavailable'
    assert report['scores']['Q'] is None and report['scores']['J'] is None          # not fabricated
    lines = (env['evals'] / 'scoreboard.jsonl').read_text().splitlines()
    assert json.loads(lines[0]) == report and (env['episodes'] / 'one' / 'eval-report.json').is_file()
    code, _, err = run(['eval', 'ghost', *env['base']], capsys)
    assert code == 2 and 'not found' in err
    assert run(['eval', *env['base']], capsys)[0] == 2


def test_eval_cli_judge_flag_loads_env_provider_and_rate_calibrate_roundtrip(env, capsys, monkeypatch):
    fake = FakeJudge()
    monkeypatch.setattr(LLMProvider, 'from_env', staticmethod(lambda: fake))
    code, out, _ = run(['eval', 'one', '--judge', *env['base']], capsys)
    report = json.loads(out)
    assert code == 0 and fake.calls == 1 and report['judge']['status'] == 'scored' and report['scores']['Q'] is not None
    # non-interactive: both flags required
    code, _, err = run(['rate', 'one', '--score', '4', *env['base']], capsys, interactive=False)
    assert code == 2 and 'non-interactive' in err
    code, out, _ = run(['rate', 'one', '--score', '4', '--publish-without-edits', 'yes', '--rater', 'rk',
                        *env['base']], capsys, interactive=False)
    saved = json.loads(out)
    assert code == 0 and saved['render_hash'] == report['render_hash'] and saved['publish_without_edits'] is True
    # interactive asks for both missing values
    answers = iter(['3', 'no'])
    prompts = []
    code, out, _ = run(['rate', 'two', '--rater', 'rk', *env['base']], capsys, interactive=True,
                       input_fn=lambda p: prompts.append(p) or next(answers))
    assert code == 0 and len(prompts) == 2 and json.loads(out)['score'] == 3
    # invalid values and hash mismatch are refused, nothing appended
    bad = ['rate', 'one', '--score', '9', '--publish-without-edits', 'yes', *env['base']]
    assert run(bad, capsys, interactive=False)[0] == 2
    bad = ['rate', 'one', '--score', '3', '--publish-without-edits', 'maybe', *env['base']]
    assert run(bad, capsys, interactive=False)[0] == 2
    code, _, err = run(['rate', 'one', '--score', '3', '--publish-without-edits', 'no', '--render-hash', 'deadbeef',
                        *env['base']], capsys, interactive=False)
    assert code == 2 and 'render hash' in err
    assert len((env['evals'] / 'human-ratings.jsonl').read_text().splitlines()) == 2
    code, out, _ = run(['eval', '--calibration', '--rater', 'rk', *env['base']], capsys)
    cal = json.loads(out)
    assert code == 0 and cal['samples'] == 1 and cal['agreement_within_1_pct'] == 100.0   # 'two' has no evaluation
    assert cal['unmatched'][0]['episode_id'] == 'two' and cal['kappa'] is None


def test_rate_refuses_stale_render_and_missing_render(env, capsys):
    assert run(['eval', 'one', *env['base']], capsys)[0] == 0
    render = env['episodes'] / 'one' / 'finished.mp4'
    render.write_bytes(render.read_bytes() + b'\0')                      # render changed after evaluation
    args = ['--score', '3', '--publish-without-edits', 'no', *env['base']]
    code, _, err = run(['rate', 'one', *args], capsys, interactive=False)
    assert code == 2 and 'different render' in err
    (env['episodes'] / 'two' / 'finished.mp4').unlink()
    code, _, err = run(['rate', 'two', *args], capsys, interactive=False)
    assert code == 2 and 'finished.mp4' in err
    assert not (env['evals'] / 'human-ratings.jsonl').exists()


def test_calibration_cli_with_no_data_is_unavailable_nonzero(env, capsys):
    code, out, _ = run(['eval', '--calibration', *env['base']], capsys)
    assert code == 3 and json.loads(out)['agreement_within_1_pct'] is None


class PairJudge:
    """Prefers the edit shown in sheet-a.jpg regardless of slot, or always slot A when positional."""
    provider, model = 'fake', 'pair-1'

    def __init__(self, positional=False):
        self.calls, self.positional, self.dirs = 0, positional, []

    def complete_json(self, **kwargs):
        self.calls += 1
        self.dirs.append(kwargs['episode_dir'])
        first = Path(kwargs['images'][0]).name
        if self.positional:
            return LLMResult(data=dict(winner='A', reason='always first'), usage={})
        return LLMResult(data=dict(winner='A' if first == 'sheet-a.jpg' else 'B', reason='better opening'), usage={})


class BadPair:
    provider, model = 'fake', 'bad'

    def complete_json(self, **kwargs):
        return LLMResult(data=dict(winner='C'), usage={})


def test_pairwise_uses_isolated_ledger_and_detects_position_bias(env, capsys, monkeypatch):
    monkeypatch.setattr(LLMProvider, 'from_env', staticmethod(lambda: None))
    for name in ('one', 'two'):
        assert run(['eval', name, *env['base']], capsys)[0] == 0
    # without opt-in / provider => unavailable, zero calls
    code, out, _ = run(['eval', '--pairwise', 'one', 'two', *env['base']], capsys)
    assert code == 3 and 'pass --judge' in json.loads(out)['reason']
    code, out, _ = run(['eval', '--pairwise', 'one', 'two', '--judge', *env['base']], capsys)
    assert code == 3 and 'zero calls' in json.loads(out)['reason']
    one, two = env['episodes'] / 'one', env['episodes'] / 'two'
    fair = PairJudge()
    res = tools.pairwise_compare(one, two, provider=fair, evals_root=env['evals'])
    assert res['status'] == 'compared' and res['verdict'] == 'A' and res['winner_episode'] == 'one' and fair.calls == 2
    assert [o['order'] for o in res['outcomes']] == ['AB', 'BA']
    assert all(Path(d).parent.name == 'pairwise' and Path(d) not in (one, two) for d in fair.dirs)   # isolated ledger
    assert len((env['evals'] / 'scoreboard.jsonl').read_text().splitlines()) == 2      # pairwise is not an episode row
    assert json.loads((env['evals'] / 'pairwise.jsonl').read_text().splitlines()[0])['winner_episode'] == 'one'
    biased = tools.pairwise_compare(one, two, provider=PairJudge(positional=True), evals_root=env['evals'],
                                    persist=False)
    assert biased['verdict'] == 'inconsistent' and biased['winner_episode'] is None
    assert tools.pairwise_compare(one, two, provider=BadPair(), evals_root=env['evals'],
                                  persist=False)['status'] == 'failed'
    (two / 'eval-report.json').unlink()
    res = tools.pairwise_compare(one, two, provider=fair, evals_root=env['evals'])
    assert res['status'] == 'unavailable' and 'run eval first' in res['reason']


def test_usage_errors_exit_2(capsys):
    assert run(['eval', '--bogus'], capsys)[0] == 2
    assert run(['rate'], capsys)[0] == 2
    assert run(['eval', 'x', '--calibration', '--golden'], capsys)[0] == 2
