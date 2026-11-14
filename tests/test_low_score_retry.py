"""CPU-only repair/budget proofs: real FFmpeg artifacts, offline provider transports."""
from decimal import Decimal
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from pipelines.ffmpeg import cli, llm_provider as accounting
from pipelines.ffmpeg.contracts import StageError, read_json, validate_report
from pipelines.ffmpeg.eval_gates import generate_distinct_footage
from pipelines.ffmpeg.eval_judge import CRITERIA
from scripts import arizona_demo

FIX = 'Replace the weak opening with the strongest trip moment, then keep the remaining shot order'


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    # No credentials are needed and the test must never consume inherited keys.
    for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', accounting.BUDGET_ENV,
                 accounting.TOKEN_BUDGET_ENV, 'FRAMEFOUNDRY_PROJECT_SPENT_USD'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_BUDGET_DIR', str(tmp_path / 'budget'))
    monkeypatch.setenv('FRAMEFOUNDRY_CACHE_DIR', str(tmp_path / 'cache'))


@pytest.fixture(scope='module')
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp('low-score-media')
    return root, generate_distinct_footage(root)


def offline_provider(scores, *, change_render=True, short_initial=False, fix=FIX):
    calls = []

    def send(url, headers, body, timeout):
        request = json.loads(body)
        prompt = json.loads(next(c['text'] for c in request['messages'][0]['content'] if c['type'] == 'text'))
        if 'candidates' in prompt:
            remaining = 8000 if short_initial and not calls else 10000
            segments = []
            for candidate in prompt['candidates']:
                duration = min(candidate['end_ms'] - candidate['start_ms'], remaining)
                if duration:
                    segments.append(dict(source_id=candidate['source_id'],
                                         raw_start_ms=candidate['start_ms'],
                                         raw_end_ms=candidate['start_ms'] + duration,
                                         reason='Offline sampled moment'))
                    remaining -= duration
            if calls and change_render:
                segments.reverse()
            data = dict(interpretation='Trip highlights', assumptions=['Synthetic offline test'],
                        segments=segments, coverage=dict(status='full', limitations=[]))
            calls.append(('planner', prompt))
        else:
            ids = [s['shot_id'] for s in prompt['edit']['shots']]
            # Intentionally supply pass; production normalization owns the verdict.
            data = dict(scores=scores, evidence={c: ids[:1] for c in CRITERIA},
                        weakest_shot=ids[0], fix_instruction=fix, verdict='pass',
                        unsupported_goal=False, limitations=['Still frames only'])
            calls.append(('judge', prompt))
        return 200, json.dumps(dict(stop_reason='end_turn',
                                    content=[dict(type='text', text=json.dumps(data))],
                                    usage=dict(input_tokens=1000, output_tokens=100))).encode()

    return accounting.LLMProvider('anthropic', 'offline-fake-key', 'claude-sonnet-4-5',
                                  (Decimal(3), Decimal(15)), transport=send), calls


def run(media, tmp_path, provider, **kwargs):
    report = cli.run_episode(media[0], 'Create a 10 second family-trip teaser',
                             episodes_root=tmp_path / 'episodes', episode_id='repair',
                             transcript_fixture=media[1], provider=provider,
                             evals_root=tmp_path / 'evals', **kwargs)
    root = tmp_path / 'episodes' / 'repair'
    assert report['passed'], report
    assert -15 <= report['integrated_lufs'] <= -13
    assert report['true_peak_dbtp'] <= -1 and (root / 'contact-sheet.jpg').stat().st_size > 0
    validate_report(read_json(root / 'qa-report.json'))
    return report, read_json(root / 'eval-report.json'), root


@pytest.mark.parametrize('scores,expected_j', [
    ({c: 3 for c in CRITERIA}, 0.5),
    ({c: 2 if c == 'opening' else 5 for c in CRITERIA}, 0.875),
    ({c: 2 for c in CRITERIA}, 0.25),
])
def test_scored_weak_edit_gets_one_concrete_replan_and_no_stale_final_scores(media, tmp_path, scores, expected_j):
    provider, calls = offline_provider(scores)
    report, evaluation, root = run(media, tmp_path, provider)
    assert [role for role, _ in calls] == ['planner', 'judge', 'planner']
    assert calls[-1][1]['fix_instruction'] == FIX
    assert len(report['attempts']) == 2 <= 3
    retained = [json.loads(change) for change in report['attempts'][0]['changes'] if change.startswith('{')]
    assert len(retained) == 1
    original = retained[0]
    assert original['kind'] == 'judged_attempt' and original['attempt'] == 1
    assert original['judge']['status'] == 'scored' and original['judge']['scores'] == scores
    assert original['judge']['verdict'] == ('fail' if expected_j == 0.25 else 'pass')
    assert original['judge']['fix_instruction'] == FIX and original['judge']['evidence']
    assert original['scores']['J'] == expected_j and original['scores']['Q'] is not None
    assert original['render_hash'] == hashlib.sha256((root / 'work/attempt-1/candidate.mp4').read_bytes()).hexdigest()
    assert original['render_hash'] != evaluation['render_hash']
    assert evaluation['render_hash'] == hashlib.sha256((root / 'finished.mp4').read_bytes()).hexdigest()
    assert evaluation['scores']['J'] is None and evaluation['scores']['Q'] is None
    assert evaluation['scores']['G'] == 1 and evaluation['judge']['status'] == 'unavailable'
    assert report['goal_judge']['score'] is None
    assert sum(row['role'] == 'judge' for row in report['ai_usage']) == 1
    assert sum(row['role'] == 'planner' for row in report['ai_usage']) == 2


def test_identical_repair_reuses_only_hash_matched_judgment_without_dispatch(media, tmp_path):
    provider, calls = offline_provider({c: 3 for c in CRITERIA}, change_render=False)
    report, evaluation, root = run(media, tmp_path, provider)
    assert [role for role, _ in calls] == ['planner', 'judge', 'planner']
    assert len(report['attempts']) == 2
    assert evaluation['judge']['cache_hit'] and evaluation['scores']['J'] == 0.5
    assert evaluation['scores']['Q'] is not None
    assert (root / 'work/attempt-1/candidate.mp4').read_bytes() == (root / 'finished.mp4').read_bytes()


def test_good_score_does_not_replan_even_with_a_fix(media, tmp_path):
    provider, calls = offline_provider({c: 4 for c in CRITERIA})
    report, evaluation, _ = run(media, tmp_path, provider)
    assert [role for role, _ in calls] == ['planner', 'judge']
    assert len(report['attempts']) == 1 and evaluation['scores']['J'] == 0.75


def test_retry_opt_out_still_applies_to_low_score(media, tmp_path):
    provider, calls = offline_provider({c: 3 for c in CRITERIA})
    report, evaluation, _ = run(media, tmp_path, provider, planner_retry=False)
    assert [role for role, _ in calls] == ['planner', 'judge']
    assert len(report['attempts']) == 1 and evaluation['scores']['J'] == 0.5


def test_deterministic_duration_gate_still_gets_one_repair(media, tmp_path):
    provider, calls = offline_provider({c: 4 for c in CRITERIA}, short_initial=True)
    report, evaluation, _ = run(media, tmp_path, provider)
    assert [role for role, _ in calls] == ['planner', 'planner', 'judge']
    assert 'duration_target' in calls[1][1]['fix_instruction']
    assert len(report['attempts']) == 2 and evaluation['first_try_passed'] is False


@pytest.mark.parametrize('status,j,opening,verdict,fix,expected', [
    ('scored', 0.5, 3, 'pass', '  ' + FIX + '  ', FIX),
    ('scored', 0.875, 2, 'pass', FIX, FIX),
    ('scored', 0.9, 5, 'fail', FIX, FIX),
    ('scored', 0.75, 4, 'pass', FIX, None),
    ('scored', 0.5, 2, 'fail', '', None),
    ('scored', 0.5, 2, 'fail', '  ', None),
    ('scored', 0.5, 2, 'fail', None, None),
    ('unavailable', 0.5, 2, 'fail', FIX, None),
    ('failed', 0.5, 2, 'fail', FIX, None),
])
def test_repair_requires_scored_weakness_and_nonempty_fix(status, j, opening, verdict, fix, expected):
    evaluation = dict(scores=dict(J=j), judge=dict(status=status, scores=dict(opening=opening),
                                                  verdict=verdict, fix_instruction=fix))
    assert cli._judge_repair_fix(evaluation) == expected


@pytest.mark.parametrize('configured,expected', [(None, '0.10'), ('.25', '0.10'), ('5', '0.10'),
                                                 ('.10', '0.10'), ('.03', '0.03')])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_arizona_passes_shared_budget_in_child_env_preserving_stricter_limit_and_flags(tmp_path, monkeypatch, configured, expected, diagnostic):
    if configured is not None:
        monkeypatch.setenv(accounting.BUDGET_ENV, configured)
    # Prove inherited secret configuration is copied, never edited by the runner.
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-never-used')
    before = dict(os.environ)
    monkeypatch.setattr(arizona_demo, 'fetch', lambda _: tmp_path / 'clips')
    dispatched = []
    def subprocess_run(command, *, check, env):
        dispatched.append((command, check, env))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(arizona_demo.subprocess, 'run', subprocess_run)
    args = ['--episode', 'arizona', '--episodes-root', str(tmp_path), '--evals-root', str(tmp_path / 'evals')]
    if diagnostic:
        args += ['--visual-only', '--asr-provider', 'local', '--no-planner-retry']
    assert arizona_demo.main(args) == 7
    assert dict(os.environ) == before
    command, check, env = dispatched[0]
    assert check is False and env is not os.environ
    assert env[accounting.BUDGET_ENV] == expected and env['OPENAI_API_KEY'] == before['OPENAI_API_KEY']
    assert ('--visual-only' in command) == diagnostic
    assert command[command.index('--asr-provider') + 1] == ('local' if diagnostic else 'auto')
    assert ('--no-planner-retry' in command) == diagnostic


@pytest.mark.parametrize('name,value', [(accounting.BUDGET_ENV, value) for value in
                                       ('', 'NaN', 'inf', '-1', '0', 'abc', '5.01', '1e1')] +
                                      [(accounting.TOKEN_BUDGET_ENV, 'broken')])
def test_arizona_malformed_budget_is_terminal_before_fetch_or_pipeline(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(arizona_demo, 'fetch', lambda _: pytest.fail('Do not fetch with invalid config'))
    with pytest.raises(SystemExit) as error:
        arizona_demo.main([])
    assert error.value.code == 2 and os.environ[name] == value


def test_arizona_budget_reaches_real_child_without_changing_parent_env(monkeypatch):
    import subprocess
    monkeypatch.setenv(accounting.BUDGET_ENV, '.25')
    env = arizona_demo.child_environment()
    result = subprocess.run([arizona_demo.sys.executable, '-c',
                             'from pipelines.ffmpeg.llm_provider import episode_budget_usd; print(episode_budget_usd())'],
                            env=env, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == '0.10' and os.environ[accounting.BUDGET_ENV] == '.25'


@pytest.mark.parametrize('role', ['planner', 'judge'])
def test_arizona_asr_and_llm_share_one_cap(media, tmp_path, monkeypatch, role):
    from pipelines.ffmpeg.proxy import probe
    from pipelines.ffmpeg.transcription import OpenAITranscriber
    env = arizona_demo.child_environment()
    monkeypatch.setenv(accounting.BUDGET_ENV, env[accounting.BUDGET_ENV])
    source_path = sorted(media[0].glob('*.mp4'))[0]
    duration = probe(source_path)['duration_ms']
    source = dict(source_id='source-0001', path=str(source_path), duration_ms=duration, has_audio=True)
    asr_calls = []
    def transcribe(*args):
        asr_calls.append(True)
        return 200, json.dumps(dict(text='hello', duration=duration / 1000,
                                    words=[dict(word='hello', start=.2, end=.6)])).encode()
    OpenAITranscriber('offline-fake-key', transport=transcribe).words(source, episode_dir=tmp_path)
    assert len(asr_calls) == 1
    # Reserve just below .10 for this LLM request. It fits alone, but not after ASR.
    p = accounting.LLMProvider('anthropic', 'offline-fake-key', 'claude-sonnet-4-5',
                               (Decimal(0), Decimal('99.8')), transport=lambda *a: pytest.fail('Shared cap must block dispatch'))
    with pytest.raises(accounting.ProviderBudgetError, match='budget would exceed'):
        p.complete_json(system='offline', prompt='{}', images=[], role=role,
                        episode_dir=tmp_path, max_output_tokens=1000)
    ledger = read_json(tmp_path / 'llm-usage.json')
    assert ledger['budget_usd'] == .10
    assert [row['role'] for row in ledger['calls']] == ['transcription', role]
    assert ledger['calls'][0]['cost_usd'] > 0 and ledger['calls'][1]['status'] == 'blocked'


def test_low_score_with_no_actionable_fix_does_not_retry(media, tmp_path):
    # Whitespace satisfies the strict judge string schema but is not actionable.
    provider, calls = offline_provider({c: 3 for c in CRITERIA}, fix='   ')
    report, evaluation, _ = run(media, tmp_path, provider)
    assert [role for role, _ in calls] == ['planner', 'judge']
    assert len(report['attempts']) == 1 and evaluation['scores']['J'] == 0.5


def test_renderer_repair_plus_judge_replan_stays_within_three_renders(media, tmp_path, monkeypatch):
    original = cli.validate_candidate
    rendered = []
    def validate(*args, **kwargs):
        report = original(*args, **kwargs)
        rendered.append(True)
        if len(rendered) == 1:
            for check in report['checks']:
                if check['name'] == 'loudness':
                    check.update(status='fail', reason='Injected first-render normalization defect')
            report.update(passed=False, result='failed', output_path=None, integrated_lufs=-20,
                          error=StageError('QA_FAILED', 'validator', 'Injected loudness failure').as_dict())
        return report
    monkeypatch.setattr(cli, 'validate_candidate', validate)
    provider, calls = offline_provider({c: 3 for c in CRITERIA})
    report, evaluation, _ = run(media, tmp_path, provider)
    assert [role for role, _ in calls] == ['planner', 'judge', 'planner']
    assert len(report['attempts']) == len(rendered) == 3
    assert [attempt['passed'] for attempt in report['attempts']] == [False, True, True]
    assert evaluation['scores']['J'] is None and evaluation['scores']['Q'] is None


def test_planner_retry_does_not_erase_source_no_asr_limitations(media, tmp_path, monkeypatch):
    from pipelines.ffmpeg.transcription import OpenAITranscriber
    # Real ASR extraction and router audit, but never a real service or secret.
    transcription_calls = []
    def transcribe(url, headers, body, timeout):
        transcription_calls.append(True)
        words = [] if len(transcription_calls) < 3 else [dict(word='bad', start=1, end=.5)]
        return 200, json.dumps(dict(text='' if not words else 'bad', duration=4, words=words)).encode()
    remote = OpenAITranscriber('offline-fake-key', transport=transcribe)
    monkeypatch.setattr(OpenAITranscriber, 'from_env', classmethod(lambda cls: remote))
    original = cli.validate_candidate
    render_count = 0
    def validate(*args, **kwargs):
        nonlocal render_count
        report = original(*args, **kwargs)
        render_count += 1
        if render_count == 1:
            for check in report['checks']:
                if check['name'] == 'timeline_and_padding':
                    check.update(status='fail', reason='Injected technical repair')
            report.update(passed=False, result='failed', output_path=None,
                          error=StageError('QA_FAILED', 'validator', 'Injected technical repair').as_dict())
        return report
    monkeypatch.setattr(cli, 'validate_candidate', validate)
    provider, calls = offline_provider({c: 4 for c in CRITERIA})
    report = cli.run_episode(media[0], 'Create a 10 second family-trip teaser',
                             episodes_root=tmp_path / 'episodes', episode_id='source-no-asr',
                             provider=provider, evals_root=tmp_path / 'evals', asr_provider='openai')
    root = tmp_path / 'episodes/source-no-asr'
    assert report['passed'] and report['result'] == 'partial', report
    assert len(transcription_calls) == 3 and [role for role, _ in calls] == ['planner', 'planner']
    assert len(report['attempts']) == 2
    assert report['coverage']['status'] == 'partial'
    assert any('ASR_FAILED' in limitation and 'source-0003' in limitation for limitation in report['coverage']['limitations'])
    spec = read_json(root / 'narrative-spec.json')
    assert spec['coverage']['status'] == 'partial' and any('source-0003' in note for note in spec['coverage']['limitations'])
    evaluation = read_json(root / 'eval-report.json')
    gate = next(check for check in evaluation['gates']['checks'] if check['name'] == 'word_cut_integrity')
    assert gate['status'] == 'not_run' and 'degraded' in gate['reason']
    assert evaluation['scores']['G'] == 0 and evaluation['scores']['J'] is None
    validate_report(report)
