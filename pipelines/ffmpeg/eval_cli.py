"""Dispatch target for ``eval``, ``rate`` and ``scoreboard``: ``main(args) -> exit code``.

``args`` includes the sub-command as its first element (the parent CLI passes
``['eval', ...]``). Exit codes: 0 ok, 1 failed/regression, 2 usage or input error,
3 unavailable (a dependency, expected flag or comparable baseline is missing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import eval_tools as tools
from .eval_tools import EXIT_FAIL, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE, EvalToolError

REPO = Path(__file__).resolve().parents[2]


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise EvalToolError(message)


def _episode(root,episode_id):
    from .contracts import resolve_episode,StageError
    try:return resolve_episode(root,episode_id)
    except StageError as exc:raise EvalToolError(str(exc)) from None


def _common(p):
    p.add_argument('--episodes-root', type=Path, default=REPO / 'episodes')
    p.add_argument('--evals-root', type=Path, default=Path(os.environ.get('FRAMEFOUNDRY_EVALS_ROOT',REPO/'evals')))


def _parser():
    root = _Parser(prog='framefoundry')
    sub = root.add_subparsers(dest='command', required=True, parser_class=_Parser)
    e = sub.add_parser('eval', help='evaluate one episode, or --calibration / --golden / --pairwise')
    e.add_argument('episode_id', nargs='?')
    _common(e)
    e.add_argument('--judge', action='store_true', help='enable the LLM judge (LLMProvider.from_env, only after gates pass)')
    e.add_argument('--labels', type=Path, help='optional human highlight labels JSON (enables H)')
    mode = e.add_mutually_exclusive_group()
    mode.add_argument('--calibration', action='store_true', help='judge-vs-human agreement report')
    mode.add_argument('--golden', action='store_true', help='run the golden manifest')
    mode.add_argument('--pairwise', nargs=2, metavar=('EPISODE_A', 'EPISODE_B'), help='explicit A/B comparison (needs --judge)')
    e.add_argument('--rater', help='rater id for --calibration (required if several raters exist)')
    e.add_argument('--manifest', type=Path, default=tools.DEFAULT_MANIFEST)
    e.add_argument('--case', action='append', help='golden case id (repeatable); default all')
    e.add_argument('--golden-data', type=Path, help='directory holding the extracted external release (footage/<subdir>)')
    e.add_argument('--download', action='store_true', help='opt in to downloading the private release with gh')
    e.add_argument('--labels-dir', type=Path, help='directory with optional <case_id>.json user labels')
    e.add_argument('--visual-only', action='store_true', help='explicit diagnostic run without ASR (golden external cases)')
    e.add_argument('--planner-llm', action='store_true', help='golden: let the planner use LLMProvider.from_env')
    e.add_argument('--work-dir', type=Path, help='golden scratch directory (default: temp)')
    e.add_argument('--run-id', help='golden run id suffix')
    e.add_argument('--no-persist', action='store_true', help='do not write eval-report.json / scoreboard')
    r = sub.add_parser('rate', help='record a human rating (JSONL, locked append)')
    r.add_argument('episode_id')
    _common(r)
    r.add_argument('--score', help='1..5')
    r.add_argument('--publish-without-edits', choices=['yes', 'no'])
    r.add_argument('--rater', default=os.environ.get('FRAMEFOUNDRY_RATER', tools.DEFAULT_RATER))
    r.add_argument('--render-hash', help='optional expected render sha256; refused if it differs from finished.mp4')
    r.add_argument('--note')
    s = sub.add_parser('scoreboard', help='summarise evals/scoreboard.jsonl')
    _common(s)
    s.add_argument('--json', action='store_true')
    s.add_argument('--lines', action='store_true', help='print the raw scoreboard lines (whole reports)')
    s.add_argument('--baseline', type=Path, help='baseline scoreboard JSONL (e.g. from main) to compare mean Q against')
    s.add_argument('--tolerance', type=float, default=tools.REGRESSION_TOLERANCE)
    return root


def _print(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))


def _eval(a, out):
    from .evaluator import evaluate_episode
    if a.calibration:
        reports, _ = tools.load_reports(a.evals_root)
        ratings, _ = tools.load_ratings(a.evals_root)
        result = tools.calibrate(reports, ratings, rater=a.rater)
        _print(result)
        return EXIT_OK if result['samples'] else EXIT_UNAVAILABLE
    if a.golden:
        manifest = tools.load_manifest(a.manifest)
        with tempfile.TemporaryDirectory(prefix='golden-') as tmp:
            work = a.work_dir or Path(tmp)
            result = tools.run_golden(manifest, work_dir=work, evals_root=a.evals_root, case_ids=a.case,
                                      golden_data=a.golden_data, download=a.download, visual_only=a.visual_only,
                                      labels_dir=a.labels_dir, judge=a.judge, planner_llm=a.planner_llm, run_id=a.run_id)
        _print(result)
        return result['exit_code']
    if a.pairwise:
        provider = None
        if a.judge:
            from .llm_provider import LLMProvider, ProviderError
            try:
                provider = LLMProvider.from_env()
            except ProviderError as exc:
                _print(dict(status='unavailable', reason=f'provider configuration invalid: {exc}'))
                return EXIT_UNAVAILABLE
        result = tools.pairwise_compare(_episode(a.episodes_root,a.pairwise[0]), _episode(a.episodes_root,a.pairwise[1]),
                                        provider=provider, evals_root=a.evals_root, persist=not a.no_persist)
        if provider is None and a.judge:
            result['reason'] = 'no API key configured for the selected provider; pairwise performs zero calls'
        elif not a.judge and result.get('status') == 'unavailable':
            result['reason'] = 'pairwise makes model calls: pass --judge to opt in'
        _print(result)
        return {'compared': EXIT_OK, 'failed': EXIT_FAIL}.get(result['status'], EXIT_UNAVAILABLE)
    if not a.episode_id:
        raise EvalToolError('eval needs an episode id, or one of --calibration/--golden/--pairwise')
    directory = _episode(a.episodes_root,a.episode_id)
    if not directory.is_dir():
        raise EvalToolError(f'Episode directory not found: {directory}')
    if a.labels and not a.labels.is_file():
        raise EvalToolError(f'Labels file not found: {a.labels}')
    report = evaluate_episode(directory, provider='env' if a.judge else None, evals_root=a.evals_root,
                              labels=a.labels, persist=not a.no_persist)
    _print(report)
    return EXIT_OK if report['gates']['passed'] else EXIT_FAIL


def _rate(a, input_fn, interactive):
    directory = _episode(a.episodes_root,a.episode_id)
    if not directory.is_dir():
        raise EvalToolError(f'Episode directory not found: {directory}')
    score, publish = a.score, a.publish_without_edits
    if score is None or publish is None:
        if not interactive:
            raise EvalToolError('non-interactive rating needs both --score and --publish-without-edits')
        if score is None:
            score = input_fn('Score 1-5: ')
        if publish is None:
            publish = input_fn('Publish without edits? (yes/no): ')
    record = tools.record_rating(a.episode_id, directory, score=tools.parse_score(score),
                                 publish_without_edits=tools.parse_yes_no(publish), rater=a.rater,
                                 render_hash=a.render_hash, note=a.note, evals_root=a.evals_root)
    _print(record)
    return EXIT_OK


def _scoreboard(a):
    reports, bad = tools.load_reports(a.evals_root)
    if a.lines:
        for r in reports:
            print(json.dumps(r, sort_keys=True, allow_nan=False))
        return EXIT_OK
    summary = tools.summarize(reports, unreadable_lines=bad)
    ratings,_=tools.load_ratings(a.evals_root)
    summary['judge_vs_human']=tools.calibrate(reports,ratings)
    code = EXIT_OK
    if a.baseline:
        base, base_bad = tools.read_jsonl(a.baseline) if a.baseline.is_file() else ([], 0)
        comparison = tools.compare_baseline(reports, base, tolerance=a.tolerance)
        if not a.baseline.is_file():
            comparison['reason'] = f'baseline file not found: {a.baseline}'
        summary['baseline_comparison'] = comparison
        code = tools.compare_exit_code(comparison)
    _print(summary)
    return code


def main(args=None, *, input_fn=input, interactive=None):
    argv = list(sys.argv[1:] if args is None else args)
    try:
        a = _parser().parse_args(argv)
        if a.command == 'eval':
            return _eval(a, sys.stdout)
        if a.command == 'rate':
            tty = sys.stdin.isatty() if interactive is None else interactive
            return _rate(a, input_fn, tty)
        return _scoreboard(a)
    except EvalToolError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_USAGE
    except (EOFError, KeyboardInterrupt):
        print('error: input cancelled', file=sys.stderr)
        return EXIT_USAGE


if __name__ == '__main__':
    raise SystemExit(main())
