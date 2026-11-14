"""Read-only terminal scoreboard; no model calls or artifact contents displayed."""
from __future__ import annotations

from pathlib import Path

from . import eval_tools as tools
from .contracts import StageError, read_json, resolve_episode
from .evaluator import EvalReportError, validate_eval_report


def _qa_pass(root: Path, episode_id: str):
    """Read actual current QA, never infer it from evaluator gates or first-try."""
    try:
        directory = resolve_episode(root, episode_id)
        path = directory / 'qa-report.json'
        # Also refuse an artifact symlink escaping the isolated episode directory.
        if path.resolve().parent != directory:
            return None
        qa = read_json(path)
        if (isinstance(qa, dict) and qa.get('episode_id') == episode_id
                and type(qa.get('passed')) is bool):
            return qa['passed']
    except (StageError, OSError, ValueError, RuntimeError):
        pass
    return None


def _number(value, *, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def validated_reports(reports):
    """Return strict reports and rejection count without displaying their contents."""
    valid = []
    invalid = 0
    for report in reports:
        try:
            validate_eval_report(report)
        except (EvalReportError, ValueError, TypeError, KeyError):
            invalid += 1
        else:
            valid.append(report)
    return valid, invalid


def render_table(reports, *, episodes_root: Path, unreadable_lines=0, input_unavailable=False):
    """Validate before latest selection; count rejected lines without echoing them.

    Costs are cumulative snapshots, so history is never summed. QA is read from
    the supplied episodes root and describes current artifacts, not eval history.
    """
    valid, invalid = validated_reports(reports)
    latest = sorted(tools.latest_per_episode(valid), key=lambda r: r['episode_id'])
    headers = ['episode', 'cost USD', 'cost status', 'known_usd*', 'J', 'Q', 'D',
               'quality_per_dollar', 'wall time', 'G/pass', 'QA pass']
    rows = []
    qa_known = qa_passed = cost_known = 0
    known_spend = total_spend = 0.0
    known_missing = 0
    score_missing = {key: 0 for key in ('J', 'Q', 'D')}
    wall_missing = 0
    for report in latest:
        eid = report['episode_id']
        try:
            resolve_episode(episodes_root, eid)
            label = eid
        except (StageError, OSError, RuntimeError):
            # Unsafe identities may contain paths, terminal escapes or secrets.
            label = '[unsafe-id]'
        scores, cost = report['scores'], report['model_cost']
        usd = cost['usd']
        known = cost.get('known_usd', usd)
        if known is None:
            known_missing += 1
        else:
            known_spend += known
        if usd is not None:
            cost_known += 1
            total_spend += usd
        qpd = scores['Q'] / usd if scores['Q'] is not None and usd is not None and usd > 0 else None
        qa = _qa_pass(episodes_root, eid)
        qa_known += qa is not None
        qa_passed += qa is True
        for key in score_missing:
            score_missing[key] += scores[key] is None
        elapsed = report['timing']['elapsed_seconds']
        wall_missing += elapsed is None
        status = 'unknown/incomplete' if usd is None else cost['status']
        rows.append([label, _number(usd, digits=6), status, _number(known, digits=6),
                     *(_number(scores[k]) for k in ('J', 'Q', 'D')), _number(qpd),
                     'N/A' if elapsed is None else f'{elapsed:.2f}s',
                     f"{scores['G']}/{'PASS' if report['gates']['passed'] else 'FAIL'}",
                     'N/A' if qa is None else ('PASS' if qa else 'FAIL')])
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h)
              for i, h in enumerate(headers)]

    def line(cells):
        return ' | '.join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    output = [line(headers), '-+-'.join('-' * w for w in widths)]
    output.extend(line(row) for row in rows)
    if not rows:
        output.append('(no readable evaluation reports)')
    count = len(latest)
    output.append(f'Latest episodes: {count}; valid evaluation lines: {len(valid)}; '
                  f'invalid/corrupt lines: {invalid + unreadable_lines} '
                  f'(invalid reports: {invalid}, unreadable lines: {unreadable_lines})')
    if input_unavailable:
        output.append('Scoreboard input: unavailable (unreadable); no totals certified.')
    qa_rate = f'{100 * qa_passed / qa_known:.1f}%' if qa_known else 'N/A'
    output.append(f'Actual QA pass rate: {qa_rate} ({qa_passed}/{qa_known}); '
                  f'known: {qa_known}; unavailable: {count - qa_known} (excluded from denominator)')
    total = _number(total_spend, digits=6) if count and cost_known == count else 'N/A'
    output.append(f'Cost USD total: {total}; known_usd* subtotal: {_number(known_spend, digits=6)}; '
                  f'cost known: {cost_known}; cost missing: {count - cost_known}; '
                  f'known_usd missing: {known_missing}')
    output.append('Missing scores: ' + ', '.join(f'{k}={v}' for k, v in score_missing.items())
                  + f'; wall time missing: {wall_missing}')
    output.extend([
        '* known_usd is recorded known spend/estimate, not a complete bill when cost is unknown; local compute excluded.',
        'Cost = latest cumulative per-episode model spend; history is not summed. quality_per_dollar is N/A for missing Q or unknown/zero cost.',
        'QA = current qa-report passed field from episodes root; G/pass = evaluator gates, NOT QA. Wall time = timing.elapsed_seconds.',
    ])
    return '\n'.join(output)
