"""Static, escaped, offline scoreboard. No scripts, providers, or page-view requests."""
from __future__ import annotations

import html
import math
import os
import statistics
import tempfile
from pathlib import Path

from . import eval_tools as tools
from .contracts import StageError, resolve_episode
from .demo_scoreboard import _qa_pass, validated_reports
from .evaluator import sha256_file
from .factory_history import collect_factory_history

TREND_EPISODES = ('az-0500', 'az-0550', 'az-0600', 'az-0700')


def _escape(value):
    return html.escape(str(value), quote=True)


def _number(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _safe_directory(root, eid):
    directory = resolve_episode(root, eid)
    if (Path(root) / eid).is_symlink():
        raise ValueError('episode symlink')
    return directory


def _artifact_qa(root, eid):
    try:
        directory = _safe_directory(root, eid)
        if (directory / 'qa-report.json').is_symlink():
            return None
        return _qa_pass(root, eid)
    except (StageError, OSError, ValueError, RuntimeError):
        return None


def _shipped(report, root, qa):
    """Only verified current local exports count, not social publication or PR state."""
    if (qa is not True or report['scores']['G'] != 1 or report['gates']['passed'] is not True
            or any(check['status'] != 'pass' for check in report['gates']['checks'])):
        return False
    try:
        directory = _safe_directory(root, report['episode_id'])
        path = directory / 'finished.mp4'
        if path.is_symlink() or path.resolve().parent != directory or not path.is_file() or path.stat().st_size <= 0:
            return False
        return bool(report['render_hash']) and sha256_file(path) == report['render_hash']
    except (StageError, OSError, ValueError, RuntimeError):
        return False


def episode_metrics(reports, *, episodes_root, unreadable_lines=0):
    """Latest cumulative costs include failed episodes, never summed over retry history."""
    valid, invalid = validated_reports(reports)
    latest = sorted(tools.latest_per_episode(valid), key=lambda r: r['episode_id'])
    rows = []
    for report in latest:
        eid = report['episode_id']
        try:
            _safe_directory(episodes_root, eid)
            label = eid
        except (StageError, OSError, ValueError, RuntimeError):
            label = '[unsafe-id]'
        qa = _artifact_qa(episodes_root, eid)
        cost, scores = report['model_cost'], report['scores']
        usd = cost['usd']
        rows.append(dict(episode_id=eid, label=label, scores=scores, usd=usd,
                         known_usd=cost.get('known_usd', usd),
                         cost_status='unknown/incomplete' if usd is None else cost['status'],
                         qpd=scores['Q'] / usd if scores['Q'] is not None and usd is not None and usd > 0 else None,
                         wall=report['timing']['elapsed_seconds'], gates=report['gates']['passed'],
                         qa=qa, shipped=_shipped(report, episodes_root, qa)))
    qa_known = sum(r['qa'] is not None for r in rows)
    qa_passed = sum(r['qa'] is True for r in rows)
    shipped = sum(bool(r['shipped']) for r in rows)
    cost_known = sum(r['usd'] is not None for r in rows)
    known_subtotal = math.fsum(r['known_usd'] for r in rows if r['known_usd'] is not None)
    # usd is the total when available; known_usd only fills incomplete snapshots.
    cohort_subtotal = math.fsum(r['usd'] if r['usd'] is not None else (r['known_usd'] or 0) for r in rows)
    complete = bool(rows) and cost_known == len(rows)
    return dict(rows=rows, valid_lines=len(valid), rejected_lines=invalid + unreadable_lines,
                qa_known=qa_known, qa_passed=qa_passed, shipped=shipped, cost_known=cost_known,
                known_subtotal=known_subtotal, known_missing=sum(r['known_usd'] is None for r in rows),
                cohort_subtotal=cohort_subtotal, cost_complete=complete,
                estimated=any(r['cost_status'] == 'estimated' for r in rows),
                cost_per_shipped=cohort_subtotal / shipped if shipped else None)


def _table(headers, rows):
    return ('<div class="table-wrap"><table><thead><tr>'
            + ''.join('<th scope="col">' + _escape(h) + '</th>' for h in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + _escape(c) + '</td>' for c in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _trend(rows):
    by_id = {r['episode_id']: r['scores']['Q'] for r in rows if r['label'] != '[unsafe-id]'}
    items = list(TREND_EPISODES[:3])
    if TREND_EPISODES[3] in by_id:
        items.append(TREND_EPISODES[3])
    svg = ['<svg viewBox="0 0 720 210" role="img" aria-labelledby="trend-title trend-desc">',
           f'<title id="trend-title">Latest Q trend: {items[0]} to {items[-1]}</title>',
           '<desc id="trend-desc">Missing episodes or scores are N/A gaps, never zero. Q scale 0 to 1.</desc>']
    for score in (0, .5, 1):
        y = 155 - score * 120
        svg.append(f'<path class="grid" d="M60 {y}H670"/><text x="20" y="{y+5}">{score:g}</text>')
    previous = None
    values = []
    for index, eid in enumerate(items):
        x = 100 + index * (520 / (len(items) - 1))
        q = by_id.get(eid)
        if q is not None:
            y = 155 - q * 120
            if previous is not None:
                svg.append(f'<path class="trend" d="M{previous[0]} {previous[1]}L{x} {y}"/>')
            svg.append(f'<circle class="point" cx="{x}" cy="{y}" r="4"/>')
            svg.append(f'<text x="{x}" y="{y-12}" text-anchor="middle">{q:.3f}</text>')
            previous = (x, y)
        else:
            svg.append(f'<text x="{x}" y="105" text-anchor="middle">N/A</text>')
            previous = None
        svg.append(f'<text x="{x}" y="190" text-anchor="middle">{eid}</text>')
        values.append((eid, _number(q)))
    svg.append('</svg>')
    return ''.join(svg) + _table(['Episode', 'Latest Q'], values)


def _factory_section(factory):
    git, pr = factory['git'], factory['pull_request']
    # Source links are fixed/audited, not episode-controlled hrefs.
    from .factory_history import PR_URL
    out = ['<section><h2>Software factory</h2>',
           '<p>Generation: ' + _escape(factory['generated_at']) + '. Audited snapshot: '
           + _escape(factory['fetched_at']) + '.</p>',
           _table(['Metric', 'Value', 'Scope / source'], [
               ('Git commit count', git['commits'] if git['commits'] is not None else 'N/A',
                git['scope'] + ('; ' + git['reason'] if git['reason'] else '')),
               ('GitHub PR #1 commit count', pr['commits'] if pr['commits'] is not None else 'N/A',
                'GitHub PR commits; ' + pr['source']),
               ('GitHub PR #1 state', pr['state'] or 'N/A', pr['source']),
               ('GitHub PR #1 createdAt', pr['created_at'] or 'N/A', pr['source']),
               ('Work orders', len(factory['work_orders']), 'audited snapshot; all 7 In Review, not done'),
           ]), '<p>PR metadata observed: ' + _escape(pr['observed_at']) + '. '
           + _escape(pr['reason'] or 'Verified at generation time.') + '</p>',
           f'<p><a href="{PR_URL}">PR #1 source</a>. Local main..HEAD and GitHub PR counts have distinct scopes and may differ.</p>',
           '<h3>First recorded PR-title coverage</h3>',
           '<p>Proxy for recorded scope coverage, not completion, delivery, merge, or a mapping of commits to tickets.</p>']
    out.append(_table(['WO', 'Status (snapshot)', 'Created UTC', 'First title coverage UTC',
                       'Elapsed minutes', 'Evidence basis'], [
        (wo['id'], wo['status'], wo['created_at'], wo['first_coverage_at'],
         f"{wo['coverage_minutes']:.3f}", wo['coverage_basis']) for wo in factory['work_orders']]))
    base = 'https://factory.8090.ai/project/a54860bf-bf30-42d7-b6a8-a08af85c454c/work-orders/'
    out.append('<p>Factory sources: ' + ', '.join(f'<a href="{base}{i}">WO-{i}</a>' for i in range(1, 8)) + '.</p>')
    out.extend('<p class="note">' + _escape(note) + '</p>' for note in factory['notes'])
    out.append('</section>')
    return ''.join(out)


def _presenter(metrics, factory):
    rows=metrics['rows']
    by_id={r['episode_id']:r for r in rows}
    trend=[(eid,by_id[eid]['scores']['Q'] if eid in by_id else None) for eid in TREND_EPISODES]
    latest=next(((eid,q) for eid,q in reversed(trend) if q is not None),(None,None))
    timings=[wo['coverage_minutes'] for wo in factory['work_orders'] if wo.get('coverage_minutes') is not None]
    rate=f"{100*metrics['qa_passed']/metrics['qa_known']:.1f}%" if metrics['qa_known'] else 'N/A'
    cost=_number(metrics['cost_per_shipped'],4)
    cost_label='estimated' if metrics['estimated'] else 'recorded'
    if not metrics['cost_complete']:cost_label='known subtotal · incomplete'
    cards=[('Latest Q',_number(latest[1]),latest[0] or 'No Q available'),
           ('AI cost / shipped',cost+' USD',cost_label+' · failed episodes included'),
           ('Render QA pass rate',rate,f"{metrics['qa_passed']}/{metrics['qa_known']} known episodes"),
           ('Ticket → PR',_number(statistics.median(timings),1)+' min' if timings else 'N/A','median first title coverage · not completion'),
           ('Episodes shipped',str(metrics['shipped']),'verified local exports')]
    return ('<section id="presenter"><h2>Presenter view</h2><div class="cards">'+
            ''.join('<div class="card"><span class="eyebrow">'+_escape(label)+'</span><div class="value">'+_escape(value)+'</div><p class="note">'+_escape(note)+'</p></div>' for label,value,note in cards)+
            '</div><p class="q-trend"><strong>Q trend</strong> '+
            ' → '.join(_escape(eid)+': <strong>'+_number(q)+'</strong>' for eid,q in trend)+
            '</p></section>')


def render_html(reports, *, episodes_root, unreadable_lines=0, input_unavailable=False,
                assume_manual_minutes_per_episode=None, factory=None, baseline=None):
    """One self-contained document; missing local Mac episodes remain explicit gaps."""
    minutes = assume_manual_minutes_per_episode
    if minutes is not None and (isinstance(minutes, bool) or not math.isfinite(minutes) or minutes <= 0):
        raise ValueError('manual minutes must be positive and finite')
    m = episode_metrics(reports, episodes_root=episodes_root, unreadable_lines=unreadable_lines)
    factory = collect_factory_history() if factory is None else factory
    rows = m['rows']
    qa_rate = f"{100*m['qa_passed']/m['qa_known']:.1f}%" if m['qa_known'] else 'N/A'
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
           '<title>FrameFoundry scoreboard</title><style>',
           '*{box-sizing:border-box}body{margin:0;background:#fafafa;color:#222;font:15px/1.6 system-ui,sans-serif}'
           'main{max-width:1200px;margin:auto;padding:48px 24px}h1{font-size:32px;line-height:1.2;margin:0 0 12px}'
           'h2{font-size:23px;margin:0 0 16px}h3{font-size:18px}p{max-width:95ch}section{margin-top:40px;padding-top:28px;border-top:1px solid #ddd}'
           '.note{color:#555;font-size:13px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white}'
           'th,td{text-align:left;padding:12px 14px;border-bottom:1px solid #e5e5e5;white-space:nowrap}'
           'th{font-size:12px;color:#555}a{color:#245f85}svg{display:block;width:100%;max-width:780px;margin:24px 0}'
           'svg text{font:13px system-ui,sans-serif;fill:#555}.grid{stroke:#ddd;fill:none}.trend{stroke:#245f85;stroke-width:2;fill:none}.point{fill:#245f85}'
           '.cards{display:flex;gap:24px;flex-wrap:wrap}.card{flex:1;min-width:220px;padding:20px;background:white;border:1px solid #ddd}'
           '.value{font-size:38px;font-weight:600;color:#245f85}.eyebrow{color:#555;font-size:12px;letter-spacing:.08em;text-transform:uppercase}'
           '</style></head><body><main><p class="eyebrow">FrameFoundry · generation-time snapshot</p>',
           '<h1>Teaser quality &amp; factory scoreboard</h1>',
           _presenter(m,factory),
           '<details><summary>Detailed metrics, methods &amp; disclosures</summary>',
           '<p>Latest validated evaluation per episode. No live Mac data is fabricated; this file updates only when regenerated with local evaluation JSONL and episode artifacts.</p>',
           '<section><h2>Latest episodes</h2>',
           _table(['Episode', 'J', 'Q', 'D', 'Cost USD', 'Cost status', 'Known subtotal USD', 'Q/$',
                   'Wall time', 'G / gates', 'Actual render QA', 'Shipped (verified export)'], [
               (r['label'], *(_number(r['scores'][k]) for k in ('J', 'Q', 'D')),
                _number(r['usd'], 6), r['cost_status'], _number(r['known_usd'], 6), _number(r['qpd']),
                'N/A' if r['wall'] is None else f"{r['wall']:.2f}s",
                f"{r['scores']['G']} / {'PASS' if r['gates'] else 'FAIL'}",
                'N/A' if r['qa'] is None else ('PASS' if r['qa'] else 'FAIL'), 'YES' if r['shipped'] else 'NO') for r in rows]),
           f"<p>Latest episodes: {len(rows)}; valid evaluation lines: {m['valid_lines']}; invalid/corrupt lines: {m['rejected_lines']}.</p>"]
    if not rows:
        out.append('<p>No readable evaluation reports. Episode metrics are N/A.</p>')
    if input_unavailable:
        out.append('<p>Scoreboard input: unavailable; no totals certified.</p>')
    if baseline:
        out.append('<p>Baseline comparison: ' + _escape(baseline['status']) + '.</p>')
    out.extend(['<p class="note">J/Q/D come from evaluation scores. Q/$ is N/A for unknown or zero cost or missing Q. Wall time is pipeline timing.elapsed_seconds, not evaluation time or engineer labor. Actual QA is current qa-report.json, never inferred from G or first try.</p></section>',
                '<section><h2>Q trend</h2>', _trend(rows),
                '<p class="note">Latest validated scores in episode order, not retry history. Missing scores/episodes break the line; N/A is not zero.</p></section>',
                '<section><h2>Episode delivery metrics</h2><div class="cards">',
                '<div class="card">Render QA pass rate (latest episodes)<div class="value">' + qa_rate + '</div>',
                f"<p>{m['qa_passed']}/{m['qa_known']} known; {len(rows)-m['qa_known']} unknown excluded.</p></div>",
                f'<div class="card">Shipped episodes<div class="value">{m["shipped"]}</div><p>Verified current local exports, not social publications.</p></div></div>',
                '<p class="note">Render QA pass rate is not full unit/CI pass rate and does not count all retry attempts. Shipped requires actual QA PASS, eval G=1 and all gate checks passed, a nonempty finished.mp4, and its streaming SHA-256 matching report.render_hash. Missing/stale/unreadable media cannot certify shipment.</p>'])
    cost_label = ('estimated cohort cost' if m['estimated'] else 'cohort cost') if m['cost_complete'] else 'known cohort subtotal estimate/incomplete (not a complete bill)'
    out.append('<h3>AI cost per shipped episode</h3><p>' + _number(m['cost_per_shipped'], 6)
               + ' USD — ' + _escape(cost_label) + ' / shipped episodes.</p>')
    out.append(f"<p>Formula: {_number(m['cohort_subtotal'], 6)} USD / {m['shipped']} shipped. "
               f"Cost coverage: {m['cost_known']}/{len(rows)} known totals; {len(rows)-m['cost_known']} unknown totals. "
               f"Recorded known_usd subtotal: {_number(m['known_subtotal'], 6)} USD; {m['known_missing']} missing known subtotals.</p>")
    out.append('<p class="note">Cohort includes ALL latest per-episode cumulative model costs, including failed-run overhead. Histories/retries are not double-counted. For an unknown total, only recorded known_usd contributes to the incomplete subtotal; missing billed calls remain unknown. Local compute expense is excluded. Zero shipped denominator means N/A, never a fake zero-cost delivery.</p>')
    out.append('<h3>Manual editing effort avoided</h3>')
    if minutes is None:
        out.append('<p>not measured — no per-episode manual editing assumption supplied.</p>')
    else:
        hours = minutes * m['shipped'] / 60
        if not math.isfinite(hours):
            raise ValueError('manual effort estimate overflow')
        out.append(f'<p>{hours:.3f} hours gross estimated manual editing effort avoided — estimate, assumption: {minutes:g} min manual edit per teaser.</p>')
        out.append(f'<p>Formula: {minutes:g} min × {m["shipped"]} shipped episodes / 60 = {hours:.3f} hours.</p>')
    out.append('<p class="note">Assumes autonomous editing has zero hands-on editing. Excludes preparation, reviews, and debugging. This is an assumption-based gross estimate, not measured labor savings; never per WO and never subtracting machine wall time as engineer labor.</p></section>')
    out.extend([_factory_section(factory), '</details></main></body></html>'])
    return ''.join(out)


def export_html(path, document, *, episodes_root, input_paths=()):
    """Atomic export; resolve directory aliases, refuse symlink output files."""
    path = Path(path).absolute()
    try:
        if path.suffix.lower() != '.html':
            raise ValueError('HTML output must end in .html')
        if path.is_symlink():
            raise ValueError('HTML output symlinks are not allowed')
        resolved = path.resolve()
        path = resolved
        episode_root = Path(episodes_root).resolve()
        if resolved == episode_root or episode_root in resolved.parents:
            raise ValueError('HTML output cannot overwrite episode artifacts')
        for source in input_paths:
            source = Path(source)
            if resolved == source.resolve() or (path.exists() and source.exists() and os.path.samefile(path, source)):
                raise ValueError('HTML output cannot overwrite input files')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.scoreboard-', delete=False) as handle:
                temporary = handle.name
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
    except OSError:
        raise ValueError('HTML export failed; check output directory permissions/type') from None
    return path
