"""Synthetic strict reports and local artifacts only; no network or live Mac claims."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.ffmpeg import eval_cli, factory_history
from pipelines.ffmpeg.evaluator import validate_eval_report
from pipelines.ffmpeg.scoreboard_html import episode_metrics, export_html, render_html


def run(args, capsys):
    code = eval_cli.main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(item) + '\n' for item in items))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(factory_history.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(stdout='', returncode=1))


@pytest.fixture
def report():
    def make(eid='az-0500', *, q=.5, usd=2, passed=True, created='2026-01-01T00:00:00Z'):
        r = dict(schema_version='1.0', evaluation_id='a'*32, episode_id=eid,
                 goal='synthetic test only', created_at=created, render_hash=hashlib.sha256(b'export').hexdigest(),
                 footage_hash='b'*64, gates=dict(passed=passed, checks=[dict(name='test', status='pass' if passed else 'fail', reason=None)], version='mock', thresholds={}),
                 scores=dict(G=int(passed), D=.8, J=None, H=None, Q=q if passed else 0,
                             weights=dict(J=.625, D=.375, H=0), Q_status='computed' if passed else 'forced_zero_gate_failed',
                             D_components={}, H_detail=None, method_version='mock'),
                 judge=dict(status='unavailable', scores=None, evidence=None, weakest_shot=None,
                            fix_instruction=None, verdict=None, unsupported_goal=None, limitations=[], cache_hit=False,
                            reason='synthetic mock', provider=None, model=None, prompt_version='mock'),
                 model_cost=dict(usd=usd, status='unknown' if usd is None else 'measured', known_usd=usd or 0,
                                 llm_usd=usd, tts_usd=0, quality_per_dollar=q/usd if q is not None and usd else None, notes=[]),
                 tokens=dict(input=0, output=0, complete=True), timing=dict(elapsed_seconds=12, footage_seconds=60, evaluation_seconds=99),
                 provenance=dict(branch=None, commit=None, pr=None, fixture_mode=True, transcription_status='fixture', evaluator_version='mock'),
                 first_try_passed=passed, limitations=[], budget=dict(status='unavailable', api_limit_usd=None, token_limit=None,
                    accounted_llm_usd=None, accounted_tokens=None, unknown_usage_reservations=None, blocked_calls=None, reason='mock'))
        validate_eval_report(r)
        return r
    return make


def artifact(root, eid, *, passed=True, data=b'export'):
    d = root / eid
    d.mkdir(parents=True, exist_ok=True)
    (d / 'qa-report.json').write_text(json.dumps(dict(episode_id=eid, passed=passed)))
    (d / 'finished.mp4').write_bytes(data)
    return d


def test_latest_cost_failed_overhead_qa_and_shipped(report, tmp_path):
    root = tmp_path / 'episodes'
    artifact(root, 'az-0500')
    artifact(root, 'az-0550', passed=False)
    rs = [report(usd=99), report(usd=2, created='2026-01-02T00:00:00Z'),
          report('az-0550', usd=3, passed=False), report('az-0600', usd=4)]
    m = episode_metrics(rs, episodes_root=root)
    assert len(m['rows']) == 3 and m['valid_lines'] == 4
    assert (m['qa_passed'], m['qa_known'], m['shipped']) == (1, 2, 1)
    assert m['cohort_subtotal'] == 9 and m['cost_per_shipped'] == 9
    out = render_html(rs, episodes_root=root, assume_manual_minutes_per_episode=12.5)
    assert '50.0%' in out and '1/2 known; 1 unknown excluded' in out
    assert '9.000000 USD' in out and '12.5 min manual edit per teaser' in out
    assert '12.5 min × 1 shipped episodes / 60 = 0.208 hours' in out
    assert 'Render QA pass rate (latest episodes)' in out
    assert '<td>12.00s</td>' in out and '99.00s' not in out
    assert '7</td>' in out and 'In Review' in out


def test_incomplete_cost_with_failed_overhead_zero_denominator_and_default_hours(report, tmp_path):
    a = report(usd=None)
    a['model_cost']['known_usd'] = .04
    m = episode_metrics([a, report('az-0550', usd=2, passed=False)], episodes_root=tmp_path)
    assert not m['cost_complete'] and m['cost_known'] == 1
    assert m['cohort_subtotal'] == pytest.approx(2.04)
    assert m['cost_per_shipped'] is None
    out = render_html([a], episodes_root=tmp_path)
    assert 'unknown/incomplete' in out and 'not a complete bill' in out and 'not measured' in out
    assert 'N/A USD' in out and '0/0 known; 1 unknown excluded' in out


@pytest.mark.parametrize('kind', ['qa-fail', 'qa-missing', 'qa-corrupt', 'qa-nonboolean', 'qa-wrong-identity',
                                  'gate-fail', 'hash-mismatch', 'empty-media', 'missing-media', 'symlink-media', 'symlink-qa'])
def test_shipped_requires_current_complete_evidence(report, tmp_path, kind):
    d = artifact(tmp_path, 'az-0500')
    r = report()
    if kind == 'qa-fail':
        artifact(tmp_path, 'az-0500', passed=False)
    elif kind.startswith('qa-'):
        path = d / 'qa-report.json'
        if kind == 'qa-missing':
            path.unlink()
        else:
            path.write_text({'qa-corrupt':'{', 'qa-nonboolean':'{"episode_id":"az-0500","passed":1}',
                             'qa-wrong-identity':'{"episode_id":"other","passed":true}'}[kind])
    elif kind == 'gate-fail':
        r = report(passed=False)
    elif kind == 'hash-mismatch':
        (d / 'finished.mp4').write_bytes(b'different')
    elif kind == 'empty-media':
        (d / 'finished.mp4').write_bytes(b'')
    elif kind == 'missing-media':
        (d / 'finished.mp4').unlink()
    elif kind.startswith('symlink-'):
        filename = 'finished.mp4' if kind == 'symlink-media' else 'qa-report.json'
        path = d / filename
        outside = tmp_path / ('outside-' + filename)
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    assert episode_metrics([r], episodes_root=tmp_path)['shipped'] == 0


def test_qa_not_inferred_from_g_and_hash_reads_streamed(report, tmp_path, monkeypatch):
    artifact(tmp_path, 'az-0500')
    r = report(passed=False)
    m = episode_metrics([r], episodes_root=tmp_path)
    assert m['qa_passed'] == 1 and m['shipped'] == 0
    def no_read_bytes(*args):
        raise AssertionError('media must be streamed')
    monkeypatch.setattr(Path, 'read_bytes', no_read_bytes)
    assert episode_metrics([report()], episodes_root=tmp_path)['shipped'] == 1


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_xss_all_dynamic_text_escaped_and_no_pageview_dependencies(report, tmp_path):
    factory = factory_history.collect_factory_history(tmp_path)
    evil = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    factory['notes'].append(evil)
    factory['pull_request']['reason'] = evil
    factory['work_orders'][0]['coverage_basis'] = evil
    out = render_html([report(evil)], episodes_root=tmp_path, factory=factory)
    tags = Tags()
    tags.feed(out)
    assert evil not in out and '&lt;script&gt;' in out and '[unsafe-id]' in out
    assert not any(tag in ('script', 'img', 'link', 'iframe') for tag, attrs in tags.tags)
    assert not any('src' in attrs or any(k.startswith('on') for k in attrs) for tag, attrs in tags.tags)
    assert out.count('<html ') == 1 and '<svg ' in out and 'Content-Security-Policy' in out


@pytest.mark.parametrize('eid', ['../outside', '/tmp/outside', 'UPPER'])
def test_path_traversal_never_reads_artifact(report, tmp_path, monkeypatch, eid):
    monkeypatch.setattr('pipelines.ffmpeg.scoreboard_html._qa_pass', lambda *a: pytest.fail('unsafe read'))
    out = render_html([report(eid)], episodes_root=tmp_path)
    assert '[unsafe-id]' in out and eid not in out


def test_episode_symlink_guard(report, tmp_path):
    outside = tmp_path / 'outside'
    artifact(outside, 'az-0500')
    root = tmp_path / 'episodes'
    root.mkdir()
    (root / 'az-0500').symlink_to(outside / 'az-0500', target_is_directory=True)
    m = episode_metrics([report()], episodes_root=root)
    assert m['rows'][0]['label'] == '[unsafe-id]' and m['qa_known'] == 0 and m['shipped'] == 0


def test_q_trend_gaps_not_fabricated_or_connected(report, tmp_path):
    out = render_html([report(q=.3), report('az-0600', q=.9)], episodes_root=tmp_path)
    assert out.count('class="point"') == 2 and 'class="trend"' not in out
    assert '<td>az-0550</td><td>N/A</td>' in out
    missing = report('az-0550', q=None)
    assert render_html([missing], episodes_root=tmp_path).count('class="point"') == 0
    out = render_html([report(q=.3), report('az-0550', q=.6), report('az-0600', q=.9), report('az-0700', q=1)], episodes_root=tmp_path)
    assert out.count('class="trend"') == 3 and 'az-0700' in out


def test_invalid_latest_and_corrupt_jsonl_report_counts(report, tmp_path, capsys):
    evals = tmp_path / 'evals'
    invalid = report(created='2027-01-01T00:00:00Z', usd=99)
    invalid['scores']['Q'] = 'SECRET'
    write_jsonl(evals / 'scoreboard.jsonl', [report(), invalid, {}])
    with (evals / 'scoreboard.jsonl').open('a') as f:
        f.write('{bad\n{"v":NaN}\n')
    out = tmp_path / 'nested' / 'out.html'
    code, stdout, err = run(['scoreboard', '--html', str(out), '--evals-root', str(evals)], capsys)
    assert (code, stdout, err) == (0, '', '')
    text = out.read_text()
    assert 'valid evaluation lines: 1; invalid/corrupt lines: 4' in text
    assert 'SECRET' not in text and '99.000000' not in text


@pytest.mark.parametrize('value', ['NaN', 'inf', '-inf', '-1', '0', 'wat'])
def test_cli_rejects_invalid_assumption(tmp_path, capsys, value):
    code, _, err = run(['scoreboard', '--html', str(tmp_path/'o.html'), '--assume-manual-minutes-per-episode', value], capsys)
    assert code == 2 and err and not (tmp_path/'o.html').exists()


@pytest.mark.parametrize('flags', [['--table'], ['--lines'], ['--json']])
def test_incompatible_html_flags(tmp_path, capsys, flags):
    code, _, err = run(['scoreboard', '--html', str(tmp_path/'o.html'), *flags], capsys)
    assert code == 2 and 'cannot be combined' in err


def test_assumption_requires_html_and_fraction_works(tmp_path, capsys):
    code, _, err = run(['scoreboard', '--assume-manual-minutes-per-episode', '1.5'], capsys)
    assert code == 2 and 'requires --html' in err
    path = tmp_path/'o.html'
    code, stdout, err = run(['scoreboard', '--html', str(path), '--evals-root', str(tmp_path/'absent'), '--assume-manual-minutes-per-episode', '1.5'], capsys)
    assert (code, stdout, err) == (0, '', '')
    assert '1.5 min manual edit per teaser' in path.read_text()


def test_source_file_symlinks_refused_before_read(tmp_path, capsys, monkeypatch):
    input_file = tmp_path / 'input'
    input_file.write_text('SECRET')
    (tmp_path / 'scoreboard.jsonl').symlink_to(input_file)
    monkeypatch.setattr(eval_cli.tools, 'load_reports', lambda *a: pytest.fail('symlink source read'))
    code, _, err = run(['scoreboard', '--html', str(tmp_path/'o.html'), '--evals-root', str(tmp_path)], capsys)
    assert code == 2 and 'symlinks' in err and 'SECRET' not in err


def test_output_protects_artifacts_inputs_symlinks_and_mkdir(tmp_path):
    episodes = tmp_path/'episodes'
    path = tmp_path/'new'/'out.html'
    export_html(path, '<html></html>', episodes_root=episodes)
    assert path.read_text() == '<html></html>'
    with pytest.raises(ValueError, match='artifacts'):
        export_html(episodes/'az-0500'/'out.html', 'bad', episodes_root=episodes)
    with pytest.raises(ValueError, match='input'):
        export_html(path, 'bad', episodes_root=episodes, input_paths=[path])
    alias = tmp_path/'alias.html'
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match='input'):
        export_html(alias, 'bad', episodes_root=episodes, input_paths=[path])
    symlink = tmp_path/'symlink.html'
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match='symlinks'):
        export_html(symlink, 'bad', episodes_root=episodes)
    folder = tmp_path/'folder'
    folder.symlink_to(path.parent, target_is_directory=True)
    exported=export_html(folder/'o.html', 'resolved parent', episodes_root=episodes)
    assert exported==path.parent/'o.html' and exported.read_text()=='resolved parent'
    alias_dir=tmp_path/'artifact-alias'
    alias_dir.symlink_to(episodes, target_is_directory=True)
    with pytest.raises(ValueError, match='artifacts'):
        export_html(alias_dir/'az-0500'/'out.html', 'bad', episodes_root=episodes)
    with pytest.raises(ValueError, match='.html'):
        export_html(tmp_path/'scoreboard.jsonl', 'bad', episodes_root=episodes)
    assert path.read_text() == '<html></html>'


def test_atomic_failed_export_preserves_previous_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path/'out.html'
    path.write_text('previous')
    monkeypatch.setattr('pipelines.ffmpeg.scoreboard_html.os.replace', lambda *a: (_ for _ in ()).throw(OSError('SECRET')))
    with pytest.raises(ValueError, match='export failed'):
        export_html(path, 'new', episodes_root=tmp_path/'episodes')
    assert path.read_text() == 'previous'
    assert not list(tmp_path.glob('.scoreboard-*'))


def test_unreadable_scoreboard_html_exits_unavailable(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(eval_cli.tools, 'load_reports', lambda *a: (_ for _ in ()).throw(PermissionError('SECRET')))
    path = tmp_path/'out.html'
    code, stdout, err = run(['scoreboard', '--html', str(path)], capsys)
    assert (code, stdout, err) == (3, '', '')
    assert 'Scoreboard input: unavailable' in path.read_text() and 'SECRET' not in path.read_text()


def test_html_baseline_preserves_regression_exit(report, tmp_path, capsys):
    write_jsonl(tmp_path/'scoreboard.jsonl', [report(q=.2)])
    write_jsonl(tmp_path/'baseline.jsonl', [report(q=.9)])
    path = tmp_path/'out.html'
    code, stdout, err = run(['scoreboard', '--html', str(path), '--evals-root', str(tmp_path), '--baseline', str(tmp_path/'baseline.jsonl')], capsys)
    assert (code, stdout, err) == (1, '', '') and 'Baseline comparison: regression' in path.read_text()


def test_zero_and_estimated_cost_and_missing_scores(report, tmp_path):
    zero = report(usd=0)
    zero['scores']['D'] = None
    zero['timing']['elapsed_seconds'] = None
    estimated = report('az-0550', usd=.02)
    estimated['model_cost']['status'] = 'estimated'
    m = episode_metrics([zero, estimated], episodes_root=tmp_path)
    assert m['rows'][0]['qpd'] is None and m['rows'][1]['qpd'] == 25
    assert m['estimated'] and m['cost_complete']
    out = render_html([zero, estimated], episodes_root=tmp_path)
    assert 'estimated cohort cost' in out and '<td>0.000000</td>' in out
    assert '<td>N/A</td>' in out and '<td>25.000</td>' in out


def test_shipped_rejects_inconsistent_gate_checks(report, tmp_path):
    artifact(tmp_path, 'az-0500')
    r = report()
    r['gates']['checks'][0]['status'] = 'fail'
    assert episode_metrics([r], episodes_root=tmp_path)['shipped'] == 0


def test_baseline_source_symlink_refused(tmp_path, capsys):
    source = tmp_path/'source.jsonl'
    source.write_text('SECRET')
    link = tmp_path/'baseline.jsonl'
    link.symlink_to(source)
    code, _, err = run(['scoreboard', '--html', str(tmp_path/'out.html'), '--baseline', str(link)], capsys)
    assert code == 2 and 'symlinks' in err and 'SECRET' not in err


def test_output_cli_failure_clear_no_file(tmp_path, capsys):
    path = tmp_path/'episodes'/'az-0500'/'out.html'
    code, _, err = run(['scoreboard', '--html', str(path), '--episodes-root', str(tmp_path/'episodes')], capsys)
    assert code == 2 and 'episode artifacts' in err and not path.exists()


def test_html_never_initializes_model_provider(report, tmp_path, monkeypatch, capsys):
    from pipelines.ffmpeg.llm_provider import LLMProvider
    monkeypatch.setattr(LLMProvider, 'from_env', lambda *a, **k: pytest.fail('model provider initialized'))
    write_jsonl(tmp_path/'scoreboard.jsonl', [report()])
    code, stdout, err = run(['scoreboard', '--html', str(tmp_path/'out.html'), '--evals-root', str(tmp_path)], capsys)
    assert (code, stdout, err) == (0, '', '')


def test_presenter_is_first_with_scoped_numbers_and_collapsed_audit(report,tmp_path):
    artifact(tmp_path,'az-0700')
    out=render_html([report('az-0700',q=.825,usd=.038)],episodes_root=tmp_path)
    assert out.index('Presenter view')<out.index('<details>')<out.index('Latest episodes')
    assert '<summary>Detailed metrics, methods &amp; disclosures</summary>' in out
    assert '<details open' not in out
    assert '0.825' in out and '0.0380 USD' in out and '100.0%' in out
    assert 'Ticket → clip' in out and 'No recorded ticket-to-Episode delivery mapping' in out
    assert 'Pipeline → clip' in out and 'not completion' in out
    assert 'Episodes shipped' in out and 'verified local exports' in out


def test_resolved_parent_alias_cannot_overwrite_input(tmp_path):
    real=tmp_path/'real';real.mkdir();source=real/'x.html';source.write_text('keep')
    alias=tmp_path/'tmp';alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError,match='input'):
        export_html(alias/'x.html','bad',episodes_root=tmp_path/'episodes',input_paths=[source])
    assert source.read_text()=='keep'


def test_presenter_includes_three_latest_mac_checks_from_local_reports(report,tmp_path):
    # Synthetic report fixtures test presentation only; never live Mac verification.
    records=[report('az-0700',q=.825,usd=.038),report('az-0810',q=.756,usd=.035),report('az-0835',q=.808,usd=.035)]
    out=render_html(records,episodes_root=tmp_path)
    presenter=out.split('<details>')[0]
    assert all(eid in presenter for eid in ('az-0700','az-0810','az-0835'))
    assert all(q in presenter for q in ('0.825','0.756','0.808'))
    assert 'az-0835' in presenter and out.count('class="point"')==3
    missing=render_html(records[:1],episodes_root=tmp_path)
    assert '<td>az-0810</td><td>N/A</td>' in missing
    assert '<td>az-0835</td><td>N/A</td>' in missing
    assert '0.808' not in missing


def test_demo_presenter_jq_quality_cost_first_try_and_timing_are_distinct(report,tmp_path):
    first=report('az-0700',q=.825,usd=.038);last=report('az-0900',q=.808,usd=.0235)
    retry=report('az-0810',q=.756,usd=.035);retry['first_try_passed']=False
    missing=report('az-0835',q=.808,usd=None);missing['first_try_passed']=None
    for r in (first,last,retry,missing):artifact(tmp_path,r['episode_id'])
    records=[first,retry,missing,last]
    m=episode_metrics(records,episodes_root=tmp_path)
    assert (m['first_passed'],m['first_known'])==(2,3)
    assert m['pipeline_median']==12
    out=render_html(records,episodes_root=tmp_path);presenter=out.split('<details>')[0]
    assert '66.7%' in presenter and '34.4' in presenter
    assert 'Ticket → clip' in presenter and 'No recorded ticket-to-Episode delivery mapping' in presenter
    assert 'Pipeline → clip' in presenter and '12.0 s' in presenter
    assert 'known subtotal · incomplete' in presenter
    assert 'Judge J / Q trend' in presenter and 'az-0900' in presenter
    assert 'median first title coverage' not in presenter
    assert '0.808' in presenter and '0.0235' not in presenter  # cost card is cohort cost, not silently latest cost.


def test_demo_missing_latest_quality_is_not_hidden_by_previous_success(report,tmp_path):
    older=report('az-0835',q=.808,usd=.0235)
    latest=report('az-0900',q=None,usd=0)
    out=render_html([older,latest],episodes_root=tmp_path).split('<details>')[0]
    assert 'Latest J / Q' in out and 'N/A / N/A' in out
    assert 'Quality per $' in out and 'az-0900' in out


def test_demo_jq_plot_missing_scores_break_series_independently(report,tmp_path):
    from pipelines.ffmpeg.scoreboard_html import _judge_trend
    records=[report('az-0700',q=.825),report('az-0810',q=None),report('az-0835',q=.808),report('az-0900',q=.808)]
    rows=episode_metrics(records,episodes_root=tmp_path)['rows']
    out=_judge_trend(rows)
    assert out.count('class="jq-line Q"')==1 and 'class="jq-line J"' not in out
    assert out.count('class="jq-point Q"')==3
