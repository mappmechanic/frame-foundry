"""Offline metadata tests: subprocess responses are mocks, not live claims."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from pipelines.ffmpeg import factory_history as history


def response(stdout='', returncode=0):
    return SimpleNamespace(stdout=stdout, returncode=returncode)


def test_scopes_numeric_rest_count_and_frozen_title_coverage(monkeypatch, tmp_path):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == 'git':
            return response('42\n')
        return response(json.dumps(dict(number=1, commits=47, state='open',
                                        created_at='2026-11-14T07:10:12Z', title='changed later')))
    monkeypatch.setattr(history.subprocess, 'run', run)
    result = history.collect_factory_history(tmp_path)
    assert result['git']['commits'] == 42 and result['git']['scope'] == 'local main..HEAD'
    assert result['pull_request']['commits'] == 47
    assert result['pull_request']['source'] == 'GitHub REST at generation'
    assert len(result['work_orders']) == 7
    assert all(wo['status'] == 'In Review' for wo in result['work_orders'])
    assert [wo['first_coverage_at'] for wo in result['work_orders'][:6]] == ['2026-11-14T07:10:12Z'] * 6
    seven = result['work_orders'][6]
    assert seven['first_coverage_at'] == '2026-11-14T09:17:31Z'
    assert seven['coverage_minutes'] == pytest.approx(71.9486841167)
    assert all(wo['coverage_minutes'] > 0 for wo in result['work_orders'])
    assert calls[1][0] == ['gh', 'api', 'repos/mappmechanic/frame-foundry/pulls/1']
    assert all(call[1]['timeout'] == 5 for call in calls)


@pytest.mark.parametrize('failure', ['missing', 'auth', 'timeout', 'json', 'array', 'boolean-count', 'negative-count', 'missing-field', 'nan'])
def test_gh_fallback_never_invents_zero(monkeypatch, tmp_path, failure):
    def run(args, **kwargs):
        if args[0] == 'git':
            return response('9')
        if failure == 'missing':
            raise FileNotFoundError()
        if failure == 'auth':
            return response('PRIVATE TOKEN', 1)
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(args, 5)
        texts = {'json': '{bad', 'array': '[]', 'boolean-count': '{"commits":true}',
                 'negative-count': '{"commits":-1}', 'missing-field': '{"commits":2}',
                 'nan': '{"commits":NaN}'}
        return response(texts[failure])
    monkeypatch.setattr(history.subprocess, 'run', run)
    result = history.collect_factory_history(tmp_path)
    pr = result['pull_request']
    assert pr['commits'] is None and pr['state'] is None
    assert pr['created_at'] == '2026-11-14T07:10:12Z'
    assert pr['source'] == 'audited snapshot' and pr['observed_at'] == result['fetched_at']
    assert pr['reason'] and 'N/A' in pr['reason'] and 'PRIVATE' not in pr['reason']
    assert result['git']['commits'] == 9


@pytest.mark.parametrize('stdout,code', [('garbage', 0), ('-1', 0), ('', 1), ('true', 0)])
def test_git_unavailable_is_not_zero(monkeypatch, tmp_path, stdout, code):
    monkeypatch.setattr(history.subprocess, 'run', lambda args, **kw: response(stdout, code))
    result = history.collect_factory_history(tmp_path)
    assert result['git']['commits'] is None and result['git']['reason']


def test_verified_zero_and_merged_state(monkeypatch, tmp_path):
    monkeypatch.setattr(history.subprocess, 'run', lambda args, **kw:
                        response('0') if args[0] == 'git' else response(json.dumps(dict(
                            number=1, commits=0, state='closed', merged_at='2026-11-15T01:00:00Z',
                            created_at='2026-11-14T07:10:12Z'))))
    result = history.collect_factory_history(tmp_path)
    assert result['git']['commits'] == 0 and result['pull_request']['commits'] == 0
    assert result['pull_request']['state'] == 'merged'


def test_negative_or_naive_coverage_rejected():
    with pytest.raises(ValueError):
        history.coverage_minutes('2026-11-14T08:05:34Z', '2026-11-14T07:10:12Z')
    with pytest.raises(ValueError):
        history.coverage_minutes('2026-11-14T06:00:00', '2026-11-14T07:00:00Z')
