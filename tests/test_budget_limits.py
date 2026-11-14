"""Offline project/episode allowance tests; environment variables only, no network."""
import json

import pytest

from pipelines.ffmpeg.budget import (PROJECT_BUDGET_USD, assert_can_spend,
                                     reserve_project_episode, tts_usage_status)
from pipelines.ffmpeg.llm_provider import ProviderBudgetError, ProviderError

BUDGET_ENV = 'FRAMEFOUNDRY_EPISODE_API_BUDGET_USD'
SPENT_ENV = 'FRAMEFOUNDRY_PROJECT_SPENT_USD'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (BUDGET_ENV, SPENT_ENV, 'FRAMEFOUNDRY_EPISODE_TOKEN_BUDGET'):
        monkeypatch.delenv(name, raising=False)


def ledger(root):
    return json.loads((root / '.ai-project-spend.json').read_text())


def test_default_reservation_is_configured_quarter_dollar_not_five(tmp_path):
    reserve_project_episode(tmp_path, 'episode-1')
    data = ledger(tmp_path)
    assert data['episodes'] == {'episode-1': '0.25'}
    assert data['initial_spend_usd'] == '0'
    assert PROJECT_BUDGET_USD == 500


def test_reservation_is_idempotent_and_follows_configured_budget(tmp_path, monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, '1.5')
    reserve_project_episode(tmp_path, 'a')
    reserve_project_episode(tmp_path, 'a')
    monkeypatch.setenv(BUDGET_ENV, '.5')
    reserve_project_episode(tmp_path, 'b')
    assert ledger(tmp_path)['episodes'] == {'a': '1.5', 'b': '0.5'}


def test_project_cap_is_500_with_explicit_initial_spend(tmp_path, monkeypatch):
    monkeypatch.setenv(SPENT_ENV, '499.5')
    reserve_project_episode(tmp_path, 'fits')          # 499.5 + .25
    reserve_project_episode(tmp_path, 'fits-exactly')  # + .25 == 500
    with pytest.raises(ProviderBudgetError, match=r'\$500'):
        reserve_project_episode(tmp_path, 'over')
    assert 'over' not in ledger(tmp_path)['episodes']
    # A recorded ledger keeps its initial spend even if the environment changes later.
    monkeypatch.setenv(SPENT_ENV, '0')
    with pytest.raises(ProviderError):
        reserve_project_episode(tmp_path, 'still-over')


def test_two_thousand_default_episodes_fit_exactly_in_project_cap(tmp_path):
    for index in range(2000):
        reserve_project_episode(tmp_path, f'episode-{index}')
    with pytest.raises(ProviderError):
        reserve_project_episode(tmp_path, 'one-too-many')


def test_budgets_are_independent_across_roots(tmp_path, monkeypatch):
    monkeypatch.setenv(SPENT_ENV, '499.7')
    reserve_project_episode(tmp_path / 'one', 'e')
    with pytest.raises(ProviderError):
        reserve_project_episode(tmp_path / 'one', 'f')
    monkeypatch.setenv(SPENT_ENV, '0')
    reserve_project_episode(tmp_path / 'two', 'f')     # a different root has its own ledger
    assert list(ledger(tmp_path / 'two')['episodes']) == ['f']
    assert list(ledger(tmp_path / 'one')['episodes']) == ['e']


@pytest.mark.parametrize('value', ['NaN', 'inf', '-1', '0', '', 'abc', '5.01', '1e1'])
def test_invalid_or_nonfinite_episode_budget_env_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setenv(BUDGET_ENV, value)
    with pytest.raises(ProviderError):
        reserve_project_episode(tmp_path, 'e')
    with pytest.raises(ProviderError):
        assert_can_spend(.01, paid_enabled=True, episode_spent_usd=0, project_spent_usd=0)
    assert not (tmp_path / '.ai-project-spend.json').exists()


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-5', 'abc', ''])
def test_invalid_initial_project_spend_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setenv(SPENT_ENV, value)
    with pytest.raises(ProviderError, match='Invalid project allowance ledger'):
        reserve_project_episode(tmp_path, 'e')
    assert not (tmp_path / '.ai-project-spend.json').exists()


def test_corrupt_project_ledger_fails_closed(tmp_path):
    reserve_project_episode(tmp_path, 'a')
    path = tmp_path / '.ai-project-spend.json'
    for text in ('{bad', '{"initial_spend_usd":"0","episodes":{"a":"-1"}}',
                 '{"initial_spend_usd":"NaN","episodes":{}}', '{"episodes":{}}'):
        path.write_text(text)
        with pytest.raises(ProviderError):
            reserve_project_episode(tmp_path, 'b')
        assert path.read_text() == text


def test_legacy_five_dollar_reservations_still_count_in_full(tmp_path):
    (tmp_path / '.ai-project-spend.json').write_text(json.dumps(
        {'initial_spend_usd': '0', 'episodes': {f'old-{i}': 5 for i in range(100)}}))
    with pytest.raises(ProviderBudgetError):
        reserve_project_episode(tmp_path, 'new')


def test_assert_can_spend_uses_configured_episode_budget(monkeypatch):
    ok = dict(paid_enabled=True, project_spent_usd=0)
    assert_can_spend(.05, episode_spent_usd=.2, **ok)
    with pytest.raises(ProviderBudgetError):
        assert_can_spend(.06, episode_spent_usd=.2, **ok)
    monkeypatch.setenv(BUDGET_ENV, '1')
    assert_can_spend(.06, episode_spent_usd=.2, **ok)
    with pytest.raises(ProviderBudgetError):
        assert_can_spend(.06, episode_spent_usd=0, paid_enabled=True, project_spent_usd=499.95)
    for bad in (dict(paid_enabled=False), dict(max_charge_usd=None), dict(max_charge_usd='NaN'),
                dict(max_charge_usd=-1)):
        kwargs = dict(max_charge_usd=.01, episode_spent_usd=0, **ok)
        kwargs.update(bad)
        with pytest.raises(ProviderError):
            assert_can_spend(**kwargs)


def test_tts_accounting_is_not_used_until_a_ledger_exists(tmp_path):
    assert tts_usage_status(tmp_path) == dict(status='not_used', implemented=True,
                                              shared_episode_budget=True, known_usd=0.0,
                                              unpriced_calls=0)
    (tmp_path / 'tts-usage.json').write_text(json.dumps({'calls': [
        {'status': 'success', 'cost_usd': 0.02}, {'status': 'blocked', 'cost_usd': None},
        {'status': 'failed', 'cost_usd': None, 'reserved_usd': 0.01}]}))
    status = tts_usage_status(tmp_path)
    assert (status['status'], status['known_usd'], status['unpriced_calls']) == ('recorded', 0.03, 0)
    (tmp_path / 'tts-usage.json').write_text(json.dumps({'calls': [{'status': 'failed'}]}))
    assert tts_usage_status(tmp_path)['status'] == 'pending'
    (tmp_path / 'tts-usage.json').write_text('broken')
    with pytest.raises(ProviderError):
        tts_usage_status(tmp_path)
