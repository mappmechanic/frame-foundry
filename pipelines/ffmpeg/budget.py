"""Conservative project AI allowances, including unknown failed-call reservations.

Each episode reserves the *configured* episode API budget (default $0.25, env
``FRAMEFOUNDRY_EPISODE_API_BUDGET_USD``) against the $500 project ledger. Prior
project spend is explicit: ``FRAMEFOUNDRY_PROJECT_SPENT_USD`` (default 0) seeds a new
ledger and is validated; an existing ledger keeps its recorded initial spend.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import fcntl, os
from .contracts import atomic_write_json, read_json
from .llm_provider import ProviderBudgetError, ProviderError, _tts_summary, episode_budget_usd

PROJECT_BUDGET_USD = Decimal(500)


def project_limit_usd():
    try:limit=Decimal(os.environ.get('FRAMEFOUNDRY_PROJECT_API_BUDGET_USD',str(PROJECT_BUDGET_USD)))
    except Exception:raise ProviderBudgetError('Invalid project credit limit')
    if not limit.is_finite() or limit<=0 or limit>PROJECT_BUDGET_USD:raise ProviderBudgetError('Project credit limit must be positive and cannot exceed original cap')
    return limit

def reserve_project_episode(root: Path, episode_id: str):
    """Reserve one episode's configured allowance once; repeat calls are idempotent."""
    episode_budget = episode_budget_usd()     # invalid/non-finite env fails closed
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.ai-project.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / '.ai-project-spend.json'
        try:
            if path.exists():
                data = read_json(path)
            else:
                data = dict(initial_spend_usd=str(os.environ.get('FRAMEFOUNDRY_PROJECT_SPENT_USD', '0')),
                            episodes={})
            initial = Decimal(data['initial_spend_usd'])
            if not initial.is_finite() or initial < 0:
                raise ValueError('Invalid initial spend')
            reservations = data['episodes']
            amounts = [Decimal(str(x)) for x in reservations.values()]
            if any(not x.is_finite() or x < 0 for x in amounts):
                raise ValueError('Invalid reservations')
            if episode_id not in reservations:
                if initial + sum(amounts, Decimal(0)) + episode_budget > project_limit_usd():
                    raise ProviderBudgetError('Project AI allowance would exceed $500')
                reservations[episode_id] = str(episode_budget)
                atomic_write_json(path, data)
        except ProviderError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError, InvalidOperation):
            raise ProviderError('Invalid project allowance ledger') from None


def assert_can_spend(max_charge_usd, *, paid_enabled, episode_spent_usd, project_spent_usd):
    if not paid_enabled:
        raise ProviderError('Paid actions disabled')
    if max_charge_usd is None:
        raise ProviderError('Maximum charge unknown')
    try:
        values = [Decimal(str(x)) for x in [max_charge_usd, episode_spent_usd, project_spent_usd]]
    except InvalidOperation:
        raise ProviderError('Invalid cost') from None
    if any(not x.is_finite() or x < 0 for x in values):
        raise ProviderError('Invalid cost')
    charge, episode, project = values
    if episode + charge > episode_budget_usd() or project + charge > project_limit_usd():
        raise ProviderBudgetError('Budget exceeded')


def tts_usage_status(episode_dir: Path) -> dict:
    """TTS accounting for the shared episode budget.

    Optional ElevenLabs is implemented; absence of a ledger means ``not_used``.
    When a ledger exists, known
    charged USD is reported (and is counted by ``LLMProvider`` against the same budget);
    entries without a price are counted as ``unpriced_calls``.
    """
    if not (Path(episode_dir) / 'tts-usage.json').exists():
        return dict(status='not_used', implemented=True, shared_episode_budget=True,
                    known_usd=0.0, unpriced_calls=0)
    known, unpriced = _tts_summary(Path(episode_dir))
    return dict(status='pending' if unpriced else 'recorded', implemented=True,
                shared_episode_budget=True, known_usd=float(known), unpriced_calls=unpriced)
