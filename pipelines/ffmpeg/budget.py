"""Conservative project AI allowances, including unknown failed-call reservations."""
from pathlib import Path
from decimal import Decimal,InvalidOperation
import fcntl,os
from .contracts import atomic_write_json,read_json
from .llm_provider import ProviderError

def reserve_project_episode(root:Path,episode_id:str):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.ai-project.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'.ai-project-spend.json'
        try:
            if path.exists():data=read_json(path)
            else:data=dict(initial_spend_usd=str(os.environ.get('FRAMEFOUNDRY_PROJECT_SPENT_USD','0')),episodes={})
            initial=Decimal(data['initial_spend_usd'])
            if not initial.is_finite() or initial<0:raise ValueError('Invalid initial spend')
            reservations=data['episodes']
            amounts=[Decimal(str(x)) for x in reservations.values()]
            if any(not x.is_finite() or x<0 for x in amounts):raise ValueError('Invalid reservations')
            if episode_id not in reservations:
                if initial+sum(amounts,Decimal(0))+Decimal(5)>500:raise ProviderError('Project AI allowance would exceed $500')
                reservations[episode_id]=5
                atomic_write_json(path,data)
        except ProviderError:raise
        except (ValueError,KeyError,TypeError,InvalidOperation):raise ProviderError('Invalid project allowance ledger') from None

def assert_can_spend(max_charge_usd,*,paid_enabled,episode_spent_usd,project_spent_usd):
    if not paid_enabled:raise ProviderError('Paid actions disabled')
    if max_charge_usd is None:raise ProviderError('Maximum charge unknown')
    values=[Decimal(str(x)) for x in [max_charge_usd,episode_spent_usd,project_spent_usd]]
    if any(not x.is_finite() or x<0 for x in values):raise ProviderError('Invalid cost')
    charge,ep,project=values
    if ep+charge>5 or project+charge>500:raise ProviderError('Budget exceeded')
