"""Offline paired-run separation, authorization and UI-compatible curve contracts."""
import json
from pathlib import Path
import pytest
from scripts.improve import run_experiment, reserve_experiment, snapshot_seed, comparable_pair
from pipelines.ffmpeg.contracts import read_json
from pipelines.ffmpeg.llm_provider import ProviderBudgetError
from pipelines.ffmpeg.memory_store import LocalJsonlStore
from tests.test_memory_loop import lesson


def test_seed_frozen_all_rows_and_pending_no_execution(tmp_path):
    seed=tmp_path/'shared';LocalJsonlStore(seed/'lessons.jsonl').put_many([lesson(str(i),str(i)) for i in range(8)])
    def no_run(*a,**k):pytest.fail('pending dispatched')
    curve=run_experiment(tmp_path,'family trip',experiment_id='pending',iterations=2,episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals',seed_root=seed,runner=no_run)
    assert curve['status']=='pending' and len(curve['arms']['memory_on'])==2
    frozen=tmp_path/'evals/learning-curves/pending/seed/lessons.jsonl'
    assert len(frozen.read_text().splitlines())==8
    LocalJsonlStore(seed/'lessons.jsonl').put_many([lesson('later','later')])
    assert len(frozen.read_text().splitlines())==8
    assert read_json(tmp_path/'evals/learning-curves/latest.json')==curve
    assert not (tmp_path/'episodes').exists()
    with pytest.raises(ValueError):run_experiment(tmp_path,'family trip',experiment_id='pending',iterations=2,episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals')


def test_runner_separates_every_arm_cache_memory_and_reports_failure(tmp_path,monkeypatch):
    calls=[];monkeypatch.setenv('FRAMEFOUNDRY_CACHE_DIR','original')
    def runner(footage,goal,**kw):
        import os
        calls.append((kw,os.environ['FRAMEFOUNDRY_CACHE_DIR']))
        if kw['memory']:
            LocalJsonlStore(Path(kw['memory_root'])/'lessons.jsonl').put_many([lesson('new','new')])
        else:
            assert kw['memory_root'] is None
        raise RuntimeError('deliberately unavailable')
    c=run_experiment(tmp_path,'family trip',experiment_id='separated',iterations=2,episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals',execute=True,runner=runner)
    assert len(calls)==4
    assert [k['memory'] for k,_ in calls]==[False,True,False,True]
    assert len({cache for _,cache in calls})==2
    assert len({str(k['evals_root']) for k,_ in calls})==2  # Judge cache as well as Planner.
    assert len({k['episode_id'] for k,_ in calls})==4
    assert all(k['reuse_history'] is False and k['provider'] is None and k['asr_provider']=='local' for k,_ in calls)
    assert all(r['status']=='failed' and r['J'] is None and r['Q'] is None and r['model_usd'] is None for rows in c['arms'].values() for r in rows)
    assert all(d['comparable'] is False and d['Q'] is None for d in c['deltas'])
    assert not (tmp_path/'evals/learning-curves/separated/memory_off').exists()
    import os
    assert os.environ['FRAMEFOUNDRY_CACHE_DIR']=='original'


def test_full_forecast_reservation_never_raises_global_cap(tmp_path,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.25')
    ids=['a','b','c','d']
    info=reserve_experiment(tmp_path,ids,authorized_total=200,prior_spend=74)
    assert info['project_forecast_usd']==75
    assert len(read_json(tmp_path/'.ai-project-spend.json')['episodes'])==4
    assert reserve_experiment(tmp_path,ids,authorized_total=200,prior_spend=0)['prior_spend_usd']==74
    with pytest.raises(ProviderBudgetError):reserve_experiment(tmp_path,['e'],authorized_total=75,prior_spend=74)
    with pytest.raises(ProviderBudgetError):reserve_experiment(tmp_path,['e'],authorized_total=float('nan'),prior_spend=74)
    with pytest.raises(ProviderBudgetError):reserve_experiment(tmp_path,['e'],authorized_total=1000,prior_spend=500)
    assert len(read_json(tmp_path/'.ai-project-spend.json')['episodes'])==4


def test_authorization_precedes_any_paid_dispatch(tmp_path):
    def no_run(*a,**k):pytest.fail('budget failure dispatched')
    c=run_experiment(tmp_path,'trip',experiment_id='blocked',iterations=2,episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals',execute=True,paid=True,authorized_total=74,prior_spend=74,runner=no_run,budget_root=tmp_path/'budget')
    assert c['status']=='pending' and all(r['status']=='unavailable' for rows in c['arms'].values() for r in rows)


def test_invalid_ids_recipes_and_comparability(tmp_path):
    common=dict(iterations=1,episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals')
    for name in ('../bad','UPPER','a'*41):
        with pytest.raises(ValueError):run_experiment(tmp_path,'trip',experiment_id=name,**common)
    with pytest.raises(ValueError):run_experiment(tmp_path,'trip',experiment_id='voice',recipe={'voiceover':True},execute=True,**common)
    assert not comparable_pair(dict(status='failed'),dict(status='passed'),{}, {})


def test_malformed_seed_remains_pending_and_never_dispatches(tmp_path):
    seed=tmp_path/'seed';seed.mkdir();(seed/'lessons.jsonl').write_text('{bad}\n')
    def no_run(*a,**k):pytest.fail('invalid baseline dispatched')
    c=run_experiment(tmp_path,'trip',experiment_id='bad-seed',iterations=1,episodes_root=tmp_path/'episodes',evals_root=tmp_path/'evals',seed_root=seed,execute=True,runner=no_run)
    assert c['status']=='pending' and 'baseline not silently replaced' in ' '.join(c['limitations'])
    assert read_json(tmp_path/'evals/learning-curves/latest.json')==c
