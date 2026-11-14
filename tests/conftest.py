import pytest

@pytest.fixture(autouse=True)
def isolate_model_artifacts(tmp_path,monkeypatch):
    monkeypatch.setenv('FRAMEFOUNDRY_EVALS_ROOT',str(tmp_path/'evals'))
    monkeypatch.setenv('FRAMEFOUNDRY_CACHE_DIR',str(tmp_path/'cache'))
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(tmp_path/'project-budget'))
