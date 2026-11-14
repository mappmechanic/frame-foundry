import pytest
import os

@pytest.fixture(autouse=True)
def isolate_model_artifacts(tmp_path,monkeypatch):
    if os.environ.get('FRAMEFOUNDRY_LIVE_TESTS')!='1':
        for key in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','ELEVENLABS_API_KEY']:monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('FRAMEFOUNDRY_EVALS_ROOT',str(tmp_path/'evals'))
    monkeypatch.setenv('FRAMEFOUNDRY_CACHE_DIR',str(tmp_path/'cache'))
    monkeypatch.setenv('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(tmp_path/'project-budget'))
