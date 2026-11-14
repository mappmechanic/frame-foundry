"""Pin the unsupported OTIO runtime out; keep demo setup on Python 3.13."""
from pathlib import Path
import re
import subprocess
import sys

from packaging.specifiers import SpecifierSet
import pytest

from pipelines.ffmpeg import cli
from scripts import arizona_demo

ROOT = Path(__file__).resolve().parents[1]


def test_package_range_and_demo_version_agree():
    metadata = (ROOT / 'pyproject.toml').read_text()
    declared = re.search(r'^requires-python = "([^"]+)"', metadata, re.M).group(1)
    assert declared == '>=3.10,<3.14'
    supported = SpecifierSet(declared)
    for version in ('3.10', '3.11', '3.12', '3.13.17'):
        assert version in supported
    for version in ('3.9', '3.14.6', '4.0'):
        assert version not in supported
    assert (ROOT / '.python-version').read_text().strip() == '3.13'


@pytest.mark.parametrize('version', [(3, 10, 0), (3, 11, 0), (3, 12, 0), (3, 13, 17)])
def test_cli_dispatch_accepts_declared_range(monkeypatch, version):
    seen = []
    monkeypatch.setattr(sys, 'version_info', version)
    monkeypatch.setattr(cli, '_run_main', lambda args: seen.append(args) or 0)
    assert cli.main(['run', '--help']) == 0
    assert seen == [['--help']]


@pytest.mark.parametrize('version', [(3, 9, 0), (3, 14, 6), (4, 0, 0)])
def test_cli_and_demo_reject_unsupported_runtime_before_work(monkeypatch, capsys, version):
    monkeypatch.setattr(sys, 'version_info', version)
    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported Python must stop before dispatch or download')
    monkeypatch.setattr(cli, '_run_main', unexpected)
    monkeypatch.setattr(arizona_demo, 'fetch', unexpected)
    assert cli.main(['run']) == 2
    with pytest.raises(SystemExit) as error:
        arizona_demo.main(['--fetch-only'])
    assert error.value.code == 2
    assert 'uv venv -p 3.13' in capsys.readouterr().err


def test_setup_prefers_demo_uv_path_and_pins_python(tmp_path):
    home = tmp_path / 'demo-home'
    executable = home / '.local' / 'bin' / 'uv'
    executable.parent.mkdir(parents=True)
    executable.touch()
    result = subprocess.run(['make', '-n', 'setup', f'HOME={home}'], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert f'"{executable}" venv -p 3.13 --clear ".venv"' in result.stdout
    assert f'"{executable}" pip install --python ".venv/bin/python"' in result.stdout
    override = subprocess.run(['make', '-n', 'setup', 'UV=/custom/uv'], cwd=ROOT,
                              capture_output=True, text=True, check=True)
    assert '"/custom/uv" venv -p 3.13' in override.stdout
