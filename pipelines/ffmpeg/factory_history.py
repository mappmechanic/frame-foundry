"""Generation-time factory metadata; audited title coverage is not completion."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / 'contracts' / 'demo-factory-history.json'
PR_URL = 'https://github.com/mappmechanic/frame-foundry/pull/1'


def _timestamp(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('timezone required')
    return stamp


def coverage_minutes(created_at, first_coverage_at):
    """Elapsed minutes to the first recorded PR-title coverage, never ticket delivery."""
    minutes = (_timestamp(first_coverage_at) - _timestamp(created_at)).total_seconds() / 60
    if minutes < 0:
        raise ValueError('title coverage precedes creation')
    return minutes


def _run(args, repo):
    try:
        result = subprocess.run(args, cwd=repo, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return None, 'command unavailable'
    except subprocess.TimeoutExpired:
        return None, 'command timed out'
    except (OSError, subprocess.SubprocessError):
        return None, 'command unavailable'
    if result.returncode:
        # Do not expose auth diagnostics, private paths, or tokens.
        return None, 'command failed (unavailable or unauthenticated)'
    return result.stdout.strip(), None


def collect_factory_history(repo=REPO, *, snapshot_path=SNAPSHOT):
    """Fetch optional gh REST metadata now, never at page view. No model calls."""
    snapshot = json.loads(Path(snapshot_path).read_text())
    generated_at = datetime.now(timezone.utc).isoformat()
    work_orders = [dict(wo, coverage_minutes=coverage_minutes(wo['created_at'], wo['first_coverage_at']))
                   for wo in snapshot['work_orders']]
    git_text, git_reason = _run(['git', 'rev-list', '--count', 'main..HEAD'], repo)
    git_count = None
    if git_reason is None:
        if git_text.isascii() and git_text.isdecimal():
            git_count = int(git_text)
        else:
            git_reason = 'invalid git count'
    pr = dict(snapshot['pull_request'])
    pr.update(source='audited snapshot', observed_at=snapshot['fetched_at'], reason=None)
    text, reason = _run(['gh', 'api', 'repos/mappmechanic/frame-foundry/pulls/1'], repo)
    if reason is None:
        try:
            data = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(data, dict) or type(data.get('commits')) is not int or data['commits'] < 0:
                raise ValueError('invalid commits')
            if data.get('number') != 1 or data.get('state') not in ('open', 'closed'):
                raise ValueError('invalid PR metadata')
            _timestamp(data['created_at'])
            pr.update(commits=data['commits'], state='merged' if data.get('merged_at') else data['state'],
                      created_at=data['created_at'], source='GitHub REST at generation', observed_at=generated_at)
        except (ValueError, TypeError, KeyError, AttributeError):
            reason = 'invalid GitHub JSON/metadata'
    if reason:
        pr['reason'] = reason + '; fields without verified snapshot values are N/A'
    return dict(generated_at=generated_at, fetched_at=snapshot['fetched_at'],
                notes=snapshot['notes'], work_orders=work_orders,
                git=dict(commits=git_count, scope='local main..HEAD', reason=git_reason), pull_request=pr)
