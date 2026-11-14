"""Read-only v1 learning-curve presentation; no experiment or provider imports.

Only the explicit improve command owns evals/learning-curves/latest.json.
Public output is a new allowlisted projection, never a copy of that file.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import stat

MAX_BYTES = 1_000_000
MAX_ROWS = 200
ID = re.compile(r'[a-z0-9][a-z0-9_-]{0,79}\Z')
COST_STATES = {'measured', 'estimated', 'complete', 'incomplete', 'unknown', 'unavailable', 'zero', 'not_recorded'}
METRICS = ('J', 'Q', 'model_usd', 'known_model_usd', 'elapsed_seconds')


def pending_curve():
    return dict(schema_version='1.0', experiment_id=None, status='pending', created_at=None,
                config_hash=None, arms=dict(memory_on=[], memory_off=[]), deltas=[],
                limitations=['Evaluation pending; no readable recorded learning curve.'])


def _text(value, limit=160):
    return isinstance(value, str) and 0 < len(value) <= limit


def _metric(value, unit=False):
    return value is None or (type(value) in (int, float) and math.isfinite(value)
                             and value >= 0 and (not unit or value <= 1))


def _validated(curve):
    if not isinstance(curve, dict) or curve.get('schema_version') != '1.0':
        raise ValueError('Invalid curve')
    if curve.get('status') == 'pending':
        return pending_curve()
    if curve.get('status') != 'recorded' or not _text(curve.get('experiment_id')) or not _text(curve.get('config_hash')):
        raise ValueError('Invalid curve metadata')
    created = curve.get('created_at')
    if not _text(created, 80) or datetime.fromisoformat(created.replace('Z', '+00:00')).tzinfo is None:
        raise ValueError('Invalid timestamp')
    arms = curve.get('arms')
    if not isinstance(arms, dict):
        raise ValueError('Invalid arms')
    clean = dict(schema_version='1.0', experiment_id=curve['experiment_id'], status='recorded',
                 created_at=created, config_hash=curve['config_hash'], arms={}, deltas=[])
    for arm in ('memory_on', 'memory_off'):
        rows = arms.get(arm)
        if not isinstance(rows, list) or len(rows) > MAX_ROWS:
            raise ValueError('Invalid rows')
        clean['arms'][arm] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Invalid row')
            i = row.get('iteration')
            if type(i) is not int or not 0 <= i <= 10000 or i in seen:
                raise ValueError('Invalid iteration')
            seen.add(i)
            if not _text(row.get('episode_id')) or row.get('status') not in ('passed', 'failed', 'unavailable'):
                raise ValueError('Invalid source')
            if not all(k in row and _metric(row[k], k in ('J', 'Q')) for k in METRICS):
                raise ValueError('Invalid metric')
            lessons = row.get('lessons_used')
            if not isinstance(lessons, list) or len(lessons) > 100 or not all(_text(x) for x in lessons):
                raise ValueError('Invalid lesson identities')
            if not isinstance(row.get('cost_status'), str) or row['cost_status'] not in COST_STATES:
                raise ValueError('Invalid cost status')
            clean['arms'][arm].append({k: row[k] for k in ('iteration', 'episode_id', 'status', *METRICS, 'cost_status', 'lessons_used')})
        clean['arms'][arm].sort(key=lambda r: r['iteration'])
    deltas = curve.get('deltas')
    if not isinstance(deltas, list) or len(deltas) > MAX_ROWS:
        raise ValueError('Invalid deltas')
    seen = set()
    for d in deltas:
        if not isinstance(d, dict) or type(d.get('iteration')) is not int or not 0 <= d['iteration'] <= 10000 or d['iteration'] in seen or type(d.get('comparable')) is not bool:
            raise ValueError('Invalid delta')
        seen.add(d['iteration'])
        for k in ('J', 'Q', 'model_usd'):
            v = d.get(k)
            if k not in d or (v is not None and (type(v) not in (int, float) or not math.isfinite(v))):
                raise ValueError('Invalid delta metric')
        clean['deltas'].append({k: d[k] for k in ('iteration', 'comparable', 'J', 'Q', 'model_usd')})
    limits = curve.get('limitations')
    if not isinstance(limits, list) or len(limits) > 100 or not all(_text(x, 2000) for x in limits):
        raise ValueError('Invalid limitations')
    clean['limitations'] = limits
    return clean


def _read_latest(evals_root, directory, validator, pending):
    """Bounded, no-follow read of latest.json; missing/invalid/symlink => pending.

    Refuse symlinks in every path component, open directories relative to their
    descriptors, and never follow a pointer to an experiment or private file.
    """
    descriptors = []
    try:
        path = Path(evals_root).expanduser().absolute() / directory / 'latest.json'
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(fd)
        for part in path.parts[1:-1]:
            if part == '..':
                raise ValueError('Traversal')
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        leaf = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        descriptors.append(leaf)
        info = os.fstat(leaf)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError('Not a bounded regular file')
        with os.fdopen(os.dup(leaf), 'rb') as handle:
            raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('Too large')
        return validator(json.loads(raw))
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        return pending()
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def read_curve(evals_root):
    return _read_latest(evals_root, 'learning-curves', _validated, pending_curve)


def pending_policy_comparison():
    return dict(schema_version='1.0', status='pending', baseline=None, production=None,
                comparison=dict(comparable=False, J=None, Q=None, model_usd=None))


def _validated_policy(value):
    if not isinstance(value, dict) or value.get('schema_version') != '1.0':
        raise ValueError('Invalid policy comparison')
    if value.get('status') == 'pending':
        return pending_policy_comparison()
    if value.get('status') != 'recorded' or not _text(value.get('experiment_id')):
        raise ValueError('Invalid policy metadata')
    created = value.get('created_at')
    if not _text(created, 80) or datetime.fromisoformat(created.replace('Z', '+00:00')).tzinfo is None:
        raise ValueError('Invalid timestamp')
    out = dict(schema_version='1.0', status='recorded', experiment_id=value['experiment_id'],
               created_at=created, baseline=None, production=None)
    for arm in ('baseline', 'production'):
        row = value.get(arm)
        if not isinstance(row, dict) or not isinstance(row.get('episode_id'), str) or not ID.fullmatch(row['episode_id']):
            raise ValueError('Invalid source')
        if row.get('status') not in ('passed', 'failed', 'unavailable'):
            raise ValueError('Invalid status')
        if not isinstance(row.get('cost_status'), str) or row['cost_status'] not in COST_STATES:
            raise ValueError('Invalid cost status')
        fields = ('J', 'Q', 'model_usd', 'known_model_usd')
        if not all(k in row and _metric(row[k], k in ('J', 'Q')) for k in fields):
            raise ValueError('Invalid metric')
        out[arm] = {k: row[k] for k in ('episode_id', 'status', 'cost_status', *fields)}
    if out['baseline']['episode_id'] == out['production']['episode_id']:
        raise ValueError('Same Episode is not an A/B pair')
    delta = value.get('comparison')
    if not isinstance(delta, dict) or type(delta.get('comparable')) is not bool:
        raise ValueError('Invalid comparison')
    for k in ('J', 'Q', 'model_usd'):
        v = delta.get(k)
        if k not in delta or (v is not None and (type(v) not in (int, float) or not math.isfinite(v))):
            raise ValueError('Invalid delta')
    limits = value.get('limitations')
    if not isinstance(limits, list) or len(limits) > 100 or not all(_text(x, 2000) for x in limits):
        raise ValueError('Invalid limitations')
    out['comparison'] = {k: delta[k] for k in ('comparable', 'J', 'Q', 'model_usd')}
    out['limitations'] = limits
    return out


def read_policy_comparison(evals_root):
    """Parent writes policy-comparisons/latest.json; bounded no-follow optional read."""
    return _read_latest(evals_root, 'policy-comparisons', _validated_policy, pending_policy_comparison)


def public_policy_projection(value, allowed_ids):
    """Whole pair fails closed; no free text, experiment IDs or private metadata.

    Observations require two distinct exported passed Episodes with known scores
    and complete costs. Delta eligibility additionally requires backend comparable.
    Supplied deltas are never trusted: derive production minus baseline.
    """
    try:
        value = _validated_policy(value)
    except (ValueError, TypeError, KeyError, OverflowError):
        value = pending_policy_comparison()
    out = pending_policy_comparison()
    out.update(status=value['status'], source_episode_ids=[],
               limitations=['Recorded observations only; no causal or sponsor improvement claim.',
                            'Both passed source Episodes, scores and complete costs must be publicly exported.'])
    allowed = {x for x in allowed_ids if isinstance(x, str) and ID.fullmatch(x)}
    rows = [value.get(arm) for arm in ('baseline', 'production')]
    visible = value['status'] == 'recorded' and all(
        r and r['episode_id'] in allowed and r['status'] == 'passed'
        and r['cost_status'] in ('measured', 'estimated', 'complete', 'zero')
        and all(r[k] is not None for k in ('J', 'Q', 'model_usd', 'known_model_usd'))
        and r['known_model_usd'] <= r['model_usd']
        and (r['cost_status'] != 'zero' or r['model_usd'] == 0)
        for r in rows)
    if not visible:
        return out
    for arm, row in zip(('baseline', 'production'), rows):
        out[arm] = dict(row, data_url=f"/runs/{row['episode_id']}/run.json")
    out['source_episode_ids'] = [r['episode_id'] for r in rows]
    comparable = value['comparison']['comparable']
    out['comparison'] = dict(comparable=comparable, **{
        k: rows[1][k] - rows[0][k] if comparable else None for k in ('J', 'Q', 'model_usd')})
    return out


def public_projection(curve, allowed_ids):
    """Only actual exported Episode IDs may expose metrics or evidence links.

    Failed/unpublished rows keep iteration/status but no identity, metrics or
    link. Drop all free text, hashes, lesson IDs, paths and experiment metadata.
    Deltas are derived only for recorded comparable pairs of public passed rows.
    """
    try:
        curve = _validated(curve)
    except (ValueError, TypeError, KeyError, OverflowError):
        curve = pending_curve()
    allowed = {x for x in allowed_ids if isinstance(x, str) and ID.fullmatch(x)}
    out = dict(schema_version='1.0', status=curve['status'], arms={}, deltas=[], source_episode_ids=[],
               limitations=['Recorded observations only; improvement and causation are not established.',
                            'Failed or unpublished Episodes have no public metrics or evidence links.'])
    for arm, rows in curve['arms'].items():
        out['arms'][arm] = []
        for row in rows:
            visible = row['status'] == 'passed' and row['episode_id'] in allowed
            projected = dict(iteration=row['iteration'], status=row['status'], published=visible,
                             episode_id=row['episode_id'] if visible else None,
                             data_url=f"/runs/{row['episode_id']}/run.json" if visible else None,
                             cost_status=row['cost_status'] if visible and row['cost_status'] in COST_STATES else 'unavailable')
            projected.update({k: row[k] if visible else None for k in METRICS})
            if visible and projected['cost_status'] not in ('measured', 'estimated', 'complete', 'zero'):
                projected['model_usd'] = None
            out['arms'][arm].append(projected)
    out['source_episode_ids'] = sorted({r['episode_id'] for rows in out['arms'].values() for r in rows if r['published']})
    on = {r['iteration']: r for r in out['arms']['memory_on']}
    off = {r['iteration']: r for r in out['arms']['memory_off']}
    declared = {d['iteration']: d['comparable'] for d in curve['deltas']}
    for i in sorted(on.keys() | off.keys()):
        a, b = on.get(i), off.get(i)
        comparable = bool(declared.get(i) and a and b and a['published'] and b['published'] and a['episode_id'] != b['episode_id'])
        d = dict(iteration=i, comparable=comparable)
        for k in ('J', 'Q', 'model_usd'):
            d[k] = a[k] - b[k] if comparable and a[k] is not None and b[k] is not None else None
        out['deltas'].append(d)
    return out
