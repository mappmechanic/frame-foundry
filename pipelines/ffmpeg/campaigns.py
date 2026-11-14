"""Repository-local campaign grouping; labels never certify footage or publication."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import re
import tempfile
import os

ID = re.compile(r'[a-z0-9][a-z0-9-]{0,63}\Z')
DEFAULT = {'schema_version': '1.0', 'id': 'arizona-reels', 'title': 'Arizona Reels', 'variants': [
    {'id': 'sedona-kids', 'label': 'Sedona with kids overlays', 'target_seconds': 45, 'episode_id': None},
    {'id': 'halloween-uptown', 'label': 'Halloween Uptown', 'target_seconds': 15, 'episode_id': None},
    {'id': 'bellrock-resort', 'label': 'BellRock resort', 'target_seconds': 15, 'episode_id': None}]}

def validate(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT):
        raise ValueError('Invalid campaign manifest')
    if value['schema_version'] != '1.0' or value['id'] != DEFAULT['id'] or value['title'] != DEFAULT['title']:
        raise ValueError('Unsupported campaign')
    variants = value['variants']
    if not isinstance(variants, list) or len(variants) != 3:
        raise ValueError('Campaign requires three planned slots')
    for variant, planned in zip(variants, DEFAULT['variants']):
        if not isinstance(variant, dict) or set(variant) != set(planned):
            raise ValueError('Invalid variant')
        if any(variant[k] != planned[k] for k in ('id', 'label', 'target_seconds')):
            raise ValueError('Unsupported variant')
        eid = variant['episode_id']
        if eid is not None and (not isinstance(eid, str) or not ID.fullmatch(eid)):
            raise ValueError('Unsafe Episode identifier')
    return value

def load(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Manifest symlink not allowed')
    if not path.exists():
        return deepcopy(DEFAULT)
    if path.stat().st_size > 16384:
        raise ValueError('Manifest too large')
    return validate(json.loads(path.read_text()))

def save_binding(path, variant_id=None, episode_id=None):
    """Locked atomic save; initialization refuses to overwrite existing bindings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lockpath = path.with_suffix('.lock')
    if lockpath.is_symlink():
        raise ValueError('Manifest lock symlink not allowed')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = load(path)
        if variant_id is not None:
            variant = next((v for v in value['variants'] if v['id'] == variant_id), None)
            if variant is None:
                raise ValueError('Unknown variant')
            variant['episode_id'] = episode_id
        validate(value)
        fd, temporary = tempfile.mkstemp(prefix='.campaign-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as out:
                json.dump(value, out, indent=2, allow_nan=False)
                out.write('\n')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    return value

def project(manifest, runs=(), *, failed_ids=()):
    """Only accept run summaries from the existing verified export allowlist."""
    value = deepcopy(validate(manifest))
    allowed = {r['id']: r for r in runs}
    failed = set(failed_ids)
    for variant in value['variants']:
        eid = variant['episode_id']
        variant['status'] = 'planned' if eid is None else 'failed' if eid in failed else 'unavailable'
        variant['run'] = None
        if eid in allowed:
            variant['status'] = 'verified'
            variant['run'] = {k: allowed[eid].get(k) for k in ('id', 'scores', 'cost', 'data_url', 'video_url', 'contact_sheet_url')}
    return value
