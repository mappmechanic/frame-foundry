"""Single-process episode ownership and atomic state."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
from .contracts import StageError,atomic_write_json

@contextmanager
def episode_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise StageError('RUN_ACTIVE','refinery','Another Episode is active') from None
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def save_state(ctx,stage,error=None):
    from datetime import datetime,timezone
    atomic_write_json(ctx.episode_dir/'state.json',dict(episode_id=ctx.episode_id,stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),error=error.as_dict() if error else None))
