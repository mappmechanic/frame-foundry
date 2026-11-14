"""Disk admission and owned work-media cleanup; never delete source/delivery media."""
from pathlib import Path
import os,shutil
from .contracts import StageError

MIN_FREE_BYTES=1_000_000_000  # 1 GB (decimal), inclusive
MEDIA_SUFFIXES={'.mp4','.mov','.mkv','.wav','.mp3','.m4a','.aac','.avi','.webm','.flac'}


def require_free_disk(path,*,stage='refinery'):
    path=Path(path).expanduser().resolve()
    while not path.exists() and path!=path.parent:path=path.parent
    try:free=shutil.disk_usage(path).free
    except OSError:raise StageError('DISK_CHECK_FAILED',stage,'Cannot determine free disk space for the output volume') from None
    if free<MIN_FREE_BYTES:
        raise StageError('DISK_SPACE_LOW',stage,f'Insufficient free disk space: {free/1_000_000_000:.3f} GB available; at least 1.000 GB required on {path}. Free space before generating or continuing the render.')
    return free


def cleanup_work_media(episode_dir):
    """Only ordinary media files under this Episode's real work directory."""
    root=Path(episode_dir)
    work=root/'work'
    if root.is_symlink() or work.is_symlink() or not work.is_dir():return dict(deleted=0,failed=0)
    deleted=failed=0
    for directory,dirs,files in os.walk(work,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            path=Path(directory)/name
            if path.suffix.lower() not in MEDIA_SUFFIXES or path.is_symlink():continue
            try:path.unlink();deleted+=1
            except OSError:failed+=1
    return dict(deleted=deleted,failed=failed)
