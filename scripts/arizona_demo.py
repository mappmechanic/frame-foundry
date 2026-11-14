"""Download private release outside Git and invoke the real pipeline.

No API key loading or implicit ASR/fixture substitution. Use --visual-only
explicitly for a diagnostic run when local WhisperX assets are unavailable.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile

GOAL = 'Create a teaser from this raw family-trip footage that highlights the important moments of the trip'


def fetch(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['gh', 'release', 'download', 'sample-data-arizona-480p', '-R',
                    'mappmechanic/frame-foundry', '--pattern', 'ff-arizona-480p.zip',
                    '--dir', str(root), '--skip-existing'], check=True)
    footage = root / 'footage'
    with zipfile.ZipFile(root / 'ff-arizona-480p.zip') as archive:
        for entry in archive.infolist():
            target = (footage / entry.filename).resolve()
            if not target.is_relative_to(footage) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe release archive entry')
        archive.extractall(footage)
    clips = footage / 'ff-samples'
    if len(list(clips.glob('*.mp4'))) != 26:
        raise ValueError('Arizona release must contain 26 clips')
    return clips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('FRAMEFOUNDRY_SAMPLE_DIR', '~/.cache/framefoundry/arizona')))
    parser.add_argument('--episode')
    parser.add_argument('--episodes-root', type=Path)
    parser.add_argument('--visual-only', action='store_true')
    parser.add_argument('--evals-root',type=Path)
    parser.add_argument('--no-planner-retry',action='store_true')
    parser.add_argument('--fetch-only', action='store_true')
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 13):
        parser.error("FrameFoundry requires Python 3.13 with OTIO 0.18.1; recreate .venv with uv venv --python 3.13")
    footage = fetch(args.data_dir)
    if args.fetch_only:
        print(footage)
        return 0
    command = [sys.executable, '-m', 'pipelines.ffmpeg.cli', 'run', '--footage-dir', str(footage), '--goal', GOAL]
    if args.episode: command += ['--episode', args.episode]
    if args.episodes_root: command += ['--episodes-root', str(args.episodes_root)]
    if args.visual_only: command += ['--visual-only']
    if args.evals_root: command += ['--evals-root',str(args.evals_root)]
    if args.no_planner_retry: command += ['--no-planner-retry']
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
