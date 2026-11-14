"""Download private release outside Git and invoke the real pipeline.

No API key loading or fixture substitution. Real ASR defaults to local WhisperX
with an OpenAI word-timestamp fallback when OPENAI_API_KEY is configured.
--visual-only remains explicit no-ASR diagnostic mode; it never passes word QA.
ASR, Planner and judge share a $0.10 cap; stricter configured limits are retained.
"""
import argparse
from decimal import Decimal
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ARIZONA_BUDGET_USD = Decimal('0.10')

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


def child_environment():
    """One shared ASR/Planner/judge cap, without mutating process configuration."""
    from pipelines.ffmpeg.llm_provider import BUDGET_ENV, episode_budget_usd, episode_token_budget
    # Validate before clamping: malformed configuration is terminal, not a default.
    configured = episode_budget_usd()
    episode_token_budget()
    env = os.environ.copy()
    env[BUDGET_ENV] = str(min(configured, ARIZONA_BUDGET_USD))
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('FRAMEFOUNDRY_SAMPLE_DIR', '~/.cache/framefoundry/arizona')))
    parser.add_argument('--episode')
    parser.add_argument('--episodes-root', type=Path)
    parser.add_argument('--visual-only', action='store_true')
    parser.add_argument('--asr-provider',choices=['auto','local','openai'],default='auto')
    parser.add_argument('--evals-root',type=Path)
    parser.add_argument('--no-planner-retry',action='store_true')
    parser.add_argument('--repair-from')
    parser.add_argument('--voiceover',action='store_true')
    parser.add_argument('--keep-work',action='store_true')
    parser.add_argument('--fetch-only', action='store_true')
    args = parser.parse_args(argv)
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        parser.error("FrameFoundry requires Python >=3.10,<3.14 with OTIO 0.18.1; recreate .venv with uv venv -p 3.13 --clear")
    env = None
    if not args.fetch_only:
        from pipelines.ffmpeg.llm_provider import ProviderError
        try:
            env = child_environment()
        except ProviderError as error:
            parser.error(str(error))
    footage = fetch(args.data_dir)
    if args.fetch_only:
        print(footage)
        return 0
    command = [sys.executable, '-m', 'pipelines.ffmpeg.cli', 'run', '--footage-dir', str(footage), '--goal', GOAL,'--asr-provider',args.asr_provider]
    if args.episode: command += ['--episode', args.episode]
    if args.episodes_root: command += ['--episodes-root', str(args.episodes_root)]
    if args.visual_only: command += ['--visual-only']
    if args.evals_root: command += ['--evals-root',str(args.evals_root)]
    if args.repair_from: command += ['--repair-from',args.repair_from]
    if args.voiceover: command += ['--voiceover']
    if args.keep_work: command += ['--keep-work']
    if args.no_planner_retry: command += ['--no-planner-retry']
    return subprocess.run(command, check=False, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
