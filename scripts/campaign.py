#!/usr/bin/env python3
"""Initialize or explicitly bind one Campaign variant. Never launches a render."""
import argparse
import json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from pipelines.ffmpeg.campaigns import load, save_binding

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO/'episodes/campaigns/arizona-reels.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init'); sub.add_parser('show')
    bind = sub.add_parser('bind')
    bind.add_argument('variant', choices=['sedona-kids', 'halloween-uptown', 'bellrock-resort'])
    bind.add_argument('episode_id')
    unbind = sub.add_parser('unbind')
    unbind.add_argument('variant', choices=['sedona-kids', 'halloween-uptown', 'bellrock-resort'])
    args = parser.parse_args(argv)
    try:
        value = load(args.manifest) if args.command == 'show' else save_binding(args.manifest,
            getattr(args, 'variant', None), getattr(args, 'episode_id', None))
        print(json.dumps(value, indent=2)); return 0
    except (ValueError, OSError) as error:
        print('Campaign operation failed: '+str(error), file=sys.stderr); return 1
if __name__ == '__main__': raise SystemExit(main())
