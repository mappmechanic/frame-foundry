#!/usr/bin/env python3
"""Read-only A/B projection; never performs inference or activates a default."""
import argparse
import json
from pathlib import Path
from pipelines.ffmpeg.contracts import read_json,atomic_write_json
from pipelines.ffmpeg.decisions_ab import arm,compare

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llm',required=True,type=Path)
    p.add_argument('--decisions',required=True,type=Path)
    p.add_argument('--llm-config',required=True,type=Path)
    p.add_argument('--decisions-config',required=True,type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); result=compare(arm(a.llm,read_json(a.llm_config)),arm(a.decisions,read_json(a.decisions_config)))
    if a.output: atomic_write_json(a.output,result)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
