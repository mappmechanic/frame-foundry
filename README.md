# FrameFoundry (8090 Video Factory)
Autonomous vlog & video compilation pipeline orchestrated via 8090 Software Factory and OpenHands/Claude Code.

## Architecture
- `docs/sops/`: Standard Operating Procedures for Refinery, Foundry, Planner, Assembler, and Validator stages.
- `schemas/`: JSON Contracts for narrative specs, OTIO timelines, and QA telemetry.
- `packages/remotion-studio/`: Reusable React motion graphics & brand templates.
- `pipelines/ffmpeg/`: Headless media normalization, WhisperX diarization, and EBU R128 audio tools.
- `episodes/`: Per-video work order outputs (`.otio`, proxy renders, contact sheets).
