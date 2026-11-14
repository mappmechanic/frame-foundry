# FrameFoundry — Agent Execution & Architecture Rules (`AGENTS.md`)

## 1. Repository Purpose
FrameFoundry (`mappmechanic/frame-foundry`) is an autonomous, code-first Video Editing & Vlog Factory orchestrated by **8090 Software Factory** (Control Plane) and executed by **Claude Code / OpenHands** (Worker Plane).
Every video episode is treated as a deterministic build artifact compiled from raw footage into an OpenTimelineIO (`.otio`) timeline, rendered Remotion overlays, and a verified `.mp4` cut.

## 2. Directory Topology & Ownership
- `docs/sops/`: Markdown Standard Operating Procedures synced with the 8090 Knowledge Base.
- `schemas/`: Strict JSON Schema contracts between pipeline stages (`narrative-spec.schema.json`, `qa-report.schema.json`).
- `pipelines/ffmpeg/`: Python + FFmpeg + WhisperX + OpenTimelineIO headless media processing scripts.
- `packages/remotion-studio/`: React/TypeScript Remotion compositions for titles, lower-thirds, and B-roll overlays.
- `episodes/<episode-id>/`: Isolated per-video workspace containing `transcript.json`, `narrative-spec.json`, `timeline.otio`, `contact-sheet.jpg`, and `qa-report.json`.

## 3. Hard Guardrails for Agents
1. **Never commit raw media binaries** (`.mp4`, `.mov`, `.wav`, `.mkv`) to Git. Only commit `.otio`, `.fcpxml`, `.json`, `.md`, and low-res QA `.jpg` contact sheets.
2. **Audio Cut Integrity:** Never cut on an exact word boundary. Always apply `+80ms` pre-roll before word onset and `+120ms` post-roll after word decay.
3. **Verification Gate:** No Work Order is complete until `pipelines/ffmpeg/qa_validator.py` passes both the `-14 LUFS` ($\pm 1.0$ dB) EBU R128 loudness check and generates a frame-boundary contact sheet.
4. **PR Traceability:** Always include the 8090 Work Order ID (e.g., `[WO-001]`) in branch names and Pull Request titles so 8090 Drift Bot links and validates the PR automatically.
