# FrameFoundry — Agent Execution & Architecture Rules (`AGENTS.md`)

## 1. Repository Purpose
FrameFoundry (`mappmechanic/frame-foundry`) is an autonomous, code-first Video Editing & Vlog Factory orchestrated by **8090 Software Factory** (Control Plane) and executed by **Claude Code / OpenHands** (Worker Plane).
Every video episode is treated as a source-bound build artifact compiled from raw footage into an OpenTimelineIO (`.otio`) timeline and a verified `.mp4` cut. At `ea0887e`, overlays use Pillow PNG + FFmpeg sparse active-window compositing, not Remotion. Deterministic frame math does not imply byte equality across toolchains.

## 2. Directory Topology & Ownership
- `docs/sops/`: Markdown Standard Operating Procedures synced with the 8090 Knowledge Base.
- `schemas/`: Strict JSON Schema contracts between pipeline stages (`narrative-spec.schema.json`, `qa-report.schema.json`).
- `pipelines/ffmpeg/`: Python + FFmpeg + WhisperX + OpenTimelineIO headless media processing scripts.
- `packages/remotion-studio/`: declared future React/TypeScript Remotion location; **absent/not installed** in this checkout. V3 alpha overlays are a design, not current runtime support.
- `episodes/<episode-id>/`: Isolated per-video workspace containing `transcript.json`, `narrative-spec.json`, `timeline.otio`, `contact-sheet.jpg`, and `qa-report.json`.

## 3. Hard Guardrails for Agents
1. **Never commit raw media binaries** (`.mp4`, `.mov`, `.wav`, `.mkv`) to Git. Only commit `.otio`, `.fcpxml`, `.json`, `.md`, and low-res QA `.jpg` contact sheets.
2. **Audio Cut Integrity:** Never cut on an exact word boundary. Always start `80ms` before word onset and `+120ms` post-roll after word decay.
3. **Verification Gate:** No Work Order is complete until `pipelines/ffmpeg/qa_validator.py` passes both the `-14 LUFS` ($\pm 1.0$ dB) EBU R128 loudness check and generates a frame-boundary contact sheet.
4. **PR Traceability:** Always include the 8090 Work Order ID (e.g., `[WO-001]`) in branch names and Pull Request titles so 8090 Drift Bot links and validates the PR automatically.


## 4. Curated worker routing (WO-17)

Read only the applicable small skill; do not bulk-load vendor prompts.

| Stage | Guidance | Responsibility |
| --- | --- | --- |
| Plan / rough cut | [.claude/skills/rough-cut/SKILL.md](.claude/skills/rough-cut/SKILL.md) | Verified silence, complete retakes and source/protected-word mapping |
| Plan / pacing | [.claude/skills/vlog-pacing/SKILL.md](.claude/skills/vlog-pacing/SKILL.md) | Supported first-2s hook, duration exceptions and evidence-safe framing |
| Assemble / audio | [.claude/skills/voice-and-audio/SKILL.md](.claude/skills/voice-and-audio/SKILL.md) | Authoritative padding/LUFS/peak and only authorized optional music |
| Assemble / overlays | [.claude/skills/remotion-sparse-overlays/SKILL.md](.claude/skills/remotion-sparse-overlays/SKILL.md) | Current PNG renderer vs future Remotion/V3 active alpha ranges |
| Validate | [.claude/skills/visual-qa-validator/SKILL.md](.claude/skills/visual-qa-validator/SKILL.md) | Actual encoded frames, hashes, holds, masks and honest limitations |

Read [SOP-03: Measure -> Draw -> Verify](docs/sops/SOP-03-remotion-sparse-overlays.md)
for overlay architecture. SOP-01/SOP-02 files are absent from this checkout; KB
lists PDFs but their contents are not assumed here. Do not invent local paths.
Current schemas, 80ms pre-roll / 120ms post-roll, [-15,-13] LUFS / <=-1 dBTP,
.06H cap height (~116px portrait), >=2.5s steady holds and cut-1/cut/cut+1 masks
remain authoritative. Stricter zones, landmark checks, cut+3 exclusion and the
14/120/1 spring are declared targets, not installed capabilities.

Parent/WO-18 owns the versioned opt-in production policy/runtime and any actual
schema fields; ordinary edits stay unchanged by default. This docs-only branch
modifies no Python runtime, schemas, pyproject or site. Documentation checks do
not certify video output or demonstrate a production-quality improvement. The
media QA guardrail above remains required whenever a rendered edit is delivered.

See [availability/provenance](docs/skills/README.md) and the
[pinned MIT manifest](docs/skills/provenance.json). `_vendor_skills/` is ignored
inspection staging. Preserve full linked notices, including font/media exceptions;
never import staged assets, installers or bulk prompts. Blocked noncommercial,
unlicensed and AGPL sources were not cloned/read/copied. No npm/npx installers.

## 5. Opt-in runtime availability (WO-18)

The earlier WO-17 table records the documentation-only milestone. The current
`--production` implementation is mapped in [PRODUCTION_POLICY.md](docs/PRODUCTION_POLICY.md):
`production.py` applies source-preserving hook/pacing/silence rules; `audio_policy.py`
mixes authorized optional beds; `overlays.py` applies the14/120/1 analytic spring
and stricter zones; QA adds bound policy/subject checks and cut+3 exclusion sheets.
`policy_ab.py` records matched saved-plan evaluation; publication requires actual
exported complete evidence. Ordinary edits remain legacy by default. Remotion/V3,
continuous eye/chin tracking and112% face-safe punch-ins are NOT implemented.
No new Planner/ASR/TTS call is needed for deterministic saved-plan replay.
