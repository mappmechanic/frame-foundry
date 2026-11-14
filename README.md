# FrameFoundry (8090 Video Factory)
Autonomous vlog & video compilation pipeline orchestrated via 8090 Software Factory and OpenHands/Claude Code.

## Architecture
- `docs/sops/`: Standard Operating Procedures for Refinery, Foundry, Planner, Assembler, and Validator stages.
- `schemas/`: JSON Contracts for narrative specs, OTIO timelines, and QA telemetry.
- `packages/remotion-studio/`: Reusable React motion graphics & brand templates.
- `pipelines/ffmpeg/`: Headless media normalization, WhisperX diarization, and EBU R128 audio tools.
- `episodes/`: Per-video work order outputs (`.otio`, proxy renders, contact sheets).

## Teaser v1: local pipeline with an embedded multimodal agent

### Install

Python 3.13 was used for the lightweight pipeline tests. Install FFmpeg/ffprobe,
then from the repository root:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

For real spoken footage, install `requirements-whisperx.txt` in a compatible
CPU/GPU environment and deliberately provision local model assets before a run.
`FRAMEFOUNDRY_MODEL_DIR` contains `transcription/` (faster-whisper local model)
and `alignment/` (WhisperX/Hugging Face alignment model). WhisperX/VAD assets must
already be cached. Runtime model initialization disallows network connections;
missing assets yield a reported setup failure, never fabricated transcription.
The optional WhisperX installation and real recognition were **not** validated
by the synthetic test suite. No models are downloaded implicitly by the CLI.

### Run

```sh
.venv/bin/python -m pipelines.ffmpeg.cli \
  --footage-dir /path/to/family-trip \
  --goal 'Create a 45 second teaser highlighting our family trip' \
  --episode family-trip
```

Outputs live in `episodes/family-trip/`: `finished.mp4`, `qa-report.json`,
`transcript.json`, `analysis.json`, `narrative-spec.json`, `timeline.otio`,
`contact-sheet.jpg`, and ignored proxies/attempt media. One active run only;
existing Episode IDs are rejected. Original media stays unchanged. No human
review gate is required. Defaults: 45 seconds within the 30–60-second teaser
range, vertical 9:16; a Goal may specify 5–120 seconds or landscape 16:9.

Ingest accepts up to 40 MP4/MOV sources and 30 minutes total. It creates
orientation-corrected, ≤640px, 15fps H.264 proxies before image/scene analysis,
with three sampled frames per source and timestamp mapping. A generated
26-clip 4K HEVC set is covered by tests; this is **not** the user's real 4.4-minute
footage. SDR synthetic media and rotation were tested; HDR/Dolby Vision color
fidelity and real iPhone VFR footage still require verification.

### Configure the agent

Set secrets in workspace/environment settings, never commit them:

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- `FRAMEFOUNDRY_LLM_PROVIDER=anthropic|openai` (optional; Anthropic wins if both keys exist)
- `FRAMEFOUNDRY_LLM_MODEL` (optional; defaults `claude-sonnet-4-5` / `gpt-4.1`)
- `FRAMEFOUNDRY_INPUT_USD_PER_MILLION`, `FRAMEFOUNDRY_OUTPUT_USD_PER_MILLION`
  to price a custom model; unpriced models are blocked.
- `FRAMEFOUNDRY_PROJECT_SPENT_USD` supplies initial already-spent budget when the
  project ledger is first created. Assumed initial AI spend is zero if omitted;
  that is **not proof** that labor/infrastructure/earlier project expenses are zero.
- `FRAMEFOUNDRY_PROJECT_BUDGET_DIR` optionally selects the project's durable ledger
  directory; by default it is the repository root, shared across Episode roots.

Configured keys enable the default agent: bounded sampled JPEGs and transcript
excerpts are sent to that provider. Original video is never uploaded. Prompts
label footage/Goal/transcript as untrusted data and provide no tool permissions.
The agent selects and orders moments with rationale, then local schema/source
checks validate them. No key, errors, invalid output, or budget denial trigger a
recorded deterministic fallback. The fallback is not semantic understanding.

The advisory AI judge looks at **rendered output** samples and selected
transcript, scores goal coverage 0–100, and cites evidence. An assumed score below
60 flags partial output; judge failure/no key yields a null score. It cannot
waive technical gates or trigger unlimited attempts. Standalone revalidation
makes no paid call and preserves the earlier judgment.

### Budget and QA

Provider-reported token counts, model identity, call outcomes, estimated API cost,
and unknown failed-call reservations are retained in `llm-usage.json` and
`qa-report.json`. Default rates are labeled estimates, not current billing proof.
Known API subtotal is `cost.known_ai_usd`; per-call `cost_status` distinguishes
estimated from unknown. Total run expense stays unknown if compute/billing is
unknown; do not report unknown as zero. Use known API subtotal per passing output
for API-cost-per-finished-clip and retain failures in the numerator.

A durable project ledger reserves $5 per AI-enabled Episode under a $500 ceiling;
per-call reservations enforce the same $5 Episode ceiling including failures.
Reservations conservatively remain allocated; reconcile initial spend manually
before enabling expensive runs. Changing the Episode output root does not reset
the shared project ledger. Do not change the project budget directory to bypass
existing reservations; it identifies the same durable budget across runs.

Hard gates independently inspect H.264/AAC dimensions/30fps, schema/source/OTIO
ranges, spoken padding (80ms before, 120ms after), rendered timeline duration,
EBU R128 **−14 LUFS ±1 dB**, true peak **≤−1dBTP**, and labeled cut-boundary
contact sheets. Silence/nonfinite loudness never passes. A maximum of two repairs
follows the initial attempt; only a passing candidate becomes `finished.mp4`.

```sh
.venv/bin/python pipelines/ffmpeg/qa_validator.py --episode-dir episodes/family-trip
```

### Offline verification

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python tests/fixtures/generate_footage.py --out /tmp/trip-fixture
# Diagnostic-only: explicitly labels fake/empty transcript evidence, never default.
ANTHROPIC_API_KEY= OPENAI_API_KEY= .venv/bin/python -m pipelines.ffmpeg.cli \
  --footage-dir /tmp/trip-fixture --goal 'Create a 10 second trip teaser' \
  --episode fixture-demo --transcript-fixture /tmp/trip-fixture/transcript-fixture.json
```

Tests use mocked provider transports and disposable real FFmpeg media, including
26 short 4K HEVC clips. No paid calls, API secrets, or media binaries are committed.
Generated tones/word sidecars do not prove speech recognition or artistic quality.
Live provider calls and the actual family footage remain opt-in verification.
