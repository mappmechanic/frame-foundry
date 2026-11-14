# FrameFoundry

A local footage-to-clip CLI: deterministic media processing and independent QA,
with bounded multimodal editing judgment. No media binaries or secrets belong in Git.

## Install (Mac or Linux)

Install FFmpeg/ffprobe (`brew install ffmpeg` on Mac), then from this checkout:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

The console entry point is `.venv/bin/framefoundry`. Editable installation from
this checkout keeps the bundled `schemas/` and `contracts/` available. Lightweight
verification covers Python 3.13 and 3.14; CI covers Linux 3.12/3.14 and Mac 3.14. FFmpeg is an external prerequisite.

### Python 3.14 / OTIO metadata compatibility

The assembler normalizes nested metadata to plain JSON values, then attaches it
through OTIO's native JSON reader. It never passes Timeline/Clip metadata mappings
to OTIO constructors, avoiding the reported Mac `RuntimeError: bad any cast` path.
Source intervals, bool flags and selection mappings retain their types through
`.otio` save/readback. OTIO remains pinned at 0.18.1; no Python downgrade is needed
for this construction path. Python 3.14 verification in the development workspace
uses Linux; the exact Mac mini/Python 3.14/FFmpeg 8.1 environment still needs its
local rerun, with a Mac Python 3.14 CI job added to catch platform regressions.


### Local speech recognition

For real speech/word-integrity verification, install `requirements-whisperx.txt`
in a compatible CPU/GPU environment and deliberately provision local assets:
`FRAMEFOUNDRY_MODEL_DIR/transcription/` (faster-whisper model), `alignment/`
(WhisperX alignment model), and cached VAD assets. Runtime model initialization
blocks downloads. Missing assets produce an accepted-run failure report, not fake
transcription. Genuine WhisperX and Mac hardware execution are not yet verified.

`--visual-only` is an **explicit diagnostic mode** when these assets are absent.
It can render real footage/audio and pass the existing technical QA, but does not
certify speech integrity. Its eval word gate is `not_run`, G=0 and Q=0; no judge
is called. An empty fixture transcript is never substituted for real speech.

## Arizona demo (26 real clips, private release)

Load your trusted local `.env` into the process environment; the providers do not
read or log credential files. Do not print the keys or enable shell tracing.

```sh
set -a
source .env
set +a
export FRAMEFOUNDRY_LLM_PROVIDER=openai
# With provisioned local ASR assets:
make demo-arizona ARGS='--episode arizona-demo'
# Or an explicitly labeled visual-only diagnostic:
make demo-arizona ARGS='--episode arizona-visual --visual-only'
```

The target downloads `ff-arizona-480p.zip` from private GitHub release
`sample-data-arizona-480p` using your `gh` authentication, into
`~/.cache/framefoundry/arizona` by default. Override with `--data-dir PATH` or
`FRAMEFOUNDRY_SAMPLE_DIR`. It safely extracts and verifies 26 MP4 files outside
Git, then uses the exact Goal:

> Create a teaser from this raw family-trip footage that highlights the important moments of the trip

Use a fresh Episode ID on each run. Outputs are in `episodes/<id>/`:
`finished.mp4`, `contact-sheet.jpg`, `qa-report.json`, `eval-report.json`,
`narrative-spec.json`, `timeline.otio`, transcript/analysis, and ignored work media.
Strict eval-duration failures retain `diagnostic.mp4`, not an advertised finished
clip. Default target is 45 seconds and 1080×1920, 30 fps; a Goal can request 16:9
(1920×1080) and an explicit 5–120 second target. Originals are never modified.

`make fetch-arizona` only downloads. To run the private sample integration test:

```sh
FRAMEFOUNDRY_ARIZONA_FOOTAGE="$HOME/.cache/framefoundry/arizona/footage/ff-samples" \
FRAMEFOUNDRY_VISUAL_ONLY=1 .venv/bin/python -m pytest tests/test_arizona_demo.py -q -s
```

The test is opt-in and never downloads private footage in ordinary CI. Offline
tests remove provider keys by default; run the demo target for real model calls.

## CLI

```sh
.venv/bin/framefoundry run --footage-dir /path/to/clips \
  --goal 'Create a 30 second vertical family-trip teaser' --episode trip-001
.venv/bin/framefoundry eval trip-001
.venv/bin/framefoundry eval trip-001 --judge
.venv/bin/framefoundry rate trip-001 --score 4 --publish-without-edits yes
.venv/bin/framefoundry eval --calibration
.venv/bin/framefoundry scoreboard
```

Legacy `python -m pipelines.ffmpeg.cli --footage-dir ... --goal ...` still works.
`--episodes-root`, `--evals-root`, and optional `--eval-labels` select artifact
locations. `FRAMEFOUNDRY_EVALS_ROOT` selects automatic evaluation history location.
`--no-planner-retry` disables the one optional evaluation-driven replan.

Every accepted terminal run, including early failures and interruptions, writes
an eval report and appends a locked/fsynced line to `evals/scoreboard.jsonl`.
Persistence failures fail visibly. Explicit re-evaluation appends history, while
summaries deduplicate Episode/render quality and cumulative Episode cost.
Human ratings in `evals/human-ratings.jsonl` are tied to the exact render hash.
Missing paired ratings, undefined kappa and absent scores are unavailable, not
passes. Targets: >=80% agreement within one point and kappa >=0.6.

## Models, budgets and cache

Credentials are read only from environment: `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`. Provider choice defaults to Anthropic if both exist; select
`FRAMEFOUNDRY_LLM_PROVIDER=openai` explicitly when wanted. Defaults: `gpt-4.1`
or `claude-sonnet-4-5`; set `FRAMEFOUNDRY_LLM_MODEL` to override. Custom models
need both `FRAMEFOUNDRY_INPUT_USD_PER_MILLION` and
`FRAMEFOUNDRY_OUTPUT_USD_PER_MILLION`. Model rates are estimates, not invoices.

- `FRAMEFOUNDRY_EPISODE_API_BUDGET_USD`: default **0.25** (target, not measured quality guarantee).
- `FRAMEFOUNDRY_EPISODE_TOKEN_BUDGET`: default **30000**.
- `FRAMEFOUNDRY_PROJECT_SPENT_USD`: explicit initial spend when creating the ledger.
- `FRAMEFOUNDRY_PROJECT_BUDGET_DIR`: durable project ledger location, default repository root.
- `FRAMEFOUNDRY_CACHE_DIR`: Planner cache root, default `.inference-cache/`.

Reserve request-specific text/image/output allowances before dispatch. Failed or
unknown calls keep conservative *budget reservations*; those reservations are
not reported as billed spend. Returned overrun is persisted and blocks further
calls. Budget errors terminate rather than quietly entering heuristic fallback.
Project reservations share a $500 ceiling across output roots; changing roots
must not reset the project ledger. Current rates/charge completeness stay visible.

The Planner receives at most eight locally pre-ranked candidates, <=160-character
snippets, compact quality/audio evidence and **one tiled JPEG**, never raw video
or full transcripts. Intervals must map wholly into supplied candidates. Valid
source-backed decisions are cached by footage/Goal/model/configuration; compatible
reruns add no Planner usage. Fallback remains a heuristic, not semantic vision.

The eval judge receives only the Goal, a reasoning-free edit projection and one
shot-labeled tiled sheet from the final cut. It never sees full transcripts or
Planner rationale. Strict anchored 1–5 criteria: goal fit, moment importance,
story arc, variety, opening and ending, with cited shot IDs, weakest shot, one fix
and verdict. Temperature is zero. Unsupported Goals are flagged partial and
cannot receive high goal fit. Ordinary Episodes have **one judge call maximum**;
failed gates skip it. Cache uses footage+Goal+render+prompt/model configuration.
A changed render after a judge-driven retry has final J/Q unavailable—no stale
score and no second judge call.

Sponsor ElevenLabs narration is **not implemented in this deadline milestone**.
`ELEVENLABS_API_KEY` alone does not make a call. Eval accounting can consume a
future `tts-usage.json` ledger and includes LLM+TTS model spend, separate from
unknown local compute expense.

## Evals and QA

Tier 0 independently measures loudness with `ebur128=peak=true`, word padding,
strict target ±10%, requested dimensions, constant fps/audio, black/frozen content,
repeated-shot perceptual hashes, all emitted JSON contracts and contact evidence.
Available source-edge clamps are disclosed. AAC normalization includes encoding
headroom and a bounded measured correction; delivered audio must still pass
**−14 LUFS ±1 dB and ≤−1 dBTP**, not merely carry that metadata.

Tier 1 measures Laplacian sharpness, coarse whole-pixel motion jitter, exposure,
pacing, scene/day coverage, variety and opening visual energy. Methods and missing
components are reported. Thresholds are initial synthetic-tested heuristics, not
calibrated artistic-quality guarantees; legitimate similar static shots can fail
the duplicate/freeze gates.

Tier 2 normalizes rubric mean as J=(mean−1)/4. Optional golden labels supply H,
recall of must-include source moments; a label counts when at least 50% overlaps a
selected shot. No human Arizona labels are fabricated. Composite:

- Q=G×(0.5J+0.3D+0.2H).
- Without labels: Q=G×(0.625J+0.375D).
- Failed/not-run gate: G=0 and Q=0, including undersized partial clips.
- Passed gates with missing J or D: Q unavailable.
- Quality/$ exists only for known positive LLM+TTS spend and available Q. Zero or
  unknown spend yields an unavailable ratio, never infinity or invented dollars.

Reports retain observed tokens, budget status, cached historical usage and PR
attribution (`FRAMEFOUNDRY_PR`). `qa-report.cost.usd` includes still-unknown local
compute; `eval-report.model_cost` isolates measured/estimated/unknown model spend.

Independent zero-paid-call revalidation:

```sh
.venv/bin/python pipelines/ffmpeg/qa_validator.py --episode-dir episodes/trip-001
```

## Golden set, A/B and CI

`evals/golden/manifest.json` references external Arizona media and generates
synthetic known highlights. Cases cover 30-second vertical, 60-second horizontal,
kids-only, landscapes-only and impossible beach sunset from desert footage.

```sh
.venv/bin/framefoundry eval --golden --case synthetic-vertical-30s \
  --case synthetic-horizontal-60s --work-dir /tmp/framefoundry-golden
.venv/bin/framefoundry eval --golden --golden-data "$HOME/.cache/framefoundry/arizona" \
  --download --judge --planner-llm
.venv/bin/framefoundry eval --pairwise trip-a trip-b --judge
.venv/bin/framefoundry scoreboard --baseline /path/to/main-scoreboard.jsonl
```

External download/model use is explicit. Missing assets/expected flags produce
unavailable/failing results, not skipped passes. Pairwise uses an isolated capped
allowance charged to the project budget: two swapped-order calls maximum; count
a win only when the same edit wins both. Comparison edits must have matching
Goals, usable contact evidence and gate-passing evals.

The baseline comparator fails a >5% relative mean-Q drop and refuses missing,
unscored, incomplete-case or configuration-incompatible comparisons. **An actual
main quality baseline and fully automated real-model PR-vs-main golden job are
not established yet.** The included PR workflow runs offline/mock tests only;
provider secrets must not be exposed to untrusted PR code. Mocked scores prove
mechanics, not artistic quality or calibration targets.

## Verify

```sh
make test
```

Six real seeded defects (loudness, black gap, wrong aspect, mid-word cut,
duplicate shot, freeze) are caught by offline self-tests. Tests create temporary
FFmpeg media and mocked providers; generated media never enters Git. Real 480p
Arizona demo proves media/audio execution only, not original iPhone 4K HDR/VFR
color fidelity, genuine ASR, live model judgment or human publishability.
