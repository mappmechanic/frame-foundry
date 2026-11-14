# FrameFoundry

A local footage-to-clip CLI: deterministic media processing and independent QA,
with bounded multimodal editing judgment. No media binaries or secrets belong in Git.

## Install (Mac or Linux)

Install FFmpeg/ffprobe (`brew install ffmpeg` on Mac), then from this checkout:

```sh
# uv lives here on the demo Mac; add it to PATH if needed.
export PATH="$HOME/.local/bin:$PATH"
make setup
```

`make setup` creates the Python 3.13 venv and installs dependencies, preferring `~/.local/bin/uv` when present (`UV=/path/to/uv` overrides
it). **Setup recreates `.venv`**, replacing an old Python 3.14
environment; do not run them during an active Episode. Python 3.10–3.12 are within
the package range, but the verified demo/CI runtime is 3.13.

The console entry point is `.venv/bin/framefoundry`. Editable installation from
this checkout keeps the bundled `schemas/` and `contracts/` available. Lightweight
verification and CI use **Python 3.13** on Linux and Mac. FFmpeg is an external prerequisite.

### Supported Python / OTIO runtime

Python is restricted to `>=3.10,<3.14` in `pyproject.toml` and `3.13` in
`.python-version` selects **3.13** for demo setup; OTIO remains pinned to 0.18.1. Mac Python 3.14 source builds
failed native dictionary conversion **and built-in adapter-manifest JSON loading**
with `RuntimeError: bad any cast`. Metadata normalization alone does not repair
that binary/runtime incompatibility. Use a fresh Python 3.13 environment (above),
not the old Python 3.14 venv. Linux-only Python 3.14 success is not Mac support.
The assembler still normalizes and round-trips typed metadata through native JSON.


### Real word ASR and scoring

The default `--asr-provider auto` prefers provisioned offline WhisperX. If its
package/assets are unavailable and `OPENAI_API_KEY` is configured, it uploads
normalized source audio to OpenAI `whisper-1` with `verbose_json` and word-level
timestamps. `--asr-provider openai` explicitly selects that route; `local` disables
cloud fallback. Missing keys/assets, failed requests, missing word times, or
incomplete source coverage do **not** become a verified empty transcript.
Originals remain immutable; delayed audio keeps its common source/video origin.

For local ASR, install `requirements-whisperx.txt` and provision
`FRAMEFOUNDRY_MODEL_DIR/transcription/`, `alignment/`, and cached VAD assets.
Runtime local-model loading blocks downloads. Only missing dependencies trigger
fallback; invalid local analysis is not silently replaced. No key is loaded from
`.env` by the application—export a trusted local file into the process yourself.

Transcription has `role=transcription` in `llm-usage.json`, duration-based estimated
cost, and **null/not-applicable LLM token counts**. The configured assumed rate is
`FRAMEFOUNDRY_ASR_USD_PER_MINUTE` (historical default **$0.006/min**, not a verified
current tariff/invoice). Rough Arizona input-only estimate: 263.3 s is ~$0.0264
before per-source rounding, Planner/judge, unknown failed charges, and local compute.
Check your account's rate and override it if necessary. Reservations share the
$0.25 Episode/$500 project limits; network failures retain unknown reservations.
Successful content/model-keyed transcripts cache across Goals/Episodes, with no
new charges or usage rows on cache hits. Reused historical charges are not billed
again in quality-per-dollar. Eval reports separately show `asr_usd` and include
transcription in total model cost.

`--visual-only` is an explicit **no-ASR diagnostic mode**, not the scoring demo.
It does not certify speech integrity: word gate `not_run`, G=0/Q=0, no judge.
Whisper-1 zero-length words receive up to 20 ms within the next onset/source bounds;
equal onsets share a protected interval, and touching/overlapping neighbors are
normalized without discarding their speech coverage. Reversed or genuinely
out-of-range times remain invalid. Successful source caches remain reusable.
If an OpenAI source still fails ASR, the run continues with no word entries for
that source, keeps its original audio, and records `status=degraded` plus a
source-specific coverage limitation. This is **not verified silence**: the
word-integrity gate remains unavailable, G/Q=0 and J unavailable rather than
claiming a fully scored clip. Budget/configuration/dependency failures still stop.

Remove that flag to use real ASR and make word/judge gates eligible. No fixture
transcript or fabricated score is substituted for ASR or a live judge.


## Arizona demo (26 real clips, private release)

Load your trusted local `.env` into the process environment; the providers do not
read or log credential files. Do not print the keys or enable shell tracing.

```sh
set -a
source .env
set +a
export FRAMEFOUNDRY_LLM_PROVIDER=openai
# Real ASR: local WhisperX if provisioned, otherwise configured OpenAI:
make demo-arizona ARGS='--episode arizona-demo --asr-provider auto'
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
Failed QA/evaluation renders retain `diagnostic.mp4` with a diagnostic path
in the QA report, not an advertised finished clip. Portrait output center-crops to fill 1080×1920 (no landscape letterbox); this
can remove edge subjects, and does not promise subject tracking. Teaser planning
targets 2–3 second shots, while genuine speech protection can require longer
quotes. Boundary/judge tiles match portrait content with readable labels; no extra
model images/calls were added. Default target is 45 seconds at 30 fps; a Goal can request 16:9
(1920×1080) and an explicit 5–120 second target. Originals are never modified. CLI completion prints G, Q, J, D and
`quality_per_dollar` from `eval-report.json`; missing scores remain null.

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


### Arizona short-shot policy and CPU cues

Run `make setup` after
pulling to install CPU-only OpenCV. Bundled Haar faces/HOG people process at most
three existing keyframes per source, <=320px, one thread and OpenCL disabled;
no model downloads. Counts are detector cues, not verified subjects. Sparse
motion also includes camera movement/lighting, not action recognition. A local
26-clip/78-frame detector-only benchmark took 3.18s; the exact Mac runtime remains
to be measured (full pipeline includes ASR/render time).

Normal refinery runs create short, fully protected speech phrases (<=3s) and
b-roll windows from every source. Connected speech longer than 3s is not forcibly
split: choose other safe footage, never weaken 80ms/120ms word protection.
The Planner still sees eight rows and one image, with optional evidence-backed
people/action/scenic-wide/interior/screen labels in the same call. Full source,
usable-window and selected duration are reported separately. Local duration fill
can add uninspected footage and enforce a detector/high-motion hook; absent hook
evidence is explicit partial coverage, not an invented detection.

Arizona caps the shared ASR/Planner/judge allowance at **$0.10**, retaining stricter
configured limits. One concrete judged fix triggers the single Planner repair
when J<=0.5 or opening<=2, even if verdict is pass. Original judged scores/hash are
retained in attempt evidence; changed renders do not inherit stale final J/Q,
and do not receive another judge call. Use `--no-planner-retry` for an explicitly
single-plan comparison if you need a scored unchanged cut. Zero/unknown spend
and missing semantic scores remain unavailable, never fabricated efficiency.


### Rerun from a checkout without a venv (optional intro VO)

```sh
make setup  # creates/recreates Python 3.13 .venv and installs all dependencies
set -a; source .env; set +a
export FRAMEFOUNDRY_LLM_PROVIDER=openai
make demo-arizona ARGS='--episode az-0600 --asr-provider auto --voiceover'
.venv/bin/framefoundry scoreboard --table
```

Omit `--voiceover` for unchanged source-audio-only output. Explicit VO needs
`ELEVENLABS_API_KEY`; default is stock Rachel (`ELEVENLABS_VOICE_ID` overrides),
model `eleven_flash_v2_5`. The short two-line intro is generated locally from the
saved narrative's Goal/selected moments; no extra LLM call or invented locations.
A text/voice/model/settings hash caches successful audio. One paid intro call per
Episode, no hidden HTTP retries; failed/possibly-paid charges remain visible.
Original audio ducks by 10dB during the intro; normalization and independent
loudness/peak gates still apply to the final mix. VO must fit fully inside the
video, never truncated to force the target. No voice cloning or new music.

TTS is priced at a **configurable assumed $0.30/1000 submitted characters**, not an
account tariff or invoice; set `FRAMEFOUNDRY_TTS_USD_PER_1000_CHARS` to your actual
plan/model estimate. All ASR/LLM/TTS reservations share Arizona's $0.10 cap.
`tts-usage.json` and `eval-report.model_cost.tts_usd` retain separate TTS spend;
quality-per-dollar includes it. Live ElevenLabs quality/billing need your Mac run.

CLI `status=PASS` means QA and all eval gates pass. `result=partial` is retained
as a content-coverage label and `coverage_note` spells out its limitations; it
is not a contradiction of technical success. Missing eval evidence prints
`QA_PASS_EVAL_INCOMPLETE`, not a false complete pass.

`duration_evidence` in the CLI summary distinguishes raw `source_duration_ms`
(and `source_count`), the source-bounded union `usable_short_windows_ms`, and
`selected_raw_duration_ms` / independently measured `selected_encoded_duration_ms`.
Usable windows can exclude long connected speech or tiny gaps; they are not a
claim that all raw footage was selected or semantically inspected. A shortlist
is preview evidence only, never total available footage. Partial-coverage notes
include the locally computed totals, replacing unverified model duration claims
while retaining missing-content and ASR-coverage limitations.

Intro VO now uses optional descriptive `subject_tags` from the existing Planner
call, never an extra vision/LLM call. The closed vocabulary excludes named places.
Only tags/labels with a retained selected-frame timestamp, canonical frame hash,
and matching output-crop view survive into `segments[].semantic_evidence`.
Screen depictions do not count as visiting an event. The intro is 1–2 lines,
under 90 characters; uncertain/missing subjects use a neutral disclosed fallback.
No locations or specific events are extracted from the Goal or filenames.

### Static demo scoreboard (no server)

```sh
.venv/bin/framefoundry scoreboard --html out.html
# Optional, presenter-controlled assumption; omitted means “not measured”.
.venv/bin/framefoundry scoreboard --html out.html --assume-manual-minutes-per-episode 15
open out.html  # Mac; any browser can open the file directly
```

The single HTML file embeds its CSS and SVG chart; it makes no page-view network
requests. Export from the Mac containing the live episodes/evals to populate the
az-0500 → az-0550 → az-0600 → az-0700 trend. Missing artifacts remain N/A.
Factory history combines current git/optional generation-time GitHub data with
an audited WO timestamp snapshot. Ticket-to-PR is explicitly **first recorded PR
title coverage**, not completion; WO-7 uses its later title-addition event.
Render QA rate uses known latest QA artifacts; shipped means a hash-matching local
export passing QA and evaluation gates, not a social publication or merged PR.
AI cost per shipped episode includes failed-episode cohort overhead and exposes
unknown billing. The manual-minutes flag estimates gross manual editing effort
avoided at N minutes × shipped episodes / 60, assuming zero hands-on automatic
editing and excluding prep/review/debugging; it is not measured engineer labor.
