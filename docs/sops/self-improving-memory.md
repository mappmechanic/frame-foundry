# WO-15 — Local-first edit memory and controlled experiments

**Backend complete offline; no live improvement claim.** Ordinary Episodes are
memory-off unless `--memory` is explicit. No lesson-generation LLM call is added.
Live HydraDB readiness and real Arizona quality proof are pending credentials and
operator-authorized execution. Do not execute paid examples in the workspace.

## Precise Mac commands

Run from the repository using its existing Python 3.10–3.13 environment. Keep keys
out of commands, Git, reports and lesson payloads. Use a fresh ID for every run:

```sh
PY=.venv/bin/python
ID="$(date -u +%Y%m%d-%H%M%S)-$($PY -c 'import secrets; print(secrets.token_hex(2))')"
FOOTAGE="$HOME/.cache/framefoundry/arizona/footage/ff-samples"
PRIVATE="$HOME/.local/share/framefoundry/memory"
GOAL='Things to do in Sedona with Kids'
```

An ordinary explicit memory Episode (may dispatch configured paid providers;
execute only with approved credentials/headroom):

```sh
$PY -m pipelines.ffmpeg.cli run --footage-dir "$FOOTAGE" --goal "$GOAL" \
  --episode "memory-$ID" --episodes-root episodes --evals-root evals \
  --memory --memory-backend local --memory-root "$PRIVATE" \
  --asr-provider auto --duration-seconds 30 --aspect 9:16 --overlays \
  --section 'Halloween in Uptown Sedona' --section 'Hilton Sedona Resort at Bell Rock' \
  --no-planner-retry
```

`--memory-backend auto` uses configured HydraDB or local; `hydra` requests HydraDB
but also falls back locally on absence/failure. Install optional provider support
with `pip install -e '.[memory]'` (`hydradb-sdk>=2.1.7,<3`; mocked 2.1.7 contract).
No database provisioning is attempted. SDK ingestion uses `infer:false`, canonical
`metadata.lesson_id`, and bounded `context.status(database, collection, ids)`.

Create a **pending** experiment with N on and N off identities but **no dispatch**:

```sh
$PY -m pipelines.ffmpeg.cli improve --footage-dir "$FOOTAGE" --goal "$GOAL" \
  --experiment "az-$ID" --iterations 3 --episodes-root episodes --evals-root evals \
  --seed-root "$PRIVATE" --duration-seconds 30 --aspect 9:16 --overlays \
  --section 'Halloween in Uptown Sedona' --section 'Hilton Sedona Resort at Bell Rock'
```

Pending/failed/unavailable commands exit 1, not success; use another fresh ID for
execution. For an **authorized Mac operator only**, run the same recipe with
`--execute --paid`. The reported $74 used and ambiguous $200–300 approval are
handled conservatively with a $200 total ceiling. Change explicit authorization
to $300 only if that total is confirmed. Neither command changes/increases the
existing global project cap. Preserve Arizona's existing $0.10 shared Episode
cap, or a stricter configured cap:

```sh
ID="$(date -u +%Y%m%d-%H%M%S)-$($PY -c 'import secrets; print(secrets.token_hex(2))')"
export FRAMEFOUNDRY_EPISODE_API_BUDGET_USD="$($PY -c 'from pipelines.ffmpeg.llm_provider import episode_budget_usd; from decimal import Decimal; print(min(episode_budget_usd(),Decimal("0.10")))')"
$PY -m pipelines.ffmpeg.cli improve --footage-dir "$FOOTAGE" --goal "$GOAL" \
  --experiment "az-$ID" --iterations 3 --episodes-root episodes --evals-root evals \
  --seed-root "$PRIVATE" --execute --paid --authorized-total-usd 200 --prior-spend-usd 74 \
  --asr-provider auto --duration-seconds 30 --aspect 9:16 --overlays \
  --section 'Halloween in Uptown Sedona' --section 'Hilton Sedona Resort at Bell Rock'
```

Six Episodes at $0.10 reserve $0.60, plus existing reservations and the reported
$74 floor, **before any paid dispatch**. All 2N identities are atomically reserved
in the shared ledger against min(explicit authorization, existing global cap).
Existing initial spend cannot be decreased; Episode/provider limits are unchanged.
Unknown failed-call usage stays reserved and is not fabricated as billed zero.
Unavailable headroom/provider configuration produces pending/unavailable rows.
`--voiceover` applies equally to both arms and requires configured ElevenLabs and
sufficient unchanged allowance. `--planner-retry` explicitly enables the existing
bounded repair; defaults off for controlled experiments.

Offline execution: add `--execute` **without** `--paid`, with a synthetic
`--transcript-fixture` or preprovisioned local ASR. This forces provider=None and
local ASR, disallows paid VO/Decisions, and never dispatches inherited provider
keys. Missing assets yield failed/unavailable data, not invented results.
`python -m scripts.improve` exposes the same command. Visual-only is not accepted
as a shortcut to speech-verified learning results.

## Isolation and cache behavior

* Read and validate the entire local seed once; freeze it under
  `evals/learning-curves/<id>/seed/lessons.jsonl`; copy into that experiment's
  `memory_on` scratch. Later concurrent/shared-store changes cannot alter baseline.
* Experiment memory is **local-only**, even with Hydra configured, to prevent a
  shared remote collection changing the seed. Ordinary Episodes can use Hydra.
* Off arm has `memory=False`, no memory root and no lesson read/write/artifact.
  Both arms disable implicit prior-judge `repair_latest` so history cannot leak.
* Separate per-arm Planner/ASR/VO cache environments and separate per-arm Judge
  eval/cache directories start empty. Valid matching cached inference can be
  reused only within the same arm. Every iteration gets fresh Episode IDs and
  reruns source analysis, render and evaluation. Reports are never copied;
  historical cached charges are never charged again.
* Process environment is restored after each sequential arm. Concurrent
  experiments must use separate processes, not threads mutating `os.environ`.
* Experiment scratch/seed is isolated and private. `latest.json` is an atomic
  snapshot, never a job trigger. Opening Studio/site never starts an experiment.

## Deterministic evidence and guardrails

Lessons require persisted QA/eval/spec, with actual OTIO `read_shots()` identities
and `selection_ids` for weakest intervals. Coalesced rendered shots are not
narrative segment indices. Exclusion requires the exact same footage fingerprint
and exact source interval in >=2 distinct Episodes; shot IDs are never globally
excluded. Opening preference requires recorded opening >=4/5 and retained
in-window semantic types; only an existing safe hook may be preferred, without
moving protected payoff tail, changing VO, or creating adjacent coalescing cuts.
If preferences cannot be satisfied safely, the limitation is disclosed.

Selection is <=5 lessons and <=400 UTF-8 bytes for the entire compact
`edit_memory` field envelope, a conservative <=400-token bound. It is structured
`untrusted_historical_observations` data in the existing single Planner request,
not instructions or source proof. The cache key includes the exact payload.
No extra LLM call; no inferred scene labels, relaxed 80/120-ms speech protection,
semantic/source/VO-hash/media/audio/contact-sheet gates, or spending increase.

Metadata preserves Episode/eval/Goal/footage, available rubric scores/evidence,
J/Q, known/total cost and completeness, effective Decider, source interval and
recorded outcome. Unrecorded rationale is `not_recorded`; outcomes stay `unknown`.
Comparable pair deltas disclose changed metadata and observed metric movement,
not helped/hurt causal claims. Canonical IDs deduplicate repeated persistence.
Local writes are locked, bounded and atomic; remote payloads use the provider
allowlist (no source paths, transcripts, credentials or media). Failures/timeouts
are disclosed auxiliary no-ops, not grounds to invalidate an Episode.

## Exact memory-only shared-file hooks for parent integration

* `contracts.EpisodeContext`: `memory_enabled=False`, `memory_backend='auto'`,
  `memory_root=None`, `memory_selection=None`.
* `cli.run_episode(..., memory=False, memory_backend='auto', memory_root=None,
  reuse_history=True)` passes memory fields into context. `reuse_history=False`
  disables implicit `repair_latest` in both controlled arms. CLI flags:
  `--memory`, `--memory-backend auto|local|hydra`, `--memory-root`.
  `main(['improve', ...])` delegates to `scripts.improve.main`.
* `planner.plan_teaser`: `memory_loop.prepare(ctx,analysis)` before shortlisting;
  filter learned windows; add `edit_memory` to payload and cache identity; record
  actual used IDs. `apply_preferences(ctx,spec,analysis)` runs before final overlay
  attachment and validation, with safe limitations. Core does not modify overlays.
* `evaluator.evaluate_episode`: optional `report['memory']=report_memory(root)`
  before schema validation; old/off reports require no new fields.
  `persist_report`: after eval JSON and scoreboard persistence,
  `write_completed(report,episode_dir)` writes deterministic lessons fail-open.
* `eval_gates.KNOWN_SCHEMAS`: **one registration only**,
  `memory-report.json` → `memory-report.schema.json`. Otherwise the strict
  artifact gate correctly rejects the new file. No threshold/logic relaxation.
* `contracts/eval-report.schema.json`: optional private memory provenance;
  `schemas/memory-report.schema.json`: strict emitted selection contract;
  `contracts/learning-curve.schema.json`: experiment file contract.
* `pyproject.toml`: optional provider `memory` extra and `scripts*` namespace
  package inclusion for installed improve delegation. No mandatory SDK.
* Core owns `memory_loop.py`, paired runner, memory tests/docs and these narrow
  shared-file hunks. It does **not** change assembler/render/overlays/production
  policy/site. UI PR #7 is deliberately **not merged**. Its
  `learning_view.read_curve(evals_root)` / `public_projection(curve,allowed_ids)`
  consumes the file contract below.

## Learning curve schema 1.0

`evals/learning-curves/<id>.json` and `latest.json` are complete snapshots:

```text
schema_version: "1.0"
experiment_id, status: pending|recorded, created_at (timezone-aware), config_hash
arms.memory_on[], arms.memory_off[]:
  iteration, episode_id, status: passed|failed|unavailable,
  J, Q, model_usd, known_model_usd, cost_status, elapsed_seconds, lessons_used[]
deltas[]: iteration, comparable, J, Q, model_usd (on minus off, or null)
limitations[]
budget?: authorized_total_usd, prior_spend_usd, episode_reserve_usd,
         project_forecast_usd
```

An arm passes only with persisted QA and all evaluation gates passing. Unknown
Judge/Q stays null; failed/unavailable arms never expose quality scores. Pair
comparability requires matching actual footage, Goal, narrative config, score
method/weights, gate thresholds and Judge provider/model. Cost deltas require
complete costs; known costs remain separate. Cached results report this
iteration's cost, not prior charges. Executed curves are observations, not proof
of improvement. Public export/privacy/presentation is WO-16-owned.

## Verification and remaining live gaps

Offline tests use only synthetic media and mocked transports. Full mocked Episode
passes -14±1 LUFS, true-peak and contact-sheet gates, then writes and retrieves
lessons in the single Planner call. The full paired test proves off/on cache and
lesson separation, fresh provenance, controlled cost and observed deltas. No
provider credentials, paid execution, real Arizona learning curve, Hydra readiness,
UI merge or deployment was performed. See the core PR for exact shard totals.

### Completed offline verification

Full suite: **893 passed, 1 skipped** in six isolated command shards, each bounded
with `timeout 540` in Linux (do not assume GNU `timeout` exists on macOS). The
longest shard was 306.72 seconds. The core shard is 67 passed in 64.09 seconds.
Additional targeted shared-hook verification was 43 passed; do not double-count
that targeted run in the full-suite total. Compileall and `git diff --check` pass.
PR #7's existing read-only `read_curve` / `public_projection` accepted the generated
mocked paired curve with four source Episodes and comparable deltas; UI not merged.

Reproduce the same file partition from the repository (one command per shard):

```sh
.venv/bin/python -m pytest -q tests/test_arizona_demo.py tests/test_arizona_recovery.py tests/test_assembler.py tests/test_audio_analysis.py tests/test_budget_limits.py tests/test_campaigns.py tests/test_candidates.py tests/test_closing_constraint.py
.venv/bin/python -m pytest -q tests/test_crop_eval_geometry.py tests/test_decisions.py tests/test_decisions_ab.py tests/test_demo_scoreboard.py tests/test_disk_safety.py tests/test_duration_coverage.py tests/test_eval_gates.py tests/test_eval_judge.py
.venv/bin/python -m pytest -q tests/test_eval_tools.py tests/test_evaluator.py tests/test_export_site.py tests/test_factory_history.py tests/test_free_repair.py tests/test_goal_judge.py tests/test_goal_parser.py tests/test_grounded_voiceover.py
.venv/bin/python -m pytest -q tests/test_llm_provider.py tests/test_local_server.py tests/test_local_ui_page.py tests/test_low_score_retry.py tests/test_openai_transcription.py tests/test_overlays.py tests/test_pipeline.py tests/test_pitch_site.py
.venv/bin/python -m pytest -q tests/test_planner.py tests/test_public_site.py tests/test_python_setup.py tests/test_qa_render_regressions.py tests/test_render.py tests/test_run_evals.py tests/test_scoreboard_html.py tests/test_short_speech_edit.py tests/test_teaser_pacing_crop.py tests/test_vision.py tests/test_visual_duration.py tests/test_voiceover.py
.venv/bin/python -m pytest -q tests/test_memory_loop.py tests/test_memory_store.py tests/test_improve.py tests/test_memory_episode.py
```

Shard results respectively: 108 passed/1 skipped (44.04s), 99 passed (108.87s),
201 passed (77.57s), 202 passed (306.72s), 216 passed (124.07s), and 67 passed
(64.09s). Synthetic/mocked results do not certify real footage quality or sponsor
performance. Verification logs are private workspace `/tmp/wo15-shard{1..5}.log`
and `/tmp/wo15-core-green.log`; no generated media or private curves are committed.
