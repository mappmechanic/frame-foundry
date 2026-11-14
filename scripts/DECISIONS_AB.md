# WO-12 integration slice (2026-11-14)

Experimental **advisory bounded selection hints**, not full WO-12 acceptance.
`llm` stays default. Planner/VO generation, ordinary Judge, Q weights, speech
protection and hard gates are unchanged. The existing top-eight shortlist is
used; adapter maximum is 66 candidates / 396 mixed questions (400 hard ceiling).
Section questions require retained in-window image evidence; their answers do
not automatically attach overlays or prove named locations. SDK retries are zero.
Episode reservations and caps include failures; project admission stays enforced.

No paid calls or real benchmark were performed. Optional install is
`python -m pip install -e '.[decisions]'` (Python 3.10–3.13). No key is requested
or included here. Missing access records a heuristic fallback, not inference.
No budget settings are raised. Baseline default cache identity is unchanged.

Official references verified before implementation:
- https://developers.openai.com/api/docs/guides/decisions
- https://developers.openai.com/api/docs/models/gpt-6-luna
- https://github.com/openai/openai-python/releases/tag/v3.26.0

Standard dated rates ($/M): other input .10, cached reads .01, cache writes .125,
output 0. Cache categories are disjoint portions of reported input. Missing
counts remain null; missing input/read/write usage prevents computing cost.
Computed costs are estimated, not invoices. Regional premiums are not modeled.

## Mac commands (future keyed acceptance; these can spend money)

From the integrated repository, with real media at `$ARIZONA_MEDIA`, existing
caps and already-configured provider access. Do not use `--visual-only` for
acceptance. Local ASR must already be available; voiceover is off in both arms.
These commands are for a future authorized run, not a claim that access exists.

```sh
export PYTHONPATH="$PWD"
GOAL='Create a 30 second Arizona family-trip teaser'
FRAMEFOUNDRY_CACHE_DIR="$PWD/.ab-cache/llm" .venv/bin/python -m pipelines.ffmpeg.cli run \
  --footage-dir "$ARIZONA_MEDIA" --goal "$GOAL" --episode wo12-llm \
  --episodes-root "$PWD/episodes" --decider llm --duration-seconds 30 --aspect 9:16 \
  --asr-provider local --overlays --section 'Halloween in Uptown Sedona' \
  --section 'Hilton Sedona Resort at Bell Rock'
FRAMEFOUNDRY_CACHE_DIR="$PWD/.ab-cache/decisions" .venv/bin/python -m pipelines.ffmpeg.cli run \
  --footage-dir "$ARIZONA_MEDIA" --goal "$GOAL" --episode wo12-decisions \
  --episodes-root "$PWD/episodes" --decider decisions --duration-seconds 30 --aspect 9:16 \
  --asr-provider local --overlays --section 'Halloween in Uptown Sedona' \
  --section 'Hilton Sedona Resort at Bell Rock'
```

Use fresh Episode/cache directories for a disclosed cold-cache policy. Match
actual generative Planner settings between arms. Canonical comparison requires
actual `gpt-4.1` Judge evidence; changing a config label cannot make a different
Judge qualify. If the normal provider also sets Planner to `gpt-4.1`, disclose
that baseline choice; do not pretend it matches a prior Anthropic-planned run.
Fill two copies of `decisions-ab-config.example.json` from actual run settings,
including real `footage_hash` and `planner_model`; placeholders cannot qualify.

```sh
PYTHONPATH="$PWD" .venv/bin/python scripts/decisions_ab.py \
  --llm episodes/wo12-llm --decisions episodes/wo12-decisions \
  --llm-config /tmp/wo12-llm-config.json \
  --decisions-config /tmp/wo12-decisions-config.json \
  --output /tmp/wo12-comparison.json
```

## Parent/frontend contract

Read `decisions-report.json` for requested/effective mode, fallback, typed answers,
cache identity and per-call latency/usage. `decisions_ab.arm/compare` and the script
are read-only: `arms.llm` / `arms.decisions` contain Episode identity, Q, J, complete
total model cost, Q/$, elapsed seconds and provenance. Missing/failed arms stay
visible. `status` is qualified/unqualified **eligibility only**; `default_activated`
is always false. Configurations must be supplied from retained settings, not
inferred from matching Q. Frontend wiring is parent-owned and not implemented.

Remaining acceptance: direct bounded keep/drop/section application beyond hints,
separately bounded recovery choice and exact-render J invalidation wiring,
activation/persistence of a qualified default, complete live Arizona matched A/B,
frontend display, optional separately budgeted Decisions Judge experiment, real
SDK/account/region compatibility and per-stage timing beyond request latency.
This PR reviews the safe slice only; WO-12 acceptance is incomplete. No saving,
promotion or rendered acceptance claimed.
