---
name: vlog-pacing
description: Shape an evidence-backed opening and pacing while preserving long protected speech and face-safe framing.
---

# Vlog pacing

## Authority and availability

Use after [rough cut](../rough-cut/SKILL.md), before assembly. Existing
`pipelines/ffmpeg/teaser_edit.py` has source-bound short-edit selection; it does
not imply all rules below are implemented. The [production runtime](../../../docs/PRODUCTION_POLICY.md) applies supported
hook/pacing rules only with `--production`, disabled by default. Face-safe 112%
punch-ins remain a **declared target**; saved `editing_policy` receipts identify
actual changes and limitations.
[Availability](../../../docs/skills/README.md) / [provenance](../../../docs/skills/provenance.json).

## Editorial pass

- Bring a retained, evidence-backed **hook into the first 2s**. Select an actual
  intriguing statement/action, not an invented title, reaction or result. If no
  supported hook exists, disclose that instead of making one up.
- Aim for **1.5–4s shots**; vary lengths for action, context and payoff. Treat this
  as a target, not a cut mandate. The **protected long-speech exception** wins:
  retain complete speech with 80ms pre-roll / 120ms post-roll even when >4s and
  record why pacing was not applied. Do not split a protected phrase for rhythm.
- Adjacent source windows can coalesce in OTIO. Recheck actual serialized clips;
  do not count planner segments as final shot boundaries. Never reorder unrelated
  speech solely to avoid coalescing or satisfy the target.
- Proposed jump-cut style alternates neutral **100%** with **112% punch-in**,
  `anchor_y=0.33`, only with **face-safe evidence** in the final crop at both
  ends and throughout motion. Check eyes/chin/forehead and all relevant people.
  A sparse face-box cue is not landmark tracking. If evidence is absent, keep
  neutral framing and disclose it; do not claim this anchor/zoom is a current
  schema field or existing runtime behavior.
- Use the user's real scene, screenshots, numbers and claims. Labels must remain
  grounded in the retained interval. No decorative stock media or fonts fetched.

## Handoff / verification

Provide the selected hook's source/time proof, actual shot durations, exceptions,
retained word ranges and crop measurements. Re-rendering a saved plan with only
supported deterministic policy changes makes no new Planner call through
`scripts/reoverlay.py --production`; optional `--judge` requests fresh evaluation. Route overlays to
[SOP-03](../../../docs/sops/SOP-03-remotion-sparse-overlays.md) and output checks to
[visual QA](../visual-qa-validator/SKILL.md). Docs alone do not prove improved edits.

## Attribution / license

Locally authored rules from approved scope and FrameFoundry code; real-source story
and speaker-safe framing concepts informed by **artemnovitckii/claude-editing-skills**,
`skills/storytelling/SKILL.md` and `skills/behind-the-person/SKILL.md` at
`beff6b868a8c1bdfad8661bd8a298497e75568a3`.
MIT, **Copyright (c) 2026 Artem Novitckii**; full
[local notice](../../../docs/skills/licenses/artemnovitckii-claude-editing-skills-MIT.txt).
No upstream text/code/templates copied; all fonts/media excluded (OFL/third-party
exceptions retained). Original work: [repository MIT](../../../LICENSE).
