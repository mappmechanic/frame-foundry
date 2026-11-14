---
name: rough-cut
description: Plan source-preserving rough cuts, verified dead-air removal and complete retake selection before assembly.
---

# Rough cut

## Authority and availability

Original FrameFoundry instructions from the approved WO-17 scope and existing
`pipelines/ffmpeg/assembler.py`, `teaser_edit.py` and strict narrative schema.
The [production runtime](../../../docs/PRODUCTION_POLICY.md) implements supported rules in `production.py` only with
`--production`; ordinary edits remain unchanged. Exact complete retakes are
implemented; qualified semantic retakes remain a **declared target**, not a
license to merge statements. Review the actual `editing_policy` receipt.
See the [availability matrix](../../../docs/skills/README.md) and
[provenance manifest](../../../docs/skills/provenance.json).

## Plan before assembly

1. Bind each proposed cut to source ID, original start/end, timed words and the
   measured evidence. Never infer silence from an empty/missing ASR transcript.
2. Remove dead air **>350ms only within verified silence**, subtracting all
   protected word intervals first. The removable interval must still exceed
   350ms after that subtraction. Keep breaths, intentional pauses and ambiguous
   low-level speech; waveform/activity cues alone do not establish semantic silence.
3. Protect every retained word with **80ms pre-roll / 120ms post-roll**, clamped
   to the source. Use the assembler's fixed-point neighbour protection, not a
   fresh arithmetic cut that damages a neighbouring word. Source-edge shortages
   are limitations, never fabricated handles.
4. Identify retakes only within a **30s source-local window** of the same statement.
   Keep the **final complete exact or qualified take**: exact means the same
   complete words (case/spacing/punctuation normalization only); qualified means
   evidence confirms the same claim/question with all meaningful qualifications
   preserved. Do not collapse negations, different numbers or changed meaning.
   An incomplete final attempt does not replace the last complete take. When
   identity/completeness is uncertain, keep both and record the limitation.
5. Retain whole statements in their own source context. Do not stitch syllables,
   merge unrelated claims, or stretch protected speech to force a duration target.
6. Handoff source intervals, protection checks, proposed removals and reasons to
   assembly. Apply only through the parent's supported opt-in policy; do not add
   undocumented flags/fields to `schemas/narrative-spec.schema.json`.

## Verify and report

Reopen the serialized timeline, reconcile the original-source ranges, and check
padding after coalescing. Missing silence/retake evidence means **not applied**,
not PASS. Keep before/after source mapping; never claim output improved because
this file was installed. Next: [vlog pacing](../vlog-pacing/SKILL.md).

## Attribution / license

No upstream skill text or code copied or adapted here. This original guidance is
covered by the repository [MIT notice](../../../LICENSE). No content from
appariciojunior, remotion-dev/skills or OpenMontage was cloned, read or copied.
The manifest lists the separate MIT references used by other curated skills.
