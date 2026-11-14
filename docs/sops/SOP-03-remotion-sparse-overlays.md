# SOP-03 — Sparse overlays: Measure -> Draw -> Verify

**Owner/scope:** WO-17 documentation and architecture only. Baseline `ea0887e`.
Parent/WO-18 owns versioned opt-in production policy/runtime; ordinary edits stay
unchanged by default. This SOP declares a target, not an installed renderer or a
claim that production output improved. No policy field or CLI flag is introduced.

## 0. Availability and authority

| Present in this checkout | Missing / declared target |
| --- | --- |
| `pipelines/ffmpeg/overlays.py`: Pillow RGBA PNG cards, FFmpeg active-window overlay enables, 250ms alpha fades, 8px slide, no drawtext/libass | Packed sparse-alpha browser renderer |
| `render.py`: original-source V1/A1 OTIO renderer, 30fps H.264/AAC delivery | Remotion/React rendering, transparent alpha clips on a V3 graphics track |
| `assembler.py:pad_word_interval`: source-bound 80/120ms protection | Versioned opt-in production policy and its persisted applied-change fields (parent-owned) |
| `contracts/overlay-style.json`, `overlays.py:verify`: geometry, readable holds, cut masks, font/style/plan/render hashes | Full measured eye/chin tracking and target safe-zone/cut+3 audit |
| `contact_sheet.py`: before/after cut samples at +/-34ms, QA sheet/index | Exact frame-numbered cut-1 through cut+4 overlay exclusion evidence |

The **present sparse active-window renderer** is not Remotion's packed alpha-range renderer and still encodes the base
video. `packages/remotion-studio/` and any Remotion project/package manifest are
**absent**; no Remotion commands are supported by this SOP. Existing SOP-01/SOP-02
files are **absent from the checkout**. The Knowledge Base lists PDFs with those
names; their contents were not read here. Do not invent their contents or paths.

Existing strict schemas and implementations win over external recommendations:
- `schemas/narrative-spec.schema.json`, `schemas/qa-report.schema.json`,
  `pipelines/ffmpeg/contracts.py` are current contracts; do not insert proposed
  V3, music, motion, policy or landmark fields into them.
- Retained speech: **80ms pre-roll / 120ms post-roll**, source-clamped and
  fixed-point extended around neighbouring protected words.
- Delivery audio: finite **[-15,-13] LUFS** and **true peak <=-1 dBTP** measured
  from encoded output. An image or installed skill cannot override these gates.
- Current large-card placement: top >=12%H, bottom including slide <=H/3;
  cap height >=ceil(**0.06H**) (portrait 1920H: **116px**, not font size 116),
  at most two lines / 42 chars per line, white text on 60% black backing.
- Current >=**2.5s full-strength readable hold** (75 frames at 30fps), after fades
  and hidden frames. Existing verification's .05H floor is not a design override.
- Current masks: **cut-1/cut/cut+1**. Preserve them even if a card spans separately
  grounded contiguous shots. Existing title must still fit <=3.5s total.

## 1. Measure

1. Probe the actual source and final intended output shape/fps. Retain source IDs,
   original paths/hashes, frame-time evidence, timed retained words, cut list and
   original-to-output mapping. Do not replace missing measurements with guesses.
   Current delivery uses 1080x1920 or 1920x1080 at constant 30fps.
2. Ground a section label in the retained transcript or verified in-window source
   frame. An operator's venue text is context, not visual proof. Keep uncertain
   sections omitted with reasons rather than invent scene labels or evidence.
3. In **final crop coordinates**, measure eyes, chin and the relevant face/head
   envelope at entrance, steady hold, exit, and motion extrema. Keep frame ID,
   time, crop transform and method/confidence. Check all people a card could cover.
   Measure the whole swept graphic rectangle (text, backing, shadow, animation).
   A Haar box/sparse cue is not eye/chin measurement or verified full tracking;
   empty detection is not evidence that no face exists.
4. Target safety excludes **9:16 bottom 22% / right 15% / top 10%** of output:
   y>=.10H, bottom<=.78H, right<=.85W; also keep the entire graphic inside frame.
   For **16:9 outer 10%**, x>=.10W, right<=.90W, y>=.10H, bottom<=.90H.
   These are proposed review requirements, not current runtime checks. Intersect
   them with current top-12%/upper-third gates, don't relax existing placement.
   If readable text cannot fit clear of measured eyes/chin, shorten/omit it with
   a limitation; do not silently shrink cap height or use guessed tracking.
5. Planning prerequisites: rough-cut dead air >350ms is removable only after
   verified silence excludes protected words; retakes within 30s keep the final
   complete exact/qualified take. Put a supported hook in the first 2s, target
   1.5–4s shots except protected long speech. Alternating 112% punch-in with
   `anchor_y=0.33` requires face-safe evidence. These are declared policy targets,
   not demonstrated behavior at this baseline; follow the corresponding skills.

**Measurement handoff:** source/crop-bound evidence, cut frame grid, label proof,
full graphic bounds, landmark measurements or an explicit unavailable reason,
proposed active ranges, and protected-word intervals. These are an evidence
checklist, **not** a new JSON schema. Parent must version any persisted policy.

## 2. Draw

### Current supported path (no new runtime)

Use existing assembly/overlay compilation. `compile_cards` places only grounded,
readable cards and records omissions. `card_png` uses an installed configured
font (`FRAMEFOUNDRY_OVERLAY_FONT` or current local fallbacks); no fonts are copied
into Git. `burn` uses FFmpeg overlay/fade with active-window enables and masks,
then `verify` binds output, plan, font and style hashes. No FFmpeg text filter is
needed. The current renderer performs an 8px slide, **not** the target spring.
Do not invoke inspected vendor scripts: they are not installed FrameFoundry tools.

### Declared Remotion / alpha V3 design (not installed)

1. Keep V1/A1 as original footage/dialogue; conceptually put sparse graphics on
   **V3** with transparent gaps. Current `render.py:timeline_ranges` accepts only
   V1/A1. A future adapter must own V3 acceptance/schema/compositing, never mutate
   today's strict timeline to make this design appear implemented.
2. Derive visibility as deterministic integer, half-open **[start,end)** frame
   ranges. Current clip-length rounding is `floor(seconds*30 + .5)`, minimum one
   frame; use the authoritative output cut grid, not float accumulation. Round
   measurement bounds conservatively so protected words are not lost.
3. Pack only active alpha ranges if worthwhile. Keep original global start/end,
   packed local start and frame count for every run. At packed local frame `p+i`,
   draw using its corresponding global frame `s+i`, preserving the card's original
   animation age. Never restart its spring phase or accidentally bridge a cut
   when runs are merged. Gaps outside active ranges must remain transparent.
4. Drive future React with current frame/fps and pre-resolved immutable inputs.
   Requested entrance spring: **damping=14, stiffness=120, mass=1**; clamp opacity
   and bounded transforms. No async React, network I/O, wall clock, timer-driven
   animation or unseeded randomness in components. Load/validate fonts and data
   before rendering. Pin toolchain/fonts; deterministic math alone does not
   promise byte equality across machines/Chromium/FFmpeg versions.
5. Entry/exit/settling time and all hidden frames are excluded from the >=2.5s
   steady hold. No continuous micro-motion during steady holds. Choose a finite
   settle/exit budget and demonstrate it with frames; spring config by itself
   does not prove a steady readable hold. Omit a card that cannot fit.
6. Preserve cut-1/cut/cut+1 masks. **Proposed stricter cut+3 exclusion:** keep
   graphics absent on **cut-1 through cut+3 inclusive**, permit re-entry no earlier
   than cut+4, and recheck hold length after all masks. This explicit conservative
   target is not today's +/-1 implementation. A future policy may omit a card
   instead of moving it into an unsafe grounded interval.
7. Export graphics only as **ProRes 4444** or **WebM alpha**, with preserved alpha
   and exact run dimensions/fps/frame counts. Verify decoded transparency and
   compositing compatibility; ordinary H.264 is not an alpha master. Place each
   run back at its global frame on conceptual V3, then encode the actual delivery
   H.264/AAC and verify again. No alpha media is produced by this documentation.

Before any future Remotion installation, separately review the package's own
commercial licensing; MIT licenses of these guidance repositories do **not**
license Remotion itself. No npm/npx install/render command is authorized here.

## 3. Verify

1. Reopen saved source-bound plan and timeline, validate strict schemas and the
   current QA gates. Reconcile actual cuts/word padding/duration with the decoded
   final result. Stale paths/hashes or failed required gates block delivery claims.
2. Retain the existing generated boundary contact sheet/index. For target overlay
   review, decode exact **cut-1, cut, cut+1, cut+2, cut+3, cut+4** plus entrance,
   hold and exit frames from the actual encoded output. Label frame numbers,
   timestamps, source mapping and render hash. At source/output edges mark absent
   samples unavailable; don't duplicate a clamped frame and call it a full audit.
   +/-34ms current samples alone cannot prove exact cut+3 exclusion.
3. Inspect all requested frames and moving envelopes: no prohibited alpha at
   exclusion frames, no subject eye/chin collision, no unsafe-zone overlap,
   readable cap height/backing/line count, sufficient steady hold and no orphan
   exit frames. Plan geometry alone is not encoded-pixel proof. If inspection is
   unavailable, say **pixels not inspected** and do not claim verified tracking.
4. Require encoded loudness/peak QA, speech padding, output dimensions/fps,
   source integrity and render binding as well as visual checks. Keep diagnostics
   for a failure; do not advertise production-ready output based on existence.
5. For the versioned opt-in target, parent runtime should retain policy version,
   configuration and actually applied changes, not merely a declaration that a
   rule exists. Distinguish **declared target / implemented / applied / not applied
   / unsupported / unverified** in handoff. Those are reporting concepts, not
   newly implemented schema fields. A default change needs evaluation evidence.
6. No improvement claim without matched baseline/policy source, goal, format,
   quality/judge scores, model cost and output/evaluation bindings. Missing or
   incomparable evidence is disclosed. No new paid Planner/Judge call is needed
   for documentation verification; saved deterministic re-render without a new
   Planner call remains a runtime target until supported.

## Supported commands and evidence limits

Use a prepared Python >=3.10,<3.14 environment with repository dependencies
(including OpenTimelineIO); the bare workspace Python may lack them. No setup or
dependency change is part of WO-17. From this checkout, safe checks are:

```sh
python -m pytest -q tests/test_curated_skills_docs.py
python -m pytest -q tests/test_assembler.py tests/test_overlays.py tests/test_audio_analysis.py tests/test_qa_render_regressions.py tests/test_short_speech_edit.py tests/test_teaser_pacing_crop.py
python -m pipelines.ffmpeg.cli --help
python -m pipelines.ffmpeg.qa_validator --help
```

For an **existing, complete episode** and its current artifacts (not a fabricated
example), `python -m pipelines.ffmpeg.qa_validator --episode-dir <episode-dir>`
is the supported independent QA entry point; inspect its help first. This SOP
does not run a new episode or paid evaluation. No Remotion/V3 command or missing
SOP-01/02 filename is invented. Passing documentation tests is not video QA PASS.

## Routing, provenance and licenses

- Plan: [rough-cut](../../.claude/skills/rough-cut/SKILL.md),
  [vlog-pacing](../../.claude/skills/vlog-pacing/SKILL.md).
- Assemble: [voice-and-audio](../../.claude/skills/voice-and-audio/SKILL.md),
  [remotion-sparse-overlays](../../.claude/skills/remotion-sparse-overlays/SKILL.md).
- Validate: [visual-qa-validator](../../.claude/skills/visual-qa-validator/SKILL.md).

This SOP is locally authored from approved scope and FrameFoundry code, with
Measure/Draw/Verify, sparse mapping and inspection concepts adapted from the
three permitted MIT repositories; no upstream text/code/templates were copied.
[Manifest](../skills/provenance.json) pins exact paths/commits and license hashes:
- kajisho5/remotion-ffmpeg-skill — MIT, Copyright (c) 2026 kajisho5;
  [full notice](../skills/licenses/kajisho5-remotion-ffmpeg-skill-MIT.txt).
- haidrrrry/claude-remotion-skill — MIT, Copyright (c) 2025 haidrrrry;
  [full notice](../skills/licenses/haidrrrry-claude-remotion-skill-MIT.txt).
- artemnovitckii/claude-editing-skills — MIT, Copyright (c) 2026 Artem Novitckii;
  [full notice including exceptions](../skills/licenses/artemnovitckii-claude-editing-skills-MIT.txt).

No media/assets/fonts included. Artem's OFL fonts and third-party media are
excluded. appariciojunior CC-BY-NC, remotion-dev/skills without a license and
OpenMontage AGPL were not cloned/read/copied (licensing labels supplied by user).
Original additions use the [repository MIT notice](../../LICENSE).
