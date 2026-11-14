---
name: remotion-sparse-overlays
description: Route measured sparse overlay work to the current PNG compositor or a clearly declared future Remotion alpha V3 design.
---

# Sparse overlays: Measure -> Draw -> Verify

Read [SOP-03](../../../docs/sops/SOP-03-remotion-sparse-overlays.md) for architecture,
measurement, masks and evidence. Do not load entire upstream prompts.

## Capability boundary

**Implemented:** `pipelines/ffmpeg/overlays.py` rasterizes with Pillow PNG and
composites through FFmpeg **sparse active-window** enables, 250ms fades and an
8px slide in legacy mode. The [production runtime](../../../docs/PRODUCTION_POLICY.md) uses `--production` for analytic
14/120/1 spring entry and stricter zones with sampled Haar collision checks.
It does not require drawtext/libass. It still processes the base video;
this is not a browser renderer packed into short alpha runs.

**Declared target:** Remotion/React sparse **V3** graphics, rendered only for active
ranges to **ProRes 4444 or WebM alpha**. Remotion is **not installed** in this
checkout, `packages/remotion-studio/` is absent, and the current renderer accepts
only V1/A1. Do not add V3 to existing OTIO, invent a command, run installers, or
claim this skill produces alpha media. Parent/WO-18 owns opt-in policy runtime;
Remotion integration needs separate implementation and package-license review.

## Working rules

- Measure final output/crop, source-bound cues, cut frames, eyes and chin before
  drawing. Missing landmarks are a limitation, not verified tracking.
- Preserve existing typography: cap height **>=ceil(0.06H)** (~**116px** at 1920H),
  two lines / 42 characters per line, 60% black backing, >=**2.5s steady hold**
  after entrance/exit and masked frames. Existing QA's .05H lower bound does not
  authorize shrinking the .06H design. No continuous breathing during the hold.
- Preserve **cut-1/cut/cut+1** masks. The proposed **cut+3 exclusion** and its
  exact cut-1 through cut+3 masking remain targets, not installed checks.
  Production adds a cut+3 contact-sheet sample with red exclusion zones; this
  sample is evidence for inspection, not proof of cut+3 alpha absence.
- Implemented opt-in safety: **9:16 bottom 22% / right 15% / top 10%**; **16:9 outer 10%**.
  Intersect with current upper-third/top-12% authority; shorten/omit unsafe cards
  rather than relax a gate. Measure the entire animated envelope and eye/chin
  collision safety, not a single guessed anchor.
- Future React animation uses deterministic integer frame math and local
  frame offsets, **spring damping=14, stiffness=120, mass=1** with bounded
  interpolation. No async React, clocks, timers, unseeded randomness or network
  work during rendering. Input/fonts must be resolved before rendering starts.
- Use active half-open frame ranges and retain global/local frame mapping when
  packing alpha runs. Reapply cut masks on the final timeline; don't reset spring
  phase at each packed segment. Verify alpha and duration before compositing.

## Attribution / license

Concept adaptations (no upstream text/code copied):
- **kajisho5/remotion-ffmpeg-skill**, `SKILL.md`, `scripts/_sparse.py`,
  `docs/adr/0001-render-determinism.md` at `e34a10dd2fefcc1cccd07352e272b89dcc6b3f3c`;
  MIT, **Copyright (c) 2026 kajisho5**, [full local notice](../../../docs/skills/licenses/kajisho5-remotion-ffmpeg-skill-MIT.txt).
- **haidrrrry/claude-remotion-skill**, `remotion-motion-graphics/SKILL.md` and
  references at `1dcbe5e3fc6cf970bd10d3cc05f0a8a5d19d0383`; MIT,
  **Copyright (c) 2025 haidrrrry**, [full local notice](../../../docs/skills/licenses/haidrrrry-claude-remotion-skill-MIT.txt).

The 14/120/1 spring is FrameFoundry's original implemented opt-in policy, not
upstream theme defaults or evidence that Remotion is installed.
Full paths and exceptions: [manifest](../../../docs/skills/provenance.json).
No assets/fonts/media included. Original additions: [repository MIT](../../../LICENSE).
