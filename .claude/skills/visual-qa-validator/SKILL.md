---
name: visual-qa-validator
description: Verify decoded final-output geometry, boundary masks, readable holds and honest policy evidence before claiming delivery.
---

# Visual QA validator

## Authority and availability

Validate after assembly with [SOP-03](../../../docs/sops/SOP-03-remotion-sparse-overlays.md).
`pipelines/ffmpeg/qa_validator.py`, `overlays.py:verify`, `contact_sheet.py` and
`eval_gates.py` are implemented. The [production runtime](../../../docs/PRODUCTION_POLICY.md) adds `--production`
configuration/plan, pacing, audio, safe-zone and sampled-face gates plus cut+3
red-zone contact-sheet evidence. Continuous landmarks and full **cut+3 exclusion**
masking remain **declared target** criteria; do not confuse a sample with a mask.
Remotion/V3 is not installed; present output uses Pillow PNG/FFmpeg active windows.
[Availability](../../../docs/skills/README.md) / [provenance](../../../docs/skills/provenance.json).

## Review the encoded result, not just the plan

1. Bind the actual render hash to the saved plan, style/font and source evidence.
   A stale image, planner thumbnail or another episode cannot certify this output.
2. Validate current cut masks at **cut-1/cut/cut+1** using exact decoded frames.
   For the proposed stricter audit, include **cut+2/cut+3/cut+4**, each labelled
   with frame number/time and output hash. Inspect onset/exit, midpoint/steady
   hold and worst measured eye/chin proximity too. Preserve the current generated
   boundary contact sheet; +/-34ms samples alone do not prove exact cut+3 absence.
3. Require >=**2.5s steady readable hold**, excluding fades, spring settling and
   hidden frames. Preserve .06H cap-height design (~**116px** at 1920H), two lines,
   42 chars per line and backed contrast. Do not trade readability for animation.
4. Review **9:16 bottom 22% / right 15% / top 10%**, **16:9 outer 10%**, together
   with current upper-third/top-12% gates. Check full moving rectangles against
   **measured eyes and chin** in final crop space. A face box or empty detection
   is not proof of collision-free tracking. Missing evidence means unverified;
   request safe omission/neutral framing rather than a fake PASS.
5. Keep **80ms pre-roll / 120ms post-roll**, current finite **[-15,-13] LUFS** and
   **true peak <=-1 dBTP** gates authoritative. Visual success cannot override
   failed schema, protected speech, audio, dimension, fps or duration checks.
6. Report each requested policy rule as implemented/applied, not applied,
   unsupported or unverified with reasons; parent runtime must own actual fields.
   Docs/tests passing does not prove production output improved. Compare matched
   baseline/policy renders only with source/goal/format/evaluation evidence.

## Handoff

Return actual output/report/contact-sheet paths, hashes, checks and limitations.
If pixels were not inspected, say **pixels not inspected**; generated sheets and
passing geometry tests are not visual review. Never label a failed/unverified
required policy result as production-ready. Do not perform new paid Judge calls
just to validate this documentation work.

## Attribution / license

Locally authored review instructions with inspection/evidence concepts adapted from:
- **kajisho5/remotion-ffmpeg-skill**, `SKILL.md`, determinism ADR at
  `e34a10dd2fefcc1cccd07352e272b89dcc6b3f3c`: MIT,
  **Copyright (c) 2026 kajisho5**, [full local notice](../../../docs/skills/licenses/kajisho5-remotion-ffmpeg-skill-MIT.txt).
- **haidrrrry/claude-remotion-skill**, motion skill/design references at
  `1dcbe5e3fc6cf970bd10d3cc05f0a8a5d19d0383`: MIT,
  **Copyright (c) 2025 haidrrrry**, [full local notice](../../../docs/skills/licenses/haidrrrry-claude-remotion-skill-MIT.txt).
- **artemnovitckii/claude-editing-skills**, storytelling/behind-the-person skills at
  `beff6b868a8c1bdfad8661bd8a298497e75568a3`: MIT,
  **Copyright (c) 2026 Artem Novitckii**, [full local notice](../../../docs/skills/licenses/artemnovitckii-claude-editing-skills-MIT.txt).

No upstream text/code copied, assets/fonts/media excluded, including Artem's
OFL and third-party media exceptions. Full paths: manifest above.
Original additions: [repository MIT](../../../LICENSE).
