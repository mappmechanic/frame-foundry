---
name: voice-and-audio
description: Preserve dialogue, handle only authorized optional music and verify encoded-output loudness and true peak.
---

# Voice and audio

## Authority and availability

Current `pipelines/ffmpeg/assembler.py`, `render.py`, `qa_validator.py`,
`eval_gates.py` and `contracts.py` are authoritative. Existing cut protection and
normalization are implemented. The [production runtime](../../../docs/PRODUCTION_POLICY.md) invokes `audio_policy.py`
for optional local mixing/ducking with `--production --music-bed <local-path>
--music-authorized`; installing this skill does not enable the policy.
Authorization is operator attestation, not independent license verification.
See [availability](../../../docs/skills/README.md) and
[provenance](../../../docs/skills/provenance.json).

## Assembly pass

1. Preserve retained source dialogue and the authoritative **80ms pre-roll /
   120ms post-roll**; protect neighbouring words through the assembler. Do not
   replace unclear speech with synthesized claims. New voiceover remains an
   explicit authorized operation through existing tools/budget controls.
2. Music is optional and must be operator-supplied and demonstrably authorized.
   With no bed, report **music absent**, retain dialogue and do not download,
   fabricate or synthesize substitute music/SFX. Invalid/unreadable/unauthorized
   music must be refused with an actionable diagnostic, not silently accepted.
3. For opt-in production, duck authorized music under retained speech and VO;
   use measured/timed activity and audition intelligibility. A universal gain
   value is not evidence of adequate ducking. Do not cut protected words to fit
   a beat. Inspect the saved `editing_policy.audio` receipt; do not invent schema fields.
4. Re-measure **the encoded output**, not only the source or mixer settings.
   Keep the current hard gates: finite **[-15,-13] LUFS** (-14 +/-1), finite
   **true peak <=-1 dBTP**. A recommendation like -16 LUFS from another skill is
   not an override. AAC overshoot must be caught after encoding.

## Verify / handoff

Retain measured loudness/peak, source/VO provenance and music authorization (when
present). A silent source may stay silent; nonfinite loudness does not pass the
current delivery gate. Missing measurements, damaged dialogue or gate failures
block a delivery claim; route diagnostics through existing QA, not a prose PASS.
Run the current QA/contact-sheet path described in
[SOP-03](../../../docs/sops/SOP-03-remotion-sparse-overlays.md).

## Attribution / license

Locally authored from approved scope/code. Audio-under-voice and render/inspect
concepts informed by **haidrrrry/claude-remotion-skill**,
`remotion-motion-graphics/references/design-rules.md` at
`1dcbe5e3fc6cf970bd10d3cc05f0a8a5d19d0383`.
MIT, **Copyright (c) 2025 haidrrrry**; full
[local notice](../../../docs/skills/licenses/haidrrrry-claude-remotion-skill-MIT.txt).
No upstream text/code copied; upstream mandatory SFX/music suggestions are NOT
adopted. No media/fonts included. Original work: [repository MIT](../../../LICENSE).
