# Curated production editing guidance (WO-17)

These are small, locally authored instructions, not an installed editing engine.
[provenance.json](provenance.json) pins inspected upstream paths/SHAs and copied
license hashes. Only full MIT notices were copied verbatim; the guidance adapts
listed concepts and applies FrameFoundry requirements instead of upstream prompts.
All five skills identify their provenance. Local original work uses the repository
[MIT license](../../LICENSE).

## License and source boundary

- kajisho5/remotion-ffmpeg-skill: MIT, Copyright (c) 2026 kajisho5.
- haidrrrry/claude-remotion-skill: MIT, Copyright (c) 2025 haidrrrry.
- artemnovitckii/claude-editing-skills: MIT, Copyright (c) 2026 Artem Novitckii.
  Its full notice retains OFL font and third-party media exceptions; **none** of
  those fonts/media are included. No images, audio, footage, previews or assets
  are tracked by this integration.
- appariciojunior (user-reported CC-BY-NC), remotion-dev/skills (user-reported no
  license), OpenMontage (user-reported AGPL): excluded; not cloned/read/copied.
  No independently verified licensing finding is claimed for these sources.
- Optional zhuyansen CC0 index: not used; no indexed projects copied.

Inspection used only `git clone --depth 1 https://github.com/<permitted-repo>.git
_vendor_skills/<repo-name>` for the three MIT repositories. The staging root is
ignored. No installer, npm/npx command or upstream script was executed. To audit,
read each manifest path at its pinned GitHub commit, compare the full notice hash,
and check `git check-ignore _vendor_skills/<repo-name>/LICENSE`.

## Availability at ea0887e (not a promise about later parent commits)

| Capability | Status / authoritative evidence |
| --- | --- |
| Source-word cut protection | Implemented: `pipelines/ffmpeg/assembler.py:pad_word_interval`, 80ms pre-roll / 120ms post-roll, fixed-point neighbour protection |
| Large grounded cards | Implemented: `pipelines/ffmpeg/overlays.py`, `contracts/overlay-style.json`; Pillow PNG + FFmpeg active-window compositing, no FFmpeg text filters |
| Cut masking and steady readability | Implemented: cut-1/cut/cut+1, >=75 full-strength frames at 30fps (>=2.5s), .06H layout cap height (~116px at 1920H) |
| Output audio QA | Implemented: `qa_validator.py`, `eval_gates.py`, `contracts.py`; [-15,-13] LUFS and <=-1 dBTP |
| Cut contact sheet | Implemented: `contact_sheet.py` samples cut +/-34ms; not the proposed full cut+3 exclusion audit |
| Remotion, alpha V3 layer, landmark tracking | Not installed/implemented here: `packages/remotion-studio/` is absent, no Remotion project or package manifest in checkout |
| Versioned opt-in production policy | Declared target; parent/WO-18 owns implementation and schema decisions. This branch adds no policy switch or persisted field |
| SOP-01/SOP-02 | Absent from checkout. KB lists PDFs; contents not read or invented. No local filename guessed |

Follow [SOP-03](../sops/SOP-03-remotion-sparse-overlays.md) for the declared target
and stricter review criteria. Do not add invented fields to the strict existing
schemas. A saved plan/policy claim must be backed by the runtime owner’s actual
versioned fields and applied-change evidence. Docs alone do not demonstrate an
improved output, quality score, latency or cost.

## Checks

Run `python -m pytest -q tests/test_curated_skills_docs.py` for local routing,
licenses, links, availability and policy wording checks. These are documentation
regressions, not visual inspection or end-to-end policy verification. Existing
assembler/overlay/audio/QA regression tests cover the unchanged current engine.

## Knowledge Base publication

[SOP-03 editable document](https://factory.8090.ai/project/a54860bf-bf30-42d7-b6a8-a08af85c454c/knowledge-base/af423584-7fff-499f-83e6-3ea1cb5229c2)
(resource ID `af423584-7fff-499f-83e6-3ea1cb5229c2`) publishes the SOP from
`a69f14c` with pinned repository links. Its initial content is pending review.
WO-17 links the resource; later runtime changes must update availability honestly,
not silently reinterpret this baseline documentation as shipped policy behavior.
