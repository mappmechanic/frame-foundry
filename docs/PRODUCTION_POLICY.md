# Production runtime policy (WO-18)

Current implementation, opt-in; this is not default promotion or measured quality proof.
`contracts/reel-policy.json` is hashed into source-bound narrative receipts. Planner
receives the same policy in its existing single call when `--production` is set;
assembly applies deterministic rules even when replaying a saved plan with no Planner.

## Implemented output behavior

- Promote an already retained, hash-bound action/people hook evidenced within its
  first two seconds. Preserve explicit chronological goals. Unavailable hook
  evidence is disclosed, not invented.
- Strip earlier exact complete same-source utterances within30seconds only when a
  later complete take matches. Keep incomplete/different utterances.
- Remove only measured silent gaps >350ms between ASR words outside80ms pre-roll /
  120ms post-roll, preserving minimum retained fragment length. Proxy decode failures
  retain the interval and disclose the missing measurement.
- Balance explicitly visual-only long intervals into1.5–4s target windows; ordinary
  speech/unknown intervals are preserved with exceptions. Adjacent policy shots do
  not silently coalesce. No blind speech chopping to meet a target.
- Original analytic14/120/1 damped-spring title entry, normalized to settle in250ms.
  Transparent Pillow sprites are rasterized once and displayed only over active
  ranges through FFmpeg overlay/fade; plain Homebrew FFmpeg remains supported.
- Production cards retain measured cap height/readability/pill/cut masks and use
  portrait top10/bottom22/right15 exclusions or landscape10% title-safe margins.
  Local Haar sampling records mid-shot geometry. A verified sampled face collision
  blocks delivery; unavailable geometry is disclosed. This is NOT eye/chin landmark
  tracking or a proof that every frame is face-safe.
- Optional operator-authorized local music uses output-mapped protected speech/VO
  ducking. See AUDIO_POLICY.md for pre-mix−24/−14 LUFS references and measured global
  normalization. No bed means explicit absence, not invented music.
- Production contact sheets retain before/after cut evidence and add cut+3 frames
  with translucent red portrait/landscape exclusion zones.
- Required configuration/source-plan, hook-disclosure, pacing/exception, audio,
  safe-zone and sampled-subject checks augment—not replace—existing hard gates.

## Commands

Same saved Sedona plan, no new Planner/ASR/TTS:

```sh
git pull --ff-only
RUN="az-sedona-prod-$(date +%H%M%S)"
.venv/bin/python scripts/reoverlay.py --from-episode az-sedona-1315 --episode "$RUN" --production
make site
```

Add `--music-bed /absolute/local/music.wav --music-authorized` only with permission.
Add `--judge` for a fresh bounded ordinary Judge; never reuse old render J/Q.
`make demo-arizona ARGS="--episode fresh-id --overlays --production"` also enables
policy on normal runs; memory is independently opt-in via `--memory --memory-backend local`.

Matched editing-only A/B, two saved-plan renders, no new Planner/ASR/TTS; absent
`--judge` scores remain unknown, not an improvement claim:

```sh
# Existing ledger retains its own recorded initial spend; dashboard $74 is user-reported.
# Pass a ceiling; do not overwrite a lower existing environment cap.
.venv/bin/python scripts/policy_ab.py --from-episode az-sedona-1315 \
  --experiment "sedona-policy-$(date +%H%M%S)" --judge --budget-limit-usd 300
make site
```

Both episode allowances are reserved before any paid Judge. A lower configured
limit is never raised. Runner writes private evals/policy-comparisons/latest.json;
export/pitch expose only fully measured comparable pairs whose distinct episodes
are actually published. No live Arizona gain was measured in this credential-free
workspace. Model-cost unknown/failure/comparison mismatch stays unavailable.

## Still not implemented

Remotion React/ProRes4444/WebM sparse files on OTIO V3, verified eye/chin tracking,
continuous subject tracking and112% face-safe punch-ins. SOP-03 declares those
future adapter contracts; current renderer remains Pillow/FFmpeg V1/A1. No default
promotion or production-grade certification follows solely from installing skills.
