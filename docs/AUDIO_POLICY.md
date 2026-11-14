# Authorized local music policy — WO-18 audio slice

## Exact API

```python
def apply_audio_policy(video, output, *, music_path=None, authorized=False,
                       dialogue_intervals=(), policy=None) -> dict: ...
```

Import from `pipelines.ffmpeg.audio_policy`; failures are `AudioPolicyError`, a
`StageError` with `stage='assembler'` and actionable `AUDIO_*` code.
`video`/`output` are distinct local `str`/`Path` files. Output is MP4 delivery.

### Parent integration

Call **after overlay rendering** on already-normalized rendered A/V including
any cached intro VO. Parent owns planner/assembler/CLI/schema wiring and will
supply `ctx.music_bed: Path` and `ctx.music_authorized: bool`. Parent translates
both source ASR Protected Speech and cached VO onto the **final OUTPUT timeline**.
Pass `(start_seconds, end_seconds)` pairs, including VO at its output-relative
time. Empty intervals mean no known protected speech, **not** automatic speech
detection. Non-speech time is considered candidate B-roll, not inferred visual
semantics. Parent must include every retained speech/VO interval.

Serialize the receipt with `allow_nan=False`, retain it in the policy receipt,
and propagate failures without advertising failed output as verified delivery.
Run normal pipeline QA and contact sheets afterward. This slice neither changes
source selections/frames nor wires default product behavior. It is not complete
WO-18/runtime delivery.

## Inputs and absence

* `music_path=None`: byte-for-byte copy, `status='absent'`, explicit limitation;
  no re-normalization and no optional mixing-filter check. Assumes parent audio
  has already passed authoritative normalization.
* Supplied music requires literal boolean `authorized=True`: operator attestation,
  **not** independent license verification. Missing, silent, undecodable,
  non-audio or unauthorized music is refused, never fetched/replaced/generated.
* Only local nonempty regular files; URL references rejected; media processes
  permit only `file,pipe` protocols. Binaries use `media_tools.media_binary`,
  including the renderer. No paid calls or asset generation.
* Video duration must be finite and positive, at most 3,600 seconds; at most 512
  finite pairs satisfy `0 <= start < end <= video_duration`. Overlaps/touching
  speech merge; outward sample rounding protects sub-sample speech.
* Default policy: `bed_lufs=-14`, `dialogue_lufs=-24`,
  `broll_min_seconds=2.5`, `transition_seconds=0.15`. A partial dict may adjust
  only `transition_seconds` in `[0, 0.5]`. Targets/threshold are fixed; unknown
  keys, booleans and nonfinite policy numbers are refused.

## Pre-mix levels versus final overall delivery

Music loops locally from its start, is resampled to stereo 48 kHz and bounded
to the exact output sample grid. FFmpeg `loudnorm` measures the bed's EBU R128
integrated loudness/true peak; constant lossless PCM gain normalizes its reference
to **-14 LUFS before mixing**. Floating-point lossless intermediates avoid
clipping/AAC generation loss during composition.

Dialogue/VO and **all gaps <=2.5 seconds** get additional -10 dB bed gain:
**-24 LUFS pre-mix reference**. Strictly longer non-speech gaps rise to the -14
reference. Linear amplitude ramps (default 150 ms) are wholly *inside* long
gaps; speech remains fully ducked at both boundaries. Variable music content
means regional integrated LUFS can differ: these are reference targets, not a
promise of uniform short-window loudness. Real stationary-sine tests measure
both absolute regional levels.

Mixing is additive (`amix normalize=0`), with unity source gain and unchanged
source timing before overall normalization. **Existing `render._normalize` is
reused unchanged**, including AAC headroom and bounded correction/limiter.
Encoded AAC is independently measured and must satisfy **-14 +/-1 LUFS and
true peak <=-1 dBTP** before output replacement. Final normalization changes
both source and bed level; peak limiting can change local ratios. Pre-mix bed
targets are **not** absolute music-level claims inside the final delivery. No
gates are relaxed. Video is `-c:v copy`, duration must remain within one original
frame and reported frame count must match; real tests also compare every video
packet hash, PTS and packet duration.

## Receipt, compatibility and limitations

Path-free receipt includes status, SHA-256 of **music contents only**, operator
authorization, policy/sample grid/duration, merged output-relative speech,
per-region sample bounds/type/pre-mix target/min/max relative gain, actual PCM
input/output RMS and energy gain, bed EBU R128 measurement/normalization gain,
final encoded LUFS/true peak and video-copy/duration evidence. Silent regional
gain is `null`, never NaN/Infinity. No raw media is committed.

Plain FFmpeg needs `loudnorm, aresample, aformat, apad, atrim, amix, volume,
alimiter` (last two serve authoritative normalization). Inventory parses flags
including `...`, `..C`, `T.C`. No `drawtext`, `subtitles`, sidechain or libass.
Missing filters raise typed `AUDIO_FILTERS` before mixing. Real execution tested
on installed Linux FFmpeg; Homebrew flag parsing is unit tested, not actual
macOS execution.

Chunked Python processing uses bounded memory; temporary lossless files require
storage proportional to length (several GB at one hour). No beat matching or
loop crossfade: discontinuous bed endpoints may click. Stereo downmix only,
not spatial audio/extra tracks. Source A1 and V1 retain timing/content, not source
audio codec bytes: normalization/AAC encoding changes audio samples. No source
cut/reorder, planner call, autonomous speech detection or download occurs.

## Tests

```sh
python -m pytest tests/test_audio_policy.py -q
```

Real generated sine dialogue, intro-like VO and music test encoded LUFS/true
peak, frequency-isolated 10 dB ducking, pre-mix regional EBU R128, source/intro
survival, exact video packets, byte-identical absence, unsafe paths, licensing,
interval/policy failures, filter inventory and deterministic sample-grid fades.
