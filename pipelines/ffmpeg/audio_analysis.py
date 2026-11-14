"""Offline PCM energy and explicitly qualified speech evidence.

100 ms bins use a fixed -40 dBFS RMS activity threshold, not a speech classifier.
Optional faster-whisper VAD is used only when its packaged ONNX asset exists;
no transcription model is constructed and no asset is downloaded. ``words`` must
already be aligned to this proxy's timeline (and filtered to this source).
"""
from __future__ import annotations

from array import array
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from .contracts import StageError

SAMPLE_RATE = 16000
BIN_MS = 100
ACTIVITY_RMS = 0.01
DECODE_TIMEOUT_SECONDS = 180


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def _bundled_vad(pcm_path: Path):
    """Return sample intervals or None; imports are optional, never fetch assets."""
    try:
        import importlib.util
        spec = importlib.util.find_spec('faster_whisper')
        if spec is None or not spec.origin:
            return None
        assets = Path(spec.origin).parent / 'assets'
        if not any((assets / name).is_file() for name in
                   ('silero_vad_v6.onnx', 'silero_vad.onnx')):
            return None
        import numpy as np
        from faster_whisper.vad import get_speech_timestamps
        samples = np.fromfile(pcm_path, dtype='<i2').astype(np.float32) / 32768.0
        return get_speech_timestamps(samples, sampling_rate=SAMPLE_RATE)
    except Exception:
        # Optional runtime/ONNX failures must not disable local energy evidence.
        return None


def analyze_audio(proxy_path: Path, *, words: list | None = None) -> dict:
    """Decode local audio and return RMS, activity, speech evidence and time bins.

    Speech fraction is null without local VAD or nonempty aligned ASR words.
    Empty word lists alone are not trusted proof of no speech. VAD failure falls
    back to ASR coverage/energy; decode failures raise a structured StageError.
    Time intervals are milliseconds relative to the proxy, end-exclusive.
    """
    path = Path(proxy_path).resolve()
    if not path.is_file():
        raise StageError('AUDIO_ANALYSIS_FAILED', 'refinery', 'Local audio proxy is missing')
    with tempfile.TemporaryDirectory(prefix='framefoundry-audio-') as directory:
        pcm_path = Path(directory) / 'audio.pcm'
        try:
            subprocess.run(
                ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                 '-i', str(path), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                 '-f', 's16le', str(pcm_path)],
                check=True, capture_output=True, timeout=DECODE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise StageError('AUDIO_ANALYSIS_FAILED', 'refinery',
                             'Local PCM decode failed or timed out') from error
        bins = []
        total_squared = 0
        total_samples = 0
        active_samples = 0
        with pcm_path.open('rb') as stream:
            while data := stream.read(SAMPLE_RATE * BIN_MS // 1000 * 2):
                samples = array('h')
                samples.frombytes(data)
                if sys.byteorder != 'little':
                    samples.byteswap()
                squared = sum(int(sample) ** 2 for sample in samples)
                rms = math.sqrt(squared / len(samples)) / 32768.0
                active = rms >= ACTIVITY_RMS
                start = round(total_samples * 1000 / SAMPLE_RATE)
                total_samples += len(samples)
                total_squared += squared
                if active:
                    active_samples += len(samples)
                bins.append(dict(start_ms=start, end_ms=round(total_samples * 1000 / SAMPLE_RATE),
                                 rms=round(rms, 6), active=active))
        duration_ms = round(total_samples * 1000 / SAMPLE_RATE)
        detected = _bundled_vad(pcm_path) if total_samples else None
        if detected is not None:
            intervals = _union((max(0, round(c['start'] * 1000 / SAMPLE_RATE)),
                                min(duration_ms, round(c['end'] * 1000 / SAMPLE_RATE)))
                               for c in detected)
            method = 'local_silero_vad'
        elif words:
            intervals = _union((max(0, int(w['start_ms'])), min(duration_ms, int(w['end_ms'])))
                               for w in words)
            method = 'asr_word_coverage'
        else:
            intervals = None
            method = 'energy_only'
        speech_fraction = (sum(b - a for a, b in intervals) / duration_ms
                           if intervals is not None and duration_ms else
                           0.0 if intervals is not None else None)
        return dict(rms=round(math.sqrt(total_squared / total_samples) / 32768.0, 6)
                    if total_samples else 0.0,
                    activity_fraction=round(active_samples / total_samples, 6) if total_samples else 0.0,
                    speech_fraction=round(speech_fraction, 6) if speech_fraction is not None else None,
                    method=method, duration_ms=duration_ms, bins=bins,
                    speech_intervals_ms=intervals)
