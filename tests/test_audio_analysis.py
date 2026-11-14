"""Synthetic offline PCM proofs: activity is deliberately not called speech."""
from array import array
import math
import socket
import subprocess
import wave

import pytest

from pipelines.ffmpeg import audio_analysis as audio
from pipelines.ffmpeg.contracts import StageError


def make_wave(path, amplitude=0, seconds=1):
    values = array('h', (round(amplitude * 32767 * math.sin(2 * math.pi * 440 * i / 16000))
                         for i in range(round(16000 * seconds))))
    import sys
    if sys.byteorder != 'little':
        values.byteswap()
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(values.tobytes())
    return path


@pytest.fixture(autouse=True)
def offline_energy_only(monkeypatch):
    monkeypatch.setattr(audio, '_bundled_vad', lambda _: None)
    monkeypatch.setattr(socket.socket, 'connect', lambda *args: pytest.fail('Network access forbidden'))


def test_quiet_loud_tone_and_silence_are_not_speech(tmp_path):
    silent = audio.analyze_audio(make_wave(tmp_path / 'silent.wav'))
    quiet = audio.analyze_audio(make_wave(tmp_path / 'quiet.wav', .001))
    loud = audio.analyze_audio(make_wave(tmp_path / 'tone.wav', .5))
    assert silent['rms'] == silent['activity_fraction'] == 0
    assert quiet['activity_fraction'] == 0
    assert loud['activity_fraction'] == 1
    assert loud['rms'] == pytest.approx(.5 / math.sqrt(2), abs=.001)
    assert loud['duration_ms'] == 1000 and len(loud['bins']) == 10
    for result in (silent, quiet, loud):
        assert result['speech_fraction'] is None
        assert result['method'] == 'energy_only'
        assert result['speech_intervals_ms'] is None


def test_word_coverage_unions_clamps_and_empty_words_stay_unknown(tmp_path):
    path = make_wave(tmp_path / 'tone.wav', .3)
    words = [dict(start_ms=-100, end_ms=200), dict(start_ms=100, end_ms=400),
             dict(start_ms=900, end_ms=1200)]
    result = audio.analyze_audio(path, words=words)
    assert result['speech_fraction'] == .5
    assert result['speech_intervals_ms'] == [[0, 400], [900, 1000]]
    assert result['method'] == 'asr_word_coverage'
    assert audio.analyze_audio(path, words=[])['speech_fraction'] is None


def test_provisioned_vad_evidence_and_no_speech(tmp_path, monkeypatch):
    path = make_wave(tmp_path / 'tone.wav', .3)
    monkeypatch.setattr(audio, '_bundled_vad', lambda _: [dict(start=1600, end=8000)])
    result = audio.analyze_audio(path)
    assert result['method'] == 'local_silero_vad'
    assert result['speech_fraction'] == .4
    monkeypatch.setattr(audio, '_bundled_vad', lambda _: [])
    result = audio.analyze_audio(path)
    assert result['speech_fraction'] == 0 and result['activity_fraction'] == 1


def test_missing_asset_never_invokes_vad_or_download(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    monkeypatch.undo()
    monkeypatch.setattr(socket.socket, 'connect', lambda *args: pytest.fail('Network forbidden'))
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda _: SimpleNamespace(origin=str(tmp_path / 'faster_whisper' / '__init__.py')))
    result = audio.analyze_audio(make_wave(tmp_path / 'tone.wav', .3))
    assert result['method'] == 'energy_only' and result['speech_fraction'] is None


def test_decode_error_and_timeout_are_structured(tmp_path, monkeypatch):
    with pytest.raises(StageError) as missing:
        audio.analyze_audio(tmp_path / 'missing.wav')
    assert missing.value.code == 'AUDIO_ANALYSIS_FAILED'
    path = tmp_path / 'invalid.wav'
    path.write_text('not audio')
    with pytest.raises(StageError):
        audio.analyze_audio(path)
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == 180
        assert '-protocol_whitelist' in args[0]
        raise subprocess.TimeoutExpired('ffmpeg', 180)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(StageError):
        audio.analyze_audio(path)


def test_partial_bin_and_determinism(tmp_path):
    path = make_wave(tmp_path / 'short.wav', .2, .235)
    first = audio.analyze_audio(path)
    assert first == audio.analyze_audio(path)
    assert first['bins'][-1]['end_ms'] == first['duration_ms'] == 235
    assert len(first['bins']) == 3
