"""Local candidate bounds, evidence quality, reuse, determinism and diversity."""
import copy
import json
import math
import socket

from PIL import Image, ImageDraw, ImageFilter
import pytest

from pipelines.ffmpeg import candidates as module
from pipelines.ffmpeg.candidates import extract_candidates, rank_candidates, compact_candidates
from pipelines.ffmpeg.contracts import StageError


def picture(root, name, kind='detail'):
    image = Image.new('L', (128, 128), 128)
    if kind in ('detail', 'blur', 'shift'):
        draw = ImageDraw.Draw(image)
        for x in range(8, 120, 12):
            for y in range(8, 120, 12):
                draw.rectangle((x, y, x + 5, y + 5), fill=20 if (x + y) % 24 else 230)
        if kind == 'blur':
            image = image.filter(ImageFilter.GaussianBlur(4))
        if kind == 'shift':
            from PIL import ImageChops
            image = ImageChops.offset(image, 6, 5)
    elif kind in ('black', 'white'):
        image = Image.new('L', (128, 128), 0 if kind == 'black' else 255)
    image.save(root / name)
    return name


def source(sid='s1', duration=25000, frames=None, **kwargs):
    return dict(source_id=sid, duration_ms=duration, path=None, proxy_path=None,
                has_audio=False, proxy_time_offset_ms=0, keyframes=frames or [], **kwargs)


def word(text, a, b, sid='s1'):
    return dict(source_id=sid, text=text, start_ms=a, end_ms=b)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *args: pytest.fail('Network forbidden'))


def test_scene_windows_source_bounds_and_stable_ids(tmp_path):
    analysis = dict(sources=[source()], scenes=[dict(source_id='s1', start_ms=0, end_ms=12000),
                                               dict(source_id='s1', start_ms=12000, end_ms=25000)], words=[])
    candidates = extract_candidates(analysis, tmp_path)
    assert [(c['start_ms'], c['end_ms']) for c in candidates] == [(0, 10000), (10000, 12000),
                                                               (12000, 22000), (22000, 25000)]
    assert candidates == extract_candidates(analysis, tmp_path, window_ms=30000)
    assert len({c['candidate_id'] for c in candidates}) == 4
    assert all(c['frame_path'] is None and c['frame_hash'] is None for c in candidates)
    assert all(c['quality']['visual_score'] == 0 and c['quality_method']['status'] == 'missing_frame'
               for c in candidates)
    assert len(extract_candidates(analysis, tmp_path, window_ms=3000)) == 9
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            extract_candidates(analysis, tmp_path, window_ms=invalid)


def test_frame_selection_in_window_first_and_nearest_source_labeled(tmp_path):
    for name in ('early.png', 'inside.png', 'late.png'):
        picture(tmp_path, name)
    frames = [dict(path='late.png', source_time_ms=24000), dict(path='early.png', source_time_ms=8000),
              dict(path='inside.png', source_time_ms=10001)]
    analysis = dict(sources=[source(frames=frames)], words=[])
    found = extract_candidates(analysis, tmp_path)
    assert [c['frame_path'] for c in found] == ['early.png', 'inside.png', 'late.png']
    assert found == extract_candidates(dict(analysis, sources=[source(frames=list(reversed(frames)))]), tmp_path)
    short = source(duration=5000, frames=[dict(path='early.png', source_time_ms=4500)])
    nearest = extract_candidates(dict(sources=[short]), tmp_path, window_ms=2000)
    assert nearest[0]['frame_time_ms'] == 4500
    assert nearest[0]['frame_in_window'] is False
    assert nearest[-1]['frame_in_window'] is True


def test_sharpness_exposure_and_honest_stability_surrogate(tmp_path):
    for kind in ('detail', 'blur', 'black', 'white', 'shift'):
        picture(tmp_path, kind + '.png', kind)
    infos = {k: module._image_info(tmp_path / (k + '.png')) for k in ('detail', 'blur', 'black', 'white', 'shift')}
    assert infos['detail']['sharpness'] > infos['blur']['sharpness']
    assert infos['detail']['exposure'] > infos['black']['exposure'] == infos['white']['exposure'] == 0
    assert module._stability(infos['detail'], infos['detail']) == 100
    assert module._stability(infos['detail'], infos['shift']) < 100
    frames = [dict(path='detail.png', source_time_ms=1000), dict(path='shift.png', source_time_ms=9000)]
    c = extract_candidates(dict(sources=[source(frames=frames)]), tmp_path)[0]
    assert c['quality_method']['stability'] == 'sampled_luma_change_surrogate'
    assert all(0 <= n <= 100 for n in c['quality'].values())


def test_equal_near_images_reuse_but_preserve_transcript_moments(tmp_path):
    picture(tmp_path, 'a.png')
    with Image.open(tmp_path / 'a.png') as image:
        image.save(tmp_path / 'b.jpg', quality=95)
    frames = [dict(path='a.png', source_time_ms=5000), dict(path='b.jpg', source_time_ms=15000)]
    analysis = dict(sources=[source(duration=20000, frames=frames)],
                    words=[word('first story', 1000, 2000), word('second story', 11000, 12000)])
    candidates = extract_candidates(analysis, tmp_path)
    assert len(candidates) == 2
    assert candidates[0]['canonical_frame_path'] == candidates[1]['canonical_frame_path'] == 'a.png'
    assert candidates[0]['transcript_snippet'] != candidates[1]['transcript_snippet']
    assert candidates[0]['candidate_id'] != candidates[1]['candidate_id']
    picture(tmp_path, 'black.png', 'black')
    picture(tmp_path, 'white.png', 'white')
    black = module._image_info(tmp_path / 'black.png')
    white = module._image_info(tmp_path / 'white.png')
    assert black['dhash'] == white['dhash']  # dHash collision is NOT an identity key.
    assert black['frame_hash'] != white['frame_hash']
    assert not module._near(black, white)


def test_actual_identical_sources_only_dedupe_identical_moments_with_provenance(tmp_path):
    # Stand-in local source bytes suffice: identity hashing never invents footage.
    (tmp_path / 'one').write_bytes(b'local source bytes')
    (tmp_path / 'two').write_bytes(b'local source bytes')
    sources = [source('s2', 10000), source('s1', 10000)]
    sources[0]['path'] = str(tmp_path / 'two')
    sources[1]['path'] = str(tmp_path / 'one')
    analysis = dict(sources=sources, words=[word('same', 1000, 2000, 's1'), word('same', 1000, 2000, 's2')])
    result = extract_candidates(analysis, tmp_path)
    assert len(result) == 1 and result[0]['source_id'] == 's1'
    assert [p['source_id'] for p in result[0]['provenance']] == ['s1', 's2']
    analysis['words'][0]['text'] = 'different'
    assert len(extract_candidates(analysis, tmp_path)) == 2
    (tmp_path / 'two').write_bytes(b'different source bytes')
    analysis['words'][0]['text'] = 'same'
    assert len(extract_candidates(analysis, tmp_path)) == 2


def test_ranking_is_bounded_stable_diverse_and_nonmutating(tmp_path):
    analysis = dict(sources=[source('s2', 20000), source('s1', 20000)],
                    words=[word('family trip ' * 80, 1000, 2000), word('trip', 11000, 12000)])
    candidates = extract_candidates(analysis, tmp_path)
    snapshot = copy.deepcopy(candidates)
    ranked = rank_candidates(candidates, 'family trip', top_n=4)
    assert ranked == rank_candidates(list(reversed(candidates)), 'family trip', top_n=4)
    assert candidates == snapshot
    assert [c['source_id'] for c in ranked] == ['s1', 's2', 's1', 's2']
    assert all(len(c['transcript_snippet']) <= 160 and math.isfinite(c['pre_rank']) and 0 <= c['pre_rank'] <= 100
               for c in ranked)
    assert rank_candidates(candidates, 'family trip', top_n=1)[0]['pre_rank'] > candidates[0]['pre_rank']
    assert rank_candidates(candidates, '', top_n=0) == []
    assert len(rank_candidates(candidates, '')) == 4
    assert compact_candidates(ranked)[0]['canonical_frame_path'] is None
    json.dumps(compact_candidates(ranked), allow_nan=False)
    before = {c['candidate_id']: c['pre_rank'] for c in rank_candidates(candidates, 'trip', top_n=20)}
    extra = extract_candidates(dict(sources=[source('s3')]), tmp_path)
    after = {c['candidate_id']: c['pre_rank'] for c in rank_candidates(candidates + extra, 'trip', top_n=20)}
    assert all(after[k] == v for k, v in before.items())
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            rank_candidates(candidates, '', top_n=invalid)


def test_window_audio_energy_and_asr_coverage_not_sourcewide_speech(tmp_path):
    s = source(duration=2000)
    s['has_audio'] = True
    analysis = dict(sources=[s], words=[word('known', 100, 400), word('overlap', 200, 500)],
                    audio_metrics={'s1': dict(rms=.5, speech_fraction=.8, method='energy_only', bins=[
                        dict(start_ms=0, end_ms=1000, rms=.2), dict(start_ms=1000, end_ms=2000, rms=.6)])})
    first, second = extract_candidates(analysis, tmp_path, window_ms=1000)
    assert first['audio'] == dict(rms=.2, speech_fraction=.4, method='asr_word_coverage')
    assert second['audio']['speech_fraction'] == 0 and second['audio']['rms'] == .6
    analysis['words'] = []
    assert extract_candidates(analysis, tmp_path)[0]['audio']['speech_fraction'] is None
    analysis['audio_metrics']['s1'].update(method='local_silero_vad', speech_intervals_ms=[[0, 200]])
    s['proxy_time_offset_ms'] = 100
    assert extract_candidates(analysis, tmp_path, window_ms=1000)[0]['audio']['speech_fraction'] == .2


def test_proxy_decoded_once_and_failure_degrades_honestly(tmp_path, monkeypatch):
    (tmp_path / 'proxy').write_text('placeholder')
    s = source()
    s.update(has_audio=True, proxy_path='proxy')
    calls = []
    def analyze(path, *, words):
        calls.append((path, words))
        return dict(rms=.2, method='energy_only')
    monkeypatch.setattr(module, 'analyze_audio', analyze)
    found = extract_candidates(dict(sources=[s]), tmp_path)
    assert len(found) == 3 and len(calls) == 1
    assert all(c['audio']['speech_fraction'] is None for c in found)
    def fail(*args, **kwargs):
        raise StageError('AUDIO_ANALYSIS_FAILED', 'refinery', 'bad proxy')
    monkeypatch.setattr(module, 'analyze_audio', fail)
    assert extract_candidates(dict(sources=[s]), tmp_path)[0]['audio']['method'] == 'unavailable'


def test_unsafe_corrupt_unlabeled_and_foreign_frames_are_not_used(tmp_path):
    picture(tmp_path, 'valid.png')
    (tmp_path / 'bad.png').write_text('broken')
    frames = [dict(path='../outside.png', source_time_ms=1000),
              dict(path=str(tmp_path / 'valid.png'), source_time_ms=1000),
              dict(path='valid.png'), dict(path='bad.png', source_time_ms=1000)]
    foreign = source('s2', frames=[dict(path='valid.png', source_time_ms=1000)])
    result = extract_candidates(dict(sources=[source(frames=frames), foreign]), tmp_path)
    assert all(c['frame_path'] is None for c in result if c['source_id'] == 's1')
    assert any(c['frame_path'] == 'valid.png' for c in result if c['source_id'] == 's2')


def test_nonfinite_metric_scores_are_sanitized(tmp_path):
    analysis = dict(sources=[source()], audio_metrics={'s1': dict(rms=float('inf'), method='energy_only')})
    c = extract_candidates(analysis, tmp_path)[0]
    c['quality']['visual_score'] = float('nan')
    c['audio']['speech_fraction'] = float('inf')
    assert rank_candidates([c], 'trip')[0]['pre_rank'] == 0
    assert c['audio']['rms'] == 0


def test_sparse_scene_gaps_overlaps_and_source_clamping(tmp_path):
    scenes = [dict(source_id='s1', start_ms=-500, end_ms=3000),
              dict(source_id='s1', start_ms=2000, end_ms=5000),
              dict(source_id='s1', start_ms=9000, end_ms=50000)]
    result = extract_candidates(dict(sources=[source(duration=10000)], scenes=scenes), tmp_path)
    assert [(c['start_ms'], c['end_ms']) for c in result] == [(0, 2000), (2000, 3000),
                                                          (3000, 5000), (9000, 10000)]
