"""Independent eval gates on real FFmpeg renders, including seeded defects."""
import json
import shutil

import pytest
from PIL import Image

from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.eval_gates import (THRESHOLDS, content_box, generate_distinct_footage, repeated_shot_pairs,
                                         run_gates, seeded_self_test, word_cut_violations)

GOAL = 'Create a 10 second family-trip teaser'


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp('gates')
    fixture = generate_distinct_footage(root / 'footage')
    report = run_episode(root / 'footage', GOAL, episodes_root=root / 'episodes', episode_id='base',
                         transcript_fixture=fixture, provider=None)
    assert report['passed']
    return root / 'episodes' / 'base'


def _copy(base, tmp_path, name='copy'):
    target = tmp_path / name
    shutil.copytree(base, target, ignore=shutil.ignore_patterns('work', 'proxies', 'frames'))
    return target


def _status(result, name):
    return next(c for c in result['checks'] if c['name'] == name)


def test_seeded_defects_detected_100_percent(tmp_path):
    result = seeded_self_test(tmp_path)
    assert result['pipeline_passed'] and result['control_passed'], result
    assert result['seeded'] == 6 and result['detected'] == 6, result['cases']
    assert {c['defect'] for c in result['cases']} == {'loud', 'black', 'aspect', 'midword', 'duplicate', 'freeze'}


def test_clean_render_passes_and_letterbox_is_not_black(base):
    result = run_gates(base)
    assert result['passed'], [c for c in result['checks'] if c['status'] != 'pass']
    # 16:9 sources inside a 1080x1920 frame leave large black bars that must not count.
    box = content_box(1080, 1920, 320, 180)
    assert box[3] - box[1] < 700
    assert _status(result, 'black_frames')['measurement']['runs'] == []
    loud = _status(result, 'loudness')['measurement']
    assert -15 <= loud['integrated_lufs'] <= -13 and loud['method'].startswith('ffmpeg ebur128')
    assert _status(result, 'true_peak')['measurement']['true_peak_dbtp'] <= -1
    assert _status(result, 'contact_sheet')['status'] == 'pass'
    assert not any('not full schema coverage' in x for x in result['limitations'])
    assert 'analysis.json' in _status(result,'artifact_schemas')['measurement']['validated']
    assert any('fixture' in x for x in result['limitations'])


def test_visual_only_word_gate_not_run_and_fails_g(base, tmp_path):
    target = _copy(base, tmp_path)
    analysis = json.loads((target / 'analysis.json').read_text())
    analysis['transcription_status'] = 'visual_only_unavailable'
    (target / 'analysis.json').write_text(json.dumps(analysis))
    result = run_gates(target)
    word = _status(result, 'word_cut_integrity')
    assert word['status'] == 'not_run' and not result['passed']
    del analysis['transcription_status']
    (target / 'analysis.json').write_text(json.dumps(analysis))
    assert _status(run_gates(target), 'word_cut_integrity')['status'] == 'not_run'


def test_duration_gate_is_strict_even_for_partial_coverage(base, tmp_path):
    target = _copy(base, tmp_path)
    spec = json.loads((target / 'narrative-spec.json').read_text())
    spec['target_duration_ms'] = 12000
    spec['coverage'] = dict(status='partial', limitations=['explained shortfall'])
    (target / 'narrative-spec.json').write_text(json.dumps(spec))
    result = run_gates(target)
    assert _status(result, 'duration_target')['status'] == 'fail' and not result['passed']
    spec['target_duration_ms'] = 10900  # within +/-10%
    (target / 'narrative-spec.json').write_text(json.dumps(spec))
    assert _status(run_gates(target), 'duration_target')['status'] == 'pass'


def test_prerender_failure_has_not_run_media_gates(tmp_path):
    (tmp_path / 'qa-report.json').write_text('{"schema_version": "1.0"}')
    result = run_gates(tmp_path)
    assert not result['passed']
    assert _status(result, 'artifacts_present')['status'] == 'fail'
    for name in ['loudness', 'dimensions', 'black_frames', 'repeated_shots', 'word_cut_integrity']:
        assert _status(result, name)['status'] == 'not_run'
    assert _status(result, 'artifact_schemas')['status'] == 'fail'


def test_word_padding_edges_and_source_clamp():
    shot = dict(shot_id='shot-01', source_id='s', source_start_ms=0, source_end_ms=2000, source_duration_ms=5000)
    word = lambda a, b: dict(source_id='s', text='w', start_ms=a, end_ms=b)
    # Onset 30ms: pre-roll clamped at source start -> recorded, not a violation.
    violations, clamps = word_cut_violations([shot], [word(30, 400)])
    assert violations == [] and clamps and clamps[0]['edge'] == 'start'
    assert word_cut_violations([shot], [word(1900, 2100)])[0]          # mid-word
    assert word_cut_violations([shot], [word(1700, 1900)])[0]          # inside +120ms decay
    assert word_cut_violations([shot], [word(2050, 2300)])[0]          # inside -80ms pre-roll
    assert word_cut_violations([shot], [word(1500, 1880)])[0] == []     # exactly +120ms
    assert word_cut_violations([shot], [word(2080, 2300)])[0] == []     # exactly -80ms


def test_dhash_luma_guard_excludes_flat_frames():
    stats = [dict(shot_id='shot-01', dhashes=[]), dict(shot_id='shot-02', dhashes=[]),
             dict(shot_id='shot-03', dhashes=['ffff000000000000']), dict(shot_id='shot-04', dhashes=['ffff000000000001'])]
    pairs, flat = repeated_shot_pairs(stats)
    assert pairs == [dict(a='shot-03', b='shot-04', distance=1)] and flat == ['shot-01', 'shot-02']
    assert THRESHOLDS['dhash_max_distance'] == 4


def test_corrupt_contact_sheet_and_schema_fail(base, tmp_path):
    target = _copy(base, tmp_path)
    (target / 'contact-sheet.jpg').write_bytes(b'not a jpeg')
    qa = json.loads((target / 'qa-report.json').read_text())
    qa['invented'] = True
    (target / 'qa-report.json').write_text(json.dumps(qa))
    result = run_gates(target)
    assert _status(result, 'contact_sheet')['status'] == 'fail'
    assert _status(result, 'artifact_schemas')['status'] == 'fail'
    with Image.open(base / 'contact-sheet.jpg') as image:
        assert image.size[0] > 0  # original boundary sheet preserved
