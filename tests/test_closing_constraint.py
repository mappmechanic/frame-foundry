"""Evidence-backed payoff without altering hook, word-safe pacing or frame budget."""
from copy import deepcopy
from pathlib import Path

import pytest

from pipelines.ffmpeg.assembler import build_timeline
from pipelines.ffmpeg.goal_parser import parse_goal
from pipelines.ffmpeg.planner import _reconstruct
from pipelines.ffmpeg.contracts import EpisodeContext
from pipelines.ffmpeg.teaser_edit import (
    CLOSING_MIN_EXPOSURE, CLOSING_MIN_SHARPNESS, closing_qualifies, complete,
    short_candidates,
)
from tests.test_short_speech_edit import analysis


def evidence(c, *, people=True):
    time = (c['start_ms'] + c['end_ms']) // 2
    c.update(frame_in_window=True, frame_time_ms=time, quality_frame_time_ms=time,
             cue_time_ms=time)
    c.pop('brightness_samples', None)
    c['quality'].update(exposure=50, sharpness=30, brightness=.5,
                        face_count=int(people), person_count=0,
                        cue_method='opencv_cpu_haar_face_hog_person')
    return c


def segment(c, **extra):
    return dict(source_id=c['source_id'], raw_start_ms=c['start_ms'],
                raw_end_ms=c['end_ms'], reason='Selected evidence', **extra)


def run(pool, selected=()):
    a = analysis()
    a['candidates'] = pool
    before = deepcopy((a, pool))
    response = dict(interpretation='Family trip closing payoff', segments=list(selected), assumptions=[], coverage=dict(status='full', limitations=[]))
    result = complete(a, parse_goal(a['goal']), response, pool)
    assert (a, pool) == before
    assert sum(s['raw_end_ms'] - s['raw_start_ms'] for s in result['segments']) == 45000
    spec = _reconstruct(EpisodeContext('test', Path('/tmp'), a['goal']), a, parse_goal(a['goal']), result, 'llm')
    timeline, amended = build_timeline(spec, a)
    assert timeline.duration().to_seconds() == pytest.approx(45, abs=1/30)
    assert max(c.duration().to_seconds() for c in timeline.tracks[0]) <= 3.01
    cuts = result['segments']
    for i, s in enumerate(cuts):
        assert any(c['source_id'] == s['source_id'] and c['start_ms'] <= s['raw_start_ms']
                   < s['raw_end_ms'] <= c['end_ms'] for c in pool)
        for w in a['words']:
            if w['source_id'] == s['source_id']:
                assert s['raw_end_ms'] <= w['start_ms']-80 or s['raw_start_ms'] >= w['end_ms']+120
        for other in cuts[i+1:]:
            assert s['source_id'] != other['source_id'] or s['raw_end_ms'] <= other['raw_start_ms'] or other['raw_end_ms'] <= s['raw_start_ms']
    assert all(a['source_id'] != b['source_id'] or a['raw_end_ms'] != b['raw_start_ms'] for a,b in zip(cuts,cuts[1:]))
    return result


def matching(pool, s):
    return next(c for c in pool if c['source_id'] == s['source_id'] and c['start_ms'] == s['raw_start_ms'])


@pytest.mark.parametrize('labels', [['people'], ['scenic_wide']])
def test_selected_evidenced_payoff_and_hook_preserved(labels):
    pool = analysis()['candidates']
    hook = evidence(pool[1])
    payoff = evidence(pool[8], people=False)
    second = evidence(pool[15])
    result = run(pool, [segment(hook), segment(payoff, semantic_labels=labels), segment(second)])
    assert result['segments'][0] == segment(hook)
    assert all(matching(pool, s) in (payoff, second) for s in result['segments'][-2:])
    assert not any('Closing constraint' in l for l in result['coverage']['limitations'])


def test_unused_people_replaces_weak_tail_without_extending_source():
    pool = analysis()['candidates']
    for c in pool:
        c['quality']['motion'] = .4
    hook = evidence(pool[1])
    payoff = evidence(pool[-1])
    payoff['quality']['visual_score'] = -1000  # stays out of duration fill
    result = run(pool, [segment(hook)])
    assert result['segments'][0] == segment(hook)
    assert matching(pool, result['segments'][-1]) == payoff
    assert any('only final shot qualifies' in l for l in result['coverage']['limitations'])


@pytest.mark.parametrize('metric,value', [('exposure',14.99), ('sharpness',7.99), ('brightness',.119),
                                          ('exposure',None), ('sharpness',float('nan'))])
def test_dark_blurred_or_missing_quality_never_qualifies(metric,value):
    c = evidence(analysis()['candidates'][1])
    c['quality'][metric] = value
    assert not closing_qualifies(c, segment(c), ['scenic_wide'])


def test_exact_versioned_thresholds_qualify():
    c = evidence(analysis()['candidates'][1])
    c['quality'].update(exposure=CLOSING_MIN_EXPOSURE, sharpness=CLOSING_MIN_SHARPNESS, brightness=.12)
    assert closing_qualifies(c, segment(c))


@pytest.mark.parametrize('invalid', ['motion_only', 'no_frame', 'out_quality', 'out_cue', 'unavailable'])
def test_missing_or_out_of_window_evidence_is_not_a_payoff(invalid):
    c = evidence(analysis()['candidates'][1])
    if invalid == 'motion_only':
        c['quality'].update(face_count=0, person_count=0, motion=1)
    elif invalid == 'no_frame':
        c['frame_in_window'] = False
    elif invalid == 'out_quality':
        c['quality_frame_time_ms'] = c['end_ms']
    elif invalid == 'out_cue':
        c['cue_time_ms'] = c['end_ms']
    else:
        c['quality']['cue_method'] = 'unavailable:detector'
    assert not closing_qualifies(c, segment(c))


def test_no_qualifier_records_partial_without_invented_labels():
    result = run(analysis()['candidates'])
    assert result['coverage']['status'] == 'partial'
    assert any('Closing constraint unavailable' in l for l in result['coverage']['limitations'])
    assert all('semantic_labels' not in s for s in result['segments'])


def test_trimmed_candidate_must_keep_subject_and_quality_frame():
    c = evidence(analysis()['candidates'][1])
    s = segment(c)
    s['raw_end_ms'] = c['frame_time_ms']
    assert not closing_qualifies(c, s, ['scenic_wide'])


def test_new_preview_does_not_revalidate_inherited_outside_quality():
    a = analysis()
    source = a['sources'][0]
    source['keyframes'] = [dict(path='inside.jpg', source_time_ms=8000)]
    a['scenes'][0].update(thumbnail_path='outside.jpg', brightness=.01)
    base = dict(source_id=source['source_id'], start_ms=0, end_ms=20000,
                frame_in_window=True, frame_time_ms=1000, quality=dict(sharpness=99, exposure=99), audio={})
    candidates = short_candidates(a, [base])
    c = next(c for c in candidates if c['source_id'] == source['source_id'] and c['frame_in_window'])
    assert c['quality']['sharpness'] is None and c['quality']['exposure'] is None
    assert c['quality_frame_time_ms'] is None
    assert c['quality']['brightness'] is None
    assert not closing_qualifies(c, segment(c), ['scenic_wide'])


def test_scene_luma_uses_only_in_window_samples_and_trimmed_samples():
    a = analysis()
    source = a['sources'][0]
    source['keyframes'] = [dict(path='a.jpg', source_time_ms=7500), dict(path='b.jpg', source_time_ms=8500)]
    a['scenes'] = [dict(source_id=source['source_id'], start_ms=0, end_ms=20000, motion=.4, thumbnail_path=path, brightness=value)
                   for path,value in [('a.jpg',.1),('b.jpg',.5),('outside.jpg',0)]]
    base = dict(source_id=source['source_id'], start_ms=0, end_ms=20000,
                frame_in_window=True, frame_time_ms=7500, quality=dict(sharpness=99, exposure=99), audio={})
    c = next(c for c in short_candidates(a,[base]) if c['source_id'] == source['source_id'] and c['start_ms'] <= 7500 < c['end_ms'])
    assert c['quality']['brightness'] == pytest.approx(.3)
    c.update(cue_time_ms=7500)
    c['quality'].update(face_count=1, cue_method='local')
    s = segment(c)
    s['raw_end_ms'] = 8000
    assert not closing_qualifies(c,s)  # only dark .1 sample remains


@pytest.mark.parametrize('metric', ['sharpness', 'exposure'])
def test_bad_selected_quality_is_not_promoted_to_final_payoff(metric):
    pool = analysis()['candidates']
    hook = evidence(pool[1])
    good = evidence(pool[8])
    bad = evidence(pool[15])
    bad['quality'][metric] = 0
    result = run(pool, [segment(hook), segment(bad, semantic_labels=['scenic_wide']), segment(good)])
    assert matching(pool,result['segments'][-1]) == good
    assert matching(pool,result['segments'][-1]) != bad


def test_full_speech_candidate_cannot_be_truncated_to_replace_short_tail():
    from pipelines.ffmpeg.teaser_edit import enforce_closing
    pool = analysis()['candidates']
    first, weak = pool[1], pool[8]
    speech = evidence(deepcopy(pool[-1]))
    speech.update(edit_kind='speech', start_ms=10000, end_ms=12900)
    evidence(speech)
    entries = [(first,segment(first),()), (weak,segment(weak),())]
    assert enforce_closing(entries,[speech],{},set()) == entries


def test_overlapping_unused_candidate_cannot_replace_tail():
    from pipelines.ffmpeg.teaser_edit import enforce_closing
    pool = analysis()['candidates']
    first, weak = pool[1], pool[8]
    overlap = evidence(deepcopy(first))
    overlap.update(candidate_id='overlap', start_ms=first['start_ms']+100, end_ms=first['end_ms']+500)
    evidence(overlap)
    entries = [(first,segment(first),()), (weak,segment(weak),())]
    assert enforce_closing(entries,[overlap],{},set()) == entries


def test_existing_separate_quality_timestamp_survives_regeneration():
    a = analysis()
    source = a['sources'][0]
    source['keyframes'] = [dict(path='inside.jpg', source_time_ms=8000)]
    base = dict(source_id=source['source_id'], start_ms=0, end_ms=20000,
                frame_in_window=True, frame_time_ms=8000, quality_frame_time_ms=1000,
                quality=dict(sharpness=99, exposure=99), audio={})
    c = next(c for c in short_candidates(a,[base]) if c['source_id']==source['source_id'] and c['frame_in_window'])
    assert c['quality_frame_time_ms'] is None
    assert not closing_qualifies(c,segment(c),['scenic_wide'])


def test_two_unused_people_windows_can_replace_weak_tail():
    pool = analysis()['candidates']
    hook = evidence(pool[1])
    payoffs = [evidence(pool[-1]), evidence(pool[-8])]
    for c in payoffs:
        c['quality']['visual_score'] = -1000
    result = run(pool,[segment(hook)])
    assert result['segments'][0] == segment(hook)
    assert all(matching(pool,s) in payoffs for s in result['segments'][-2:])
    assert not any('Closing constraint' in l for l in result['coverage']['limitations'])


def test_touching_same_source_payoffs_are_ordered_without_coalescing():
    from pipelines.ffmpeg.teaser_edit import enforce_closing
    base = analysis()['candidates'][1]
    entries = []
    for i in range(5):
        c = deepcopy(base)
        c.update(candidate_id=str(i), start_ms=10000+i*2500, end_ms=12500+i*2500)
        if i >= 3:
            evidence(c)
        entries.append((c,segment(c),()))
    ordered = enforce_closing(entries,[],{},set())
    assert ordered[0] is entries[0]
    assert all(closing_qualifies(c,s,l) for c,s,l in ordered[-2:])
    assert all(a[1]['raw_end_ms'] != b[1]['raw_start_ms'] for a,b in zip(ordered,ordered[1:]))


def test_action_alone_does_not_qualify_as_people_or_scenic_closer():
    c=evidence(analysis()['candidates'][8],people=False)
    assert not closing_qualifies(c,segment(c),['action'])
