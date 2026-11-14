"""Source-time matching, never blind shot-index swapping or paid inference."""
from copy import deepcopy
from pathlib import Path
import pytest
from pipelines.ffmpeg import free_repair
from pipelines.ffmpeg.planner import _reconstruct
from pipelines.ffmpeg.goal_parser import parse_goal
from pipelines.ffmpeg.contracts import EpisodeContext
from pipelines.ffmpeg.assembler import build_timeline
from pipelines.ffmpeg.teaser_edit import complete
from tests.test_grounded_voiceover import framed


def inputs(tmp_path):
    a=framed(tmp_path)
    for c in a['candidates']:c['quality']['motion']=.2
    # Old montage has enough source diversity to leave spare positive people windows.
    response=dict(interpretation='Test montage',assumptions=[],coverage=dict(status='full',limitations=[]),segments=[])
    data=complete(a,parse_goal(a['goal']),response,a['candidates'])
    spec=_reconstruct(EpisodeContext('test',tmp_path,a['goal']),a,parse_goal(a['goal']),data,'deterministic')
    timeline,_=build_timeline(spec,a);clip=timeline.tracks[0][7]
    weak=dict(shot_id='shot-08',source_id=clip.metadata['source_id'],source_start_ms=round(clip.source_range.start_time.to_seconds()*1000),source_end_ms=round(clip.source_range.end_time_exclusive().to_seconds()*1000))
    # Stronger evidence is measured from a source-time cue, not fabricated labels.
    used=[(s['source_id'],s['raw_start_ms'],s['raw_end_ms']) for s in spec['segments']]
    for c in a['candidates']:
        if not any(c['source_id']==sid and c['start_ms']<b and start<c['end_ms'] for sid,start,b in used):
            c['quality'].update(motion=.8,cue_method='opencv_cpu_haar_face_hog_person;test_mock')
            a['visual_cues'][c['source_id']].append(dict(time_ms=c['frame_time_ms'],face_count=1,person_count=0,motion=.8,method='opencv_cpu_haar_face_hog_person;test_mock'))
    return spec,a,weak


def test_free_replacement_preserves_timeline_and_source_evidence(tmp_path):
    spec,a,weak=inputs(tmp_path);before=deepcopy((spec,a))
    result,note=free_repair.replace(spec,a,weak)
    assert 'replaced shot-08' in note
    old,_=build_timeline(spec,a);new,_=build_timeline(result,a)
    assert old.duration()==new.duration() and len(old.tracks[0])==len(new.tracks[0])
    assert result['segments'][0]==spec['segments'][0] and result['segments'][-2:]==spec['segments'][-2:]
    assert (spec,a)==before
    assert not (tmp_path/'llm-usage.json').exists() and not (tmp_path/'tts-usage.json').exists()


def test_different_interval_never_becomes_blind_eighth_shot_swap(tmp_path):
    spec,a,weak=inputs(tmp_path);weak['source_start_ms']+=500
    result,note=free_repair.replace(spec,a,weak)
    assert result==spec and 'no index-only swap' in note


def test_missing_prior_episode_is_safe_disclosed_noop(tmp_path):
    spec,a,_=inputs(tmp_path)
    result=free_repair.repair_from(spec,a,episodes_root=tmp_path,episode_id='az-0600')
    assert result['segments']==spec['segments']
    assert 'unavailable or stale' in result['assumptions'][-1]


def test_hook_and_closing_are_not_replaced(tmp_path):
    spec,a,weak=inputs(tmp_path);timeline,_=build_timeline(spec,a)
    for i in (0,len(timeline.tracks[0])-1):
        c=timeline.tracks[0][i];w=dict(weak,source_id=c.metadata['source_id'],source_start_ms=round(c.source_range.start_time.to_seconds()*1000),source_end_ms=round(c.source_range.end_time_exclusive().to_seconds()*1000))
        result,note=free_repair.replace(spec,a,w)
        assert result==spec and 'protected hook/closing' in note


def test_vo_named_subject_cannot_be_removed_by_free_repair(tmp_path):
    import hashlib
    spec,a,weak=inputs(tmp_path)
    timeline,_=build_timeline(spec,a);selection=list(timeline.tracks[0][7].metadata['selection_ids'])[0]
    s=next(s for s in spec['segments'] if s['segment_id']==selection)
    source=next(source for source in a['sources'] if source['source_id']==s['source_id'])
    frame=next(f for f in source['keyframes'] if s['raw_start_ms']<=f['source_time_ms']<s['raw_end_ms'])
    s['semantic_evidence']=dict(method='planner_in_window_tile',candidate_id='mock-frame',source_id=s['source_id'],source_start_ms=s['raw_start_ms'],source_end_ms=s['raw_end_ms'],frame_time_ms=frame['source_time_ms'],frame_path=frame['path'],frame_sha256=hashlib.sha256((tmp_path/frame['path']).read_bytes()).hexdigest(),view='center_crop_9_16',subjects=['market'],labels=['people'])
    result,note=free_repair.replace(spec,a,weak,keep_intro=True)
    assert result==spec and 'left unchanged' in note
    result,note=free_repair.replace(spec,a,weak,keep_intro=False)
    assert 'replaced shot-08' in note
    assert not any(s.get('semantic_evidence') for s in result['segments'])


def test_source_metadata_weak_guard_does_not_send_provider_call(tmp_path,monkeypatch):
    from pipelines.ffmpeg import llm_provider
    monkeypatch.setattr(llm_provider,'_transport',lambda *a:pytest.fail('No paid call'))
    spec,a,weak=inputs(tmp_path)
    result,note=free_repair.replace(spec,a,weak)
    assert 'replaced' in note and result!=spec


def test_repair_prior_symlink_is_not_followed(tmp_path):
    spec,a,_=inputs(tmp_path)
    outside=tmp_path/'outside';outside.mkdir()
    root=tmp_path/'episodes';root.mkdir();(root/'old').symlink_to(outside,target_is_directory=True)
    result=free_repair.repair_from(spec,a,episodes_root=root,episode_id='old')
    assert result['segments']==spec['segments'] and 'unavailable or stale' in result['assumptions'][-1]


def test_dark_or_low_quality_candidate_is_not_a_free_improvement(tmp_path):
    spec,a,weak=inputs(tmp_path)
    for c in a['candidates']:c['quality']['exposure']=1
    result,note=free_repair.replace(spec,a,weak)
    assert result==spec and 'left unchanged' in note


def test_next_best_safe_window_can_replace_weak_shot_without_higher_motion(tmp_path):
    spec,a,weak=inputs(tmp_path)
    for cues in a['visual_cues'].values():
        for cue in cues:
            if 'test_mock' in cue['method']:cue['motion']=.16
    result,note=free_repair.replace(spec,a,weak)
    assert 'replaced shot-08' in note and result!=spec


# Strict synthetic reports bind real fixture bytes and OTIO; not live media QA.
from tests.test_scoreboard_html import report


def prior_episode(tmp_path,spec,a,weak,report,*,eid='az-0835',created='2026-11-14T16:00:00Z'):
    import json
    import opentimelineio as otio
    from pipelines.ffmpeg.evaluator import footage_fingerprint,validate_eval_report
    from pipelines.ffmpeg.eval_judge import CRITERIA
    root=tmp_path/'episodes';directory=root/eid;directory.mkdir(parents=True)
    evaluation=report(eid,created=created)
    evaluation.update(goal=spec['goal'],footage_hash=footage_fingerprint(a['sources']))
    evaluation['scores']['J']=.75
    evaluation['judge'].update(status='scored',scores={c:4 for c in CRITERIA},evidence={c:[weak['shot_id']] for c in CRITERIA},
        weakest_shot=weak['shot_id'],fix_instruction='Replace weak shot',verdict='pass',unsupported_goal=False)
    validate_eval_report(evaluation)
    for name,payload in [('analysis.json',a),('narrative-spec.json',spec),('eval-report.json',evaluation),('qa-report.json',{'passed':True})]:
        (directory/name).write_text(json.dumps(payload))
    (directory/'finished.mp4').write_bytes(b'export')
    timeline,_=build_timeline(spec,a);otio.adapters.write_to_file(timeline,str(directory/'timeline.otio'))
    return root,directory


def test_latest_compatible_judge_is_reused_automatically_without_paid_call(tmp_path,report,monkeypatch):
    from pipelines.ffmpeg import llm_provider
    monkeypatch.setattr(llm_provider,'_transport',lambda *a:pytest.fail('No paid call'))
    spec,a,weak=inputs(tmp_path)
    root,_=prior_episode(tmp_path,spec,a,weak,report)
    result=free_repair.repair_latest(spec,a,episodes_root=root)
    assert result['segments']!=spec['segments']
    assert 'replaced shot-08' in result['assumptions'][-1]


@pytest.mark.parametrize('tamper',['finished.mp4','goal','footage','aspect','duration','directory_identity'])
def test_automatic_repair_never_uses_stale_or_incompatible_prior(tmp_path,report,tamper):
    import json
    spec,a,weak=inputs(tmp_path);root,directory=prior_episode(tmp_path,spec,a,weak,report)
    if tamper=='finished.mp4':(directory/tamper).write_bytes(b'changed render')
    elif tamper in ('goal','aspect','duration'):
        previous=json.loads((directory/'narrative-spec.json').read_text())
        key={'goal':'goal','aspect':'aspect_ratio','duration':'target_duration_ms'}[tamper]
        previous[key]={'goal':'Different goal','aspect':'16:9','duration':30000}[tamper]
        (directory/'narrative-spec.json').write_text(json.dumps(previous))
    else:
        evaluation=json.loads((directory/'eval-report.json').read_text())
        evaluation['footage_hash' if tamper=='footage' else 'episode_id']='c'*64 if tamper=='footage' else 'different'
        (directory/'eval-report.json').write_text(json.dumps(evaluation))
    result=free_repair.repair_latest(spec,a,episodes_root=root)
    assert result['segments']==spec['segments']


def test_latest_stale_judgment_does_not_fall_back_to_older_episode(tmp_path,report):
    spec,a,weak=inputs(tmp_path)
    root,_=prior_episode(tmp_path,spec,a,weak,report,eid='old',created='2026-11-14T15:00:00Z')
    _,latest=prior_episode(tmp_path,spec,a,weak,report)
    (latest/'finished.mp4').write_bytes(b'changed render')
    result=free_repair.repair_latest(spec,a,episodes_root=root)
    assert result['segments']==spec['segments'] and 'stale' in result['assumptions'][-1]
