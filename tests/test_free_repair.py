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
