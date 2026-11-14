"""Real offline media regression for legacy QA recovery and diagnostic retention."""
from pathlib import Path
import hashlib
import pytest
from pipelines.ffmpeg import cli
from pipelines.ffmpeg.contracts import StageError, read_json, validate_report
from pipelines.ffmpeg.eval_gates import generate_distinct_footage
import json
from decimal import Decimal
from pipelines.ffmpeg.llm_provider import LLMProvider
from pipelines.ffmpeg.eval_judge import CRITERIA


def provider_transport(*,judge_fails=False,short_initial=False):
    calls=[]
    def send(url,headers,body,timeout):
        request=json.loads(body)
        prompt=json.loads(next(c['text'] for c in request['messages'][0]['content'] if c['type']=='text'))
        if 'candidates' in prompt:
            remaining=8000 if short_initial and not calls else 10000
            segments=[]
            for c in prompt['candidates']:
                n=min(c['end_ms']-c['start_ms'],remaining)
                if n:
                    segments.append(dict(source_id=c['source_id'],raw_start_ms=c['start_ms'],raw_end_ms=c['start_ms']+n,reason='Planner-secret-reason: locally sampled candidate'))
                    remaining-=n
            if judge_fails and calls:segments=list(reversed(segments))
            data=dict(interpretation='Observed highlights',assumptions=['Fixture test only'],segments=segments,coverage=dict(status='partial' if short_initial and not calls else 'full',limitations=['Short initial edit'] if short_initial and not calls else []))
            calls.append(('planner',prompt))
        else:
            ids=[s['shot_id'] for s in prompt['edit']['shots']]
            data=dict(scores={c:2 if judge_fails else 4 for c in CRITERIA},evidence={c:ids[:1] for c in CRITERIA},weakest_shot=ids[-1],fix_instruction='Reverse shot order for a stronger opening',verdict='fail' if judge_fails else 'pass',unsupported_goal=False,limitations=[])
            calls.append(('judge',prompt))
        return 200,json.dumps(dict(stop_reason='end_turn',content=[dict(type='text',text=json.dumps(data))],usage=dict(input_tokens=1000,output_tokens=100))).encode()
    provider=LLMProvider('anthropic','offline-not-a-real-key','claude-sonnet-4-5',(Decimal(3),Decimal(15)),transport=send)
    return provider,calls



@pytest.fixture(scope='module')
def media(tmp_path_factory):
    root=tmp_path_factory.mktemp('arizona-recovery-media')
    return root,generate_distinct_footage(root)


def reject_timeline(report):
    for check in report['checks']:
        if check['name']=='timeline_and_padding':
            check.update(status='fail',reason='Injected regression: cut order mismatch')
    report.update(passed=False,result='failed',output_path=None,error=StageError('QA_FAILED','validator','Technical QA failed').as_dict())
    validate_report(report)
    return report


def test_legacy_qa_failure_gets_one_planner_repair_before_delivery(media,tmp_path,monkeypatch):
    original=cli.validate_candidate
    count=0
    def validate(*args,**kwargs):
        nonlocal count
        count+=1
        report=original(*args,**kwargs)
        return reject_timeline(report) if count==1 else report
    monkeypatch.setattr(cli,'validate_candidate',validate)
    provider,calls=provider_transport()
    report=cli.run_episode(media[0],'Create a 10 second teaser',episodes_root=tmp_path/'episodes',episode_id='recover',transcript_fixture=media[1],provider=provider,evals_root=tmp_path/'evals')
    assert report['passed'],report
    assert [name for name,_ in calls]==['planner','planner','judge']
    assert 'Repair technical QA' in calls[1][1]['fix_instruction']
    assert len(report['attempts'])==2 and report['attempts'][0]['passed'] is False
    assert any('Planner retry' in c for c in report['attempts'][1]['changes'])
    episode=tmp_path/'episodes'/'recover'
    assert (episode/'finished.mp4').is_file()
    assert read_json(episode/'eval-report.json')['first_try_passed'] is False


def test_terminal_qa_failure_keeps_measurable_diagnostic_and_exact_timeline(media,tmp_path,monkeypatch):
    original=cli.validate_candidate
    monkeypatch.setattr(cli,'validate_candidate',lambda *a,**kw:reject_timeline(original(*a,**kw)))
    provider,calls=provider_transport()
    report=cli.run_episode(media[0],'Create a 10 second teaser',episodes_root=tmp_path/'episodes',episode_id='diagnostic',transcript_fixture=media[1],provider=provider,evals_root=tmp_path/'evals')
    episode=tmp_path/'episodes'/'diagnostic'
    assert not report['passed'] and report['output_path'] is None
    assert report['diagnostic_path']=='diagnostic.mp4'
    assert not (episode/'finished.mp4').exists()
    assert [name for name,_ in calls]==['planner','planner']
    assert len(report['attempts'])==2
    last=episode/'work'/'attempt-2'
    for root_name,work_name in [('diagnostic.mp4','candidate.mp4'),('timeline.otio','timeline.otio'),('narrative-spec.json','narrative-spec.json')]:
        assert hashlib.sha256((episode/root_name).read_bytes()).digest()==hashlib.sha256((last/work_name).read_bytes()).digest()
    evaluation=read_json(episode/'eval-report.json')
    checks={c['name']:c for c in evaluation['gates']['checks']}
    assert checks['artifacts_present']['status']=='fail'
    assert checks['readable_media']['status']=='pass' and checks['duration_target']['status']=='pass'
    assert evaluation['render_hash']==hashlib.sha256((episode/'diagnostic.mp4').read_bytes()).hexdigest()
    assert evaluation['scores']['G']==0 and evaluation['scores']['Q']==0
    assert (episode/'contact-sheet.jpg').is_file()
    assert all(c['reason'] for c in report['checks'] if c['status']=='fail')
    validate_report(read_json(episode/'qa-report.json'))


def test_cli_rejects_unsupported_python_before_dispatch(monkeypatch,capsys):
    import sys
    monkeypatch.setattr(sys,'version_info',(3,14,0))
    assert cli.main(['run'])==2
    assert 'requires Python >=3.10,<3.14' in capsys.readouterr().err


def test_report_contract_rejects_reasonless_failed_check():
    from pipelines.ffmpeg.contracts import EpisodeContext, failure_report
    report=failure_report(EpisodeContext('invalid-reason',Path('/tmp/unused-episode'),'goal'),StageError('QA_FAILED','validator','QA failed'))
    report['checks']=[dict(name='output_fps',status='fail',reason=None,required=True)]
    with pytest.raises(StageError):validate_report(report)
