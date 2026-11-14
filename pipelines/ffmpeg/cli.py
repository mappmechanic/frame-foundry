"""Footage directory + Goal -> one finished MP4 and schema-valid QA report."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,os,shutil,uuid,signal
from .contracts import *
from .episode import episode_lock,save_state
from .budget import reserve_project_episode
from .llm_provider import LLMProvider,ProviderError
from .refinery import run_refinery
from .planner import plan_teaser
from .assembler import assemble
from .qa_validator import validate_candidate

def run_episode(footage_dir,goal,*,episodes_root,episode_id=None,transcript_fixture=None,provider='env',visual_only=False):
    if visual_only and transcript_fixture is not None:raise StageError('INPUT_CONFLICT','refinery','Choose visual-only or a transcript fixture, not both')
    if not goal.strip():raise StageError('GOAL_EMPTY','refinery','Goal must not be blank')
    if not Path(footage_dir).is_dir():raise StageError('INPUT_EMPTY','refinery','Footage directory is unavailable')
    episode_id=episode_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:6]
    root=Path(episodes_root).resolve()
    with episode_lock(root):
        directory=resolve_episode(root,episode_id)
        if directory.exists():raise StageError('EPISODE_EXISTS','refinery','Episode identity already exists')
        directory.mkdir();ctx=EpisodeContext(episode_id,directory,goal,fixture_mode=transcript_fixture is not None);attempts=[]
        save_state(ctx,'accepted')
        try:
            active_provider=provider
            if provider=='env':
                try:active_provider=LLMProvider.from_env()
                except ProviderError:active_provider=None
            budget_note=None
            if active_provider is not None:
                try:reserve_project_episode(Path(os.environ.get('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(Path(__file__).resolve().parents[2]))),str(directory))
                except ProviderError as e:active_provider=None;budget_note=str(e)
            fixture=read_json(transcript_fixture) if transcript_fixture is not None else None
            def fixture_words(source):
                if fixture.get('schema_version')!='1.0':raise StageError('FIXTURE_INVALID','refinery','Unsupported fixture version')
                mapping=next((s for s in fixture.get('sources',[]) if s['source_id']==source['source_id']),None)
                if mapping is None or mapping['filename']!=Path(source['path']).name or abs(mapping['duration_ms']-source['duration_ms'])>150:
                    raise StageError('FIXTURE_INVALID','refinery','Fixture source mapping mismatch',source['source_id'])
                return [w for w in fixture['words'] if w['source_id']==source['source_id']]
            save_state(ctx,'refinery');analysis=run_refinery(ctx,footage_dir,transcriber=fixture_words if fixture is not None else (lambda source: []) if visual_only else None)
            analysis['transcription_status']='visual_only_unavailable' if visual_only else 'fixture' if fixture is not None else 'local_asr'
            atomic_write_json(directory/'analysis.json',analysis)
            if fixture and any(w['source_id'] not in {s['source_id'] for s in analysis['sources']} for w in fixture['words']):raise StageError('FIXTURE_INVALID','refinery','Unknown fixture word source')
            save_state(ctx,'planner');spec=plan_teaser(ctx,analysis,provider=active_provider)
            if visual_only:
                spec['assumptions'].append('Explicit visual-only diagnostic mode: speech not transcribed; word integrity unverified')
                spec['coverage']['status']='partial'
                spec['coverage']['limitations'].append('Speech-dependent goal coverage and word-cut integrity were not verified')
            if budget_note:spec['assumptions'].append('AI unavailable: '+budget_note)
            for index in range(1,4):
                try:
                    save_state(ctx,'assembler');result=assemble(ctx,spec,analysis,attempt=index)
                    amended=read_json(result.spec_path);save_state(ctx,'validator')
                    report=validate_candidate(ctx,result,amended,attempts=attempts,judge_provider=active_provider)
                    reasons=[c['name'] for c in report['checks'] if c['status']=='fail']
                    attempts.append(dict(index=index,stage='validator',passed=report['passed'],reasons=reasons,changes=[] if index==1 else ['Re-rendered with saved plan and independent normalization']))
                    report['attempts']=copy.deepcopy(attempts)
                    if report['passed']:
                        shutil.copy2(result.candidate_path,directory/'finished.mp4')
                        shutil.copy2(result.timeline_path,directory/'timeline.otio')
                        shutil.copy2(result.spec_path,directory/'narrative-spec.json')
                        report.update(output_path='finished.mp4',timeline_path='timeline.otio',stage='delivery')
                        validate_report(report);atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'completed');return report
                    repairable=set(reasons)<={'loudness','true_peak','contact_sheet','output_encoding','output_dimensions','output_fps','render_duration'}
                    if report['integrated_lufs'] is None:repairable=False
                    if not repairable or index==3:break
                except StageError as error:
                    attempts.append(dict(index=index,stage='assembler',passed=False,reasons=[error.code],changes=[]))
                    if not error.repairable or index==3:
                        report=failure_report(ctx,error);report['attempts']=copy.deepcopy(attempts);break
            validate_report(report);atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'failed');return report
        except (KeyboardInterrupt,SystemExit):
            error=StageError('INTERRUPTED','refinery','Episode interrupted');report=failure_report(ctx,error);report['attempts']=attempts
            atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'interrupted',error);raise
        except Exception as error:
            if not isinstance(error,StageError):error=StageError('PROCESS_FAILED','refinery','Episode processing failed')
            report=failure_report(ctx,error);report['attempts']=attempts;validate_report(report)
            atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'failed',error);return report

def _run_main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--footage-dir',required=True,type=Path);p.add_argument('--goal',required=True);p.add_argument('--episode');p.add_argument('--episodes-root',type=Path,default=Path(__file__).resolve().parents[2]/'episodes');p.add_argument('--transcript-fixture',type=Path)
    p.add_argument('--visual-only',action='store_true',help='Explicit diagnostic mode without ASR; speech integrity remains unverified')
    args=p.parse_args(argv)
    def terminate(signum,frame):raise KeyboardInterrupt()
    old=signal.signal(signal.SIGTERM,terminate)
    try:
        report=run_episode(args.footage_dir,args.goal,episodes_root=args.episodes_root,episode_id=args.episode,transcript_fixture=args.transcript_fixture,visual_only=args.visual_only)
        print(json.dumps(dict(episode_id=report['episode_id'],result=report['result'],passed=report['passed'],error=report['error'],tokens=dict(input=sum(c['input_tokens'] or 0 for c in report['ai_usage']),output=sum(c['output_tokens'] or 0 for c in report['ai_usage']),complete=all(c['input_tokens'] is not None and c['output_tokens'] is not None for c in report['ai_usage'] if c['status']!='blocked')),cost=report['cost'],goal_judge=report['goal_judge'],coverage=report['coverage'])))
        return 0 if report['passed'] else 1
    except StageError as e:print(e);return 2
    except KeyboardInterrupt:return 130
    finally:signal.signal(signal.SIGTERM,old)
def main(argv=None):
    import sys
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='run': args=args[1:]
    if args and args[0] in ('eval','rate','scoreboard'):
        from .eval_cli import main as eval_main
        return eval_main(args)
    return _run_main(args)

if __name__=='__main__':raise SystemExit(main())
