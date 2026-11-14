"""Footage directory + Goal -> one finished MP4 and schema-valid QA report."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,os,shutil,uuid,signal
from .contracts import *
from .episode import episode_lock,save_state
from .budget import reserve_project_episode
from .llm_provider import LLMProvider,ProviderError,ProviderBudgetError
from .refinery import run_refinery
from .planner import plan_teaser
from .assembler import assemble
from .qa_validator import validate_candidate
from .transcription import TranscriptionRouter

def _retain_diagnostic(directory, result, report):
    """Keep failed media and its exact timeline/spec; never advertise delivery."""
    if result is None or not result.candidate_path.is_file():
        return
    shutil.copy2(result.candidate_path, directory/'diagnostic.mp4')
    shutil.copy2(result.timeline_path, directory/'timeline.otio')
    shutil.copy2(result.spec_path, directory/'narrative-spec.json')
    if (directory/'finished.mp4').is_file():
        (directory/'finished.mp4').replace(directory/'diagnostic-previous.mp4')
    report.update(output_path=None, diagnostic_path='diagnostic.mp4', timeline_path='timeline.otio')


def _judge_repair_fix(evaluation):
    """A scored weak edit can pass the rubric and still need one concrete repair."""
    judge = evaluation['judge']
    fix = judge.get('fix_instruction')
    if judge.get('status') != 'scored' or not isinstance(fix, str) or not fix.strip():
        return None
    j = evaluation['scores'].get('J')
    opening = (judge.get('scores') or {}).get('opening')
    weak = (judge.get('verdict') == 'fail'
            or (isinstance(j, (int, float)) and j <= 0.5)
            or (isinstance(opening, (int, float)) and opening <= 2))
    return fix.strip()[:160] if weak else None


def run_episode(footage_dir,goal,*,episodes_root,episode_id=None,transcript_fixture=None,provider='env',visual_only=False,evals_root=None,eval_labels=None,planner_retry=True,asr_provider='auto',voiceover=False,repair_from=None,target_seconds=None,aspect=None):
    if visual_only and transcript_fixture is not None:raise StageError('INPUT_CONFLICT','refinery','Choose visual-only or a transcript fixture, not both')
    if not goal.strip():raise StageError('GOAL_EMPTY','refinery','Goal must not be blank')
    if not Path(footage_dir).is_dir():raise StageError('INPUT_EMPTY','refinery','Footage directory is unavailable')
    from .goal_parser import parse_goal
    parse_goal(goal,target_seconds=target_seconds,aspect=aspect)
    episode_id=episode_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:6]
    root=Path(episodes_root).resolve()
    with episode_lock(root):
        directory=resolve_episode(root,episode_id)
        if directory.exists():raise StageError('EPISODE_EXISTS','refinery','Episode identity already exists')
        directory.mkdir();ctx=EpisodeContext(episode_id,directory,goal,fixture_mode=transcript_fixture is not None,target_seconds=target_seconds,aspect_override=aspect);attempts=[]
        save_state(ctx,'accepted');report=None;evaluation=None;active_provider=None;planning_retried=False;first_eval_passed=None;last_result=None;next_changes=[]
        try:
            if voiceover:
                from .voiceover import configuration
                configuration()  # fail before any paid ASR/Planner work
            active_provider=provider
            if provider=='env':
                try:active_provider=LLMProvider.from_env()
                except ProviderBudgetError:raise
                except ProviderError:active_provider=None
            budget_note=None
            cloud_asr_enabled=not visual_only and transcript_fixture is None and asr_provider!='local' and bool(os.environ.get('OPENAI_API_KEY','').strip())
            if active_provider is not None or cloud_asr_enabled or voiceover:
                try:reserve_project_episode(Path(os.environ.get('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(Path(__file__).resolve().parents[2]))),str(directory))
                except ProviderBudgetError:raise
                except ProviderError as e:
                    if cloud_asr_enabled or voiceover:raise ProviderBudgetError('Project allowance unavailable; ASR and inference blocked') from None
                    active_provider=None;budget_note=str(e)
            fixture=read_json(transcript_fixture) if transcript_fixture is not None else None
            def fixture_words(source):
                if fixture.get('schema_version')!='1.0':raise StageError('FIXTURE_INVALID','refinery','Unsupported fixture version')
                mapping=next((s for s in fixture.get('sources',[]) if s['source_id']==source['source_id']),None)
                if mapping is None or mapping['filename']!=Path(source['path']).name or abs(mapping['duration_ms']-source['duration_ms'])>150:
                    raise StageError('FIXTURE_INVALID','refinery','Fixture source mapping mismatch',source['source_id'])
                return [w for w in fixture['words'] if w['source_id']==source['source_id']]
            router=TranscriptionRouter(ctx,mode=asr_provider) if fixture is None and not visual_only else None
            save_state(ctx,'refinery');analysis=run_refinery(ctx,footage_dir,transcriber=fixture_words if fixture is not None else (lambda source: []) if visual_only else router)
            analysis['transcription_status']='visual_only_unavailable' if visual_only else 'fixture' if fixture is not None else router.status
            if router is not None:analysis['transcription_sources']=router.sources
            atomic_write_json(directory/'analysis.json',analysis)
            if fixture and any(w['source_id'] not in {s['source_id'] for s in analysis['sources']} for w in fixture['words']):raise StageError('FIXTURE_INVALID','refinery','Unknown fixture word source')
            save_state(ctx,'planner');spec=plan_teaser(ctx,analysis,provider=active_provider)
            if repair_from:
                from .free_repair import repair_from as reuse_judge
                spec=reuse_judge(spec,analysis,episodes_root=root,episode_id=repair_from,keep_intro=voiceover)
            if visual_only:
                spec['assumptions'].append('Explicit visual-only diagnostic mode: speech not transcribed; word integrity unverified')
                spec['coverage']['status']='partial'
                spec['coverage']['limitations'].append('Speech-dependent goal coverage and word-cut integrity were not verified')
            if budget_note:spec['assumptions'].append('AI unavailable: '+budget_note)
            for index in range(1,4):
                try:
                    # A Planner repair must not erase source-level ASR limitations.
                    if router is not None:
                        degraded=[audit['limitation'] for audit in router.sources.values() if audit.get('status')=='degraded']
                        if degraded:
                            spec['coverage']['status']='partial'
                            spec['coverage']['limitations'].extend(note for note in degraded if note not in spec['coverage']['limitations'])
                    if voiceover:
                        from .voiceover import prepare_intro
                        spec['voiceover']=prepare_intro(ctx,spec)
                    save_state(ctx,'assembler');result=assemble(ctx,spec,analysis,attempt=index);last_result=result
                    amended=read_json(result.spec_path);save_state(ctx,'validator')
                    report=validate_candidate(ctx,result,amended,attempts=attempts,judge_provider=None)
                    reasons=[c['name'] for c in report['checks'] if c['status']=='fail']
                    attempts.append(dict(index=index,stage='validator',passed=report['passed'],reasons=reasons,changes=list(next_changes)))
                    report['attempts']=copy.deepcopy(attempts)
                    next_changes=[]
                    if not report['passed'] and first_eval_passed is None:first_eval_passed=False
                    if report['passed']:
                        shutil.copy2(result.candidate_path,directory/'finished.mp4')
                        shutil.copy2(result.timeline_path,directory/'timeline.otio')
                        shutil.copy2(result.spec_path,directory/'narrative-spec.json')
                        report.update(output_path='finished.mp4',timeline_path='timeline.otio',stage='delivery')
                        validate_report(report);atomic_write_json(directory/'qa-report.json',report)
                        from .evaluator import evaluate_episode
                        evaluation=evaluate_episode(directory,provider=active_provider,evals_root=evals_root,labels=eval_labels,persist=False,progress=lambda stage:save_state(ctx,stage))
                        if first_eval_passed is None:first_eval_passed=evaluation['gates']['passed']
                        failed_gates={c['name'] for c in evaluation['gates']['checks'] if c['status']=='fail'}
                        repairable_eval=bool(failed_gates & {'duration_target','word_cut_integrity','black_frames','frozen_frames','repeated_shots'})
                        judge_fix=_judge_repair_fix(evaluation)
                        if planner_retry and not planning_retried and active_provider is not None and index<3 and (repairable_eval or judge_fix):
                            # Preserve the exact judged attempt in existing strict string fields.
                            # The next render is evaluated afresh; its J/Q must never inherit these.
                            if evaluation['judge']['status']=='scored':
                                attempts[-1]['changes'].append(json.dumps(dict(
                                    kind='judged_attempt', attempt=index,
                                    render_hash=evaluation['render_hash'],
                                    scores=evaluation['scores'], judge=evaluation['judge']),
                                    sort_keys=True, allow_nan=False))
                            planning_retried=True
                            fix=judge_fix or 'Repair failed deterministic gates: '+', '.join(sorted(failed_gates))
                            save_state(ctx,'planner')
                            spec=plan_teaser(ctx,analysis,provider=active_provider,retry_fix=fix[:160])
                            if visual_only:
                                spec['coverage']['status']='partial';spec['coverage']['limitations'].append('Speech-dependent coverage and word integrity unverified in explicit visual-only mode')
                            next_changes=['One evaluation-driven Planner retry: '+fix[:160]]
                            attempts[-1]['changes'].extend(next_changes)
                            evaluation=None
                            continue
                        if failed_gates & {'duration_target','voiceover_integrity'}:
                            code='EVAL_VOICEOVER_FAILED' if 'voiceover_integrity' in failed_gates else 'EVAL_DURATION_FAILED'
                            report.update(passed=False,result='failed',output_path=None,error=StageError(code,'validator','Evaluation failed: '+', '.join(sorted(failed_gates))+'; clip retained only as diagnostic.mp4').as_dict())
                            # Keep measured render evidence and G=0/Q=0 in the final evaluation.
                        if evaluation['judge'].get('reason','') and evaluation['judge']['reason'].startswith('Judge blocked by budget:'):
                            report.update(passed=False,result='failed',output_path=None,error=StageError('BUDGET_LIMIT','validator',evaluation['judge']['reason']).as_dict())
                        return report
                    # A failed legacy QA must not bypass the bounded Planner repair.
                    # Encoding/audio defects remain deterministic renderer repairs.
                    if planner_retry and not planning_retried and active_provider is not None and index<3 and set(reasons)&{'timeline_and_padding','goal_duration','render_duration'}:
                        planning_retried=True
                        fix=('Repair technical QA: '+', '.join(reasons)+f'. Target {amended["target_duration_ms"]}ms; measured {report["actual_duration_ms"]}ms.')[:160]
                        save_state(ctx,'planner')
                        spec=plan_teaser(ctx,analysis,provider=active_provider,retry_fix=fix)
                        if visual_only:
                            spec['coverage']['status']='partial';spec['coverage']['limitations'].append('Speech-dependent coverage and word integrity unverified in explicit visual-only mode')
                        next_changes=['One technical-QA-driven Planner retry: '+fix]
                        attempts[-1]['changes'].extend(next_changes)
                        evaluation=None
                        continue
                    repairable=set(reasons)<={'loudness','true_peak','contact_sheet','output_encoding','output_dimensions','output_fps','render_duration'}
                    if report['integrated_lufs'] is None:repairable=False
                    if not repairable or index==3:break
                    next_changes=['Re-rendered with saved plan and independent normalization']
                except StageError as error:
                    if attempts and attempts[-1]['index']==index:
                        attempts[-1].update(stage=error.stage,passed=False,reasons=[error.code])
                    else:attempts.append(dict(index=index,stage='assembler',passed=False,reasons=[error.code],changes=[]))
                    if not error.repairable or index==3:
                        report=failure_report(ctx,error);report['attempts']=copy.deepcopy(attempts);break
            validate_report(report);atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'failed');return report
        except (KeyboardInterrupt,SystemExit):
            error=StageError('INTERRUPTED','refinery','Episode interrupted');report=failure_report(ctx,error);report['attempts']=attempts
            atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'interrupted',error);raise
        except Exception as error:
            if isinstance(error,ProviderBudgetError):error=StageError('BUDGET_LIMIT','planner',str(error))
            if not isinstance(error,StageError):error=StageError('PROCESS_FAILED','refinery','Episode processing failed')
            report=failure_report(ctx,error);report['attempts']=attempts;validate_report(report)
            atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'failed',error);return report
        finally:
            if report is not None:
                from .evaluator import evaluate_episode,persist_report
                try:
                    if not report['passed']:
                        _retain_diagnostic(directory,last_result,report)
                        evaluation=None  # Re-measure retained diagnostic, never a stale public artifact.
                    # QA stays schema-valid and current even when the judge added usage.
                    report['ai_usage'],report['cost']=usage_summary(directory)
                    report['elapsed_seconds']=__import__('time').monotonic()-ctx.started
                    validate_report(report);atomic_write_json(directory/'qa-report.json',report)
                    if evaluation is None:
                        evaluation=evaluate_episode(directory,provider=active_provider if report['passed'] else None,evals_root=evals_root,labels=eval_labels,persist=False,progress=lambda stage:save_state(ctx,stage))
                    evaluation['first_try_passed']=evaluation['gates']['passed'] if first_eval_passed is None else first_eval_passed
                    judge=evaluation['judge']
                    if judge['status']=='scored':
                        report['goal_judge']=dict(status='scored',score=round(evaluation['scores']['J']*100,3),rationale='Evaluation rubric: '+judge['verdict'],evidence=sorted({sid for ids in judge['evidence'].values() for sid in ids}),provider=judge['provider'],model=judge['model'])
                        if judge['unsupported_goal'] or judge['verdict']=='fail':
                            report['coverage']['status']='partial';report['coverage']['limitations'].extend(judge['limitations'] or ['Evaluation judge flags partial Goal coverage'])
                            if report['passed']:report['result']='partial'
                    else:
                        report['goal_judge']=unavailable_judge(judge['reason'] or 'Evaluation judge unavailable')
                    report['ai_usage'],report['cost']=usage_summary(directory)
                    report['elapsed_seconds']=__import__('time').monotonic()-ctx.started
                    evaluation['timing']['elapsed_seconds']=report['elapsed_seconds']
                    evaluation['provenance']['pr']=os.environ.get('FRAMEFOUNDRY_PR',evaluation['provenance'].get('pr'))
                    validate_report(report);atomic_write_json(directory/'qa-report.json',report)
                    persist_report(evaluation,directory,evals_root)
                    state=read_json(directory/'state.json')['stage']
                    if state!='interrupted':save_state(ctx,'completed' if report['passed'] else 'failed')
                except Exception as eval_error:
                    report.update(passed=False,result='failed',output_path=None,error=StageError('EVAL_PERSISTENCE_FAILED','validator','Evaluation or persistence failed: '+type(eval_error).__name__).as_dict())
                    _retain_diagnostic(directory,last_result,report)
                    atomic_write_json(directory/'qa-report.json',report);save_state(ctx,'failed')


def _run_main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--footage-dir',required=True,type=Path);p.add_argument('--goal',required=True);p.add_argument('--episode');p.add_argument('--episodes-root',type=Path,default=Path(__file__).resolve().parents[2]/'episodes');p.add_argument('--transcript-fixture',type=Path)
    p.add_argument('--evals-root',type=Path);p.add_argument('--eval-labels',type=Path);p.add_argument('--no-planner-retry',action='store_true')
    p.add_argument('--asr-provider',choices=['auto','local','openai'],default='auto',help='Prefer local WhisperX; auto falls back to OpenAI word timestamps when a key is configured')
    p.add_argument('--visual-only',action='store_true',help='Explicit diagnostic mode without ASR; speech integrity remains unverified')
    p.add_argument('--repair-from',help='Optional prior episode whose scored weakest source interval can be replaced locally with no paid call')
    p.add_argument('--voiceover',action='store_true',help='Opt in to cached ElevenLabs intro narration; requires ELEVENLABS_API_KEY')
    p.add_argument('--duration-seconds',type=float,help='Explicit duration override, 5–120 seconds')
    p.add_argument('--aspect',choices=['9:16','16:9'],help='Explicit aspect override without changing original Goal')
    args=p.parse_args(argv)
    def terminate(signum,frame):raise KeyboardInterrupt()
    old=signal.signal(signal.SIGTERM,terminate)
    try:
        report=run_episode(args.footage_dir,args.goal,episodes_root=args.episodes_root,episode_id=args.episode,transcript_fixture=args.transcript_fixture,visual_only=args.visual_only,evals_root=args.evals_root,eval_labels=args.eval_labels,planner_retry=not args.no_planner_retry,asr_provider=args.asr_provider,voiceover=args.voiceover,repair_from=args.repair_from,target_seconds=args.duration_seconds,aspect=args.aspect)
        evaluation_path=Path(args.episodes_root)/report['episode_id']/'eval-report.json'
        evaluation=read_json(evaluation_path) if evaluation_path.is_file() else None
        metrics=dict(Q=evaluation['scores']['Q'],J=evaluation['scores']['J'],D=evaluation['scores']['D'],G=evaluation['scores']['G'],quality_per_dollar=evaluation['model_cost']['quality_per_dollar']) if evaluation else dict(Q=None,J=None,D=None,G=None,quality_per_dollar=None)
        durations=None
        directory=Path(args.episodes_root)/report['episode_id']
        if (directory/'analysis.json').is_file() and (directory/'narrative-spec.json').is_file():
            from . import teaser_edit
            analysis=read_json(directory/'analysis.json');spec=read_json(directory/'narrative-spec.json')
            pool=teaser_edit.short_candidates(analysis,analysis.get('candidates',[])) if teaser_edit.enabled(analysis) else analysis.get('candidates',[])
            durations=teaser_edit.duration_evidence(analysis,pool,spec['segments'])
            durations['selected_encoded_duration_ms']=report.get('actual_duration_ms')
        summary_status='PASS' if report['passed'] and metrics['G']==1 else 'QA_PASS_EVAL_INCOMPLETE' if report['passed'] else 'FAIL'
        coverage_note='Partial content coverage, not a technical failure: '+ '; '.join(report['coverage']['limitations']) if report['passed'] and report['coverage']['status']=='partial' else None
        if coverage_note and durations:
            coverage_note+=f". Local duration evidence: {durations['source_duration_ms']/1000:.3f}s raw across {durations['source_count']} clips; {durations['usable_short_windows_ms']/1000:.3f}s usable short windows; {(durations['selected_encoded_duration_ms'] or 0)/1000:.3f}s selected output"
        print(json.dumps(dict(status=summary_status,coverage_note=coverage_note,duration_evidence=durations,evaluation=metrics,episode_id=report['episode_id'],result=report['result'],passed=report['passed'],error=report['error'],output_path=report['output_path'],diagnostic_path=report.get('diagnostic_path'),tokens=dict(input=sum(c['input_tokens'] or 0 for c in report['ai_usage']),output=sum(c['output_tokens'] or 0 for c in report['ai_usage']),complete=all(c['input_tokens'] is not None and c['output_tokens'] is not None for c in report['ai_usage'] if c['status']!='blocked' and c['role'] not in ('transcription','tts'))),cost=report['cost'],goal_judge=report['goal_judge'],coverage=report['coverage'])))
        return 0 if report['passed'] else 1
    except StageError as e:print(e);return 2
    except KeyboardInterrupt:return 130
    finally:signal.signal(signal.SIGTERM,old)
def main(argv=None):
    import sys
    if not (3,10) <= sys.version_info[:2] < (3,14):
        print('FrameFoundry requires Python >=3.10,<3.14 with OTIO 0.18.1; recreate .venv with uv venv -p 3.13 --clear',file=sys.stderr)
        return 2
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='run': args=args[1:]
    if args and args[0]=='serve':
        from .local_server import main as serve_main
        return serve_main(args[1:])
    if args and args[0] in ('eval','rate','scoreboard'):
        from .eval_cli import main as eval_main
        return eval_main(args)
    return _run_main(args)

if __name__=='__main__':raise SystemExit(main())
