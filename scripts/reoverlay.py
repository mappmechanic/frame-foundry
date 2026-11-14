"""Re-render a saved Episode plan with current overlay style; zero Planner/ASR/TTS calls."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipelines.ffmpeg.contracts import EpisodeContext,StageError,read_json,atomic_write_json,validate_narrative,validate_report,usage_summary,failure_report,resolve_episode
from pipelines.ffmpeg import assembler,overlays
from pipelines.ffmpeg.qa_validator import validate_candidate
from pipelines.ffmpeg.evaluator import evaluate_episode,persist_report,footage_fingerprint


def rerender(source,episodes_root,episode_id,*,evals_root=None,judge=False,production=False,music_bed=None,music_authorized=False):
    source=Path(source).resolve();root=resolve_episode(episodes_root,episode_id)
    if root.exists():raise StageError('EPISODE_EXISTS','assembler','Use a fresh re-render Episode identity')
    analysis=read_json(source/'analysis.json');spec=read_json(source/'narrative-spec.json')
    if not spec.get('overlays'):raise StageError('OVERLAY_PLAN','assembler','Saved Episode has no source-bound overlay plan')
    previous=read_json(source/'eval-report.json')
    if previous.get('footage_hash')!=footage_fingerprint(analysis['sources']):
        raise StageError('SOURCE_CHANGED','assembler','Saved-plan footage hash differs from current originals')
    from pipelines.ffmpeg.disk_safety import require_free_disk
    require_free_disk(Path(episodes_root),stage='assembler')
    root.mkdir(parents=True)
    ctx=EpisodeContext(episode_id,root,spec['goal'],production=production,music_bed=Path(music_bed) if music_bed else None,music_authorized=music_authorized,fixture_mode=read_json(source/'qa-report.json').get('fixture_mode',False),target_seconds=spec['target_duration_ms']/1000,aspect_override=spec['aspect_ratio'],overlay_mode=True)
    try:
        # Keep literal source/word/frame evidence and saved intro, not old usage/J/Q.
        for name in ('frames','proxies'):
            directory=source/name
            if directory.exists():
                if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob('*')):raise StageError('SOURCE_REFERENCE','assembler','Cached evidence symlinks are unsupported')
                shutil.copytree(directory,root/name)
        voice=spec.get('voiceover')
        if voice:
            path=(source/voice['path']).resolve()
            if not path.is_relative_to(source) or path.is_symlink():raise StageError('SOURCE_REFERENCE','assembler','Unsafe cached intro path')
            destination=root/voice['path'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        analysis['episode_id']=spec['episode_id']=episode_id
        transcript=read_json(source/'transcript.json');transcript['episode_id']=episode_id
        atomic_write_json(root/'analysis.json',analysis);atomic_write_json(root/'transcript.json',transcript)
        spec['overlays']['version']='goal-overlays-2'
        spec['assumptions'].append('Reused saved source-grounded plan and intro; zero new Planner, ASR or TTS calls; old render judgment invalidated')
        validate_narrative(spec,analysis)
        result=assembler.assemble(ctx,spec,analysis)
        amended=read_json(result.spec_path)
        qa=validate_candidate(ctx,result,amended,judge_provider=None)
        qa['attempts']=[dict(index=1,stage='validator',passed=qa['passed'],reasons=[c['name'] for c in qa['checks'] if c['status']=='fail'],changes=['Saved plan re-render; overlay legibility style only'])]
        output='finished.mp4' if qa['passed'] else 'diagnostic.mp4'
        shutil.copy2(result.candidate_path,root/output);shutil.copy2(result.timeline_path,root/'timeline.otio');shutil.copy2(result.spec_path,root/'narrative-spec.json')
        qa.update(output_path='finished.mp4' if qa['passed'] else None,timeline_path='timeline.otio',stage='delivery' if qa['passed'] else 'validator')
        validate_report(qa);atomic_write_json(root/'qa-report.json',qa)
        active=None
        if judge and qa['passed']:
            from pipelines.ffmpeg.llm_provider import LLMProvider
            from pipelines.ffmpeg.budget import reserve_project_episode
            import os
            reserve_project_episode(Path(os.environ.get('FRAMEFOUNDRY_PROJECT_BUDGET_DIR',str(Path(__file__).resolve().parents[1]))),str(root))
            active=LLMProvider.from_env()
        evaluation=evaluate_episode(root,provider=active,evals_root=evals_root,persist=False)
        if not evaluation['gates']['passed']:
            if (root/'finished.mp4').exists():(root/'finished.mp4').replace(root/'diagnostic.mp4')
            qa.update(passed=False,result='failed',output_path=None,error=StageError('RERENDER_GATES','validator','Re-render failed independent hard gates; diagnostic retained').as_dict())
        qa['ai_usage'],qa['cost']=usage_summary(root);validate_report(qa);atomic_write_json(root/'qa-report.json',qa)
        persist_report(evaluation,root,evals_root)
        return root,qa
    except Exception as error:
        if not isinstance(error,StageError):error=StageError('RERENDER_FAILED','assembler','Saved-plan re-render failed: '+type(error).__name__)
        qa=failure_report(ctx,error);validate_report(qa);atomic_write_json(root/'qa-report.json',qa)
        raise error


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-episode',required=True)
    p.add_argument('--episodes-root',type=Path,default=Path(__file__).resolve().parents[1]/'episodes')
    p.add_argument('--evals-root',type=Path)
    p.add_argument('--episode',default='az-sedona-legible-'+datetime.now(timezone.utc).strftime('%H%M%S'))
    p.add_argument('--production',action='store_true')
    p.add_argument('--music-bed',type=Path)
    p.add_argument('--music-authorized',action='store_true')
    p.add_argument('--judge',action='store_true',help='Opt into one fresh Judge request; old J/Q are never reused')
    args=p.parse_args(argv)
    try:
        source=resolve_episode(args.episodes_root,args.from_episode)
        root,qa=rerender(source,args.episodes_root,args.episode,evals_root=args.evals_root,judge=args.judge,production=args.production,music_bed=args.music_bed,music_authorized=args.music_authorized)
    except (StageError,OSError,ValueError) as error:print(str(error),file=sys.stderr);return 2
    print('Run id: '+root.name+'; Planner/ASR/TTS calls: 0; QA '+('PASS' if qa['passed'] else 'FAIL'))
    return 0 if qa['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
