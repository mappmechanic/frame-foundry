"""Independent hard media gates. Advisory AI cannot make a failed gate pass."""
from __future__ import annotations
from .media_tools import media_binary
# Support the repository guardrail's direct-script entrypoint as well as -m.
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='pipelines.ffmpeg'
from pathlib import Path
import argparse,copy,json,math,re,time
from fractions import Fraction
import opentimelineio as otio
from .contracts import *
from .proxy import probe,run
from .contact_sheet import build_contact_sheet
from .assembler import pad_word_interval
from .goal_judge import judge_goal
from .render import output_frames

_TIMELINE_TOLERANCE_MS = 1.0

def verify_constant_fps(path, video):
    """Verify decoded presentation timestamps, not rounded/aggregate rate tags."""
    try:
        data=json.loads(run([media_binary('ffprobe'),'-v','error','-select_streams','v:0',
            '-show_frames','-show_entries','frame=best_effort_timestamp,pkt_duration',
            '-of','json',str(path)]).stdout)
        tick=Fraction(video['time_base'])
        if tick<=0:raise ValueError('nonpositive time base')
        tolerance=min(Fraction(1,1000),max(tick,Fraction(1,1000000)))
        frames=data['frames']
        if len(frames)<2:raise ValueError('need at least two decoded frame timestamps')
        stamps=[int(f['best_effort_timestamp'])*tick for f in frames]
        for index,(previous,current) in enumerate(zip(stamps,stamps[1:]),1):
            delta=current-previous
            expected=stamps[0]+Fraction(index,30)
            if delta<=0 or abs(delta-Fraction(1,30))>tolerance or abs(current-expected)>tolerance:
                return False,(f'Expected constant 30 fps (33.333333ms/frame); frame {index} '
                    f'observed interval {float(delta)*1000:.6f}ms, grid drift '
                    f'{float(current-expected)*1000:.6f}ms; tolerance {float(tolerance)*1000:.6f}ms')
        for index,frame in enumerate(frames):
            if frame.get('pkt_duration') is not None:
                duration=int(frame['pkt_duration'])*tick
                if abs(duration-Fraction(1,30))>tolerance:
                    return False,f'Expected 33.333333ms frame duration; frame {index} observed {float(duration)*1000:.6f}ms'
        return True,None
    except Exception as exc:
        return False,f'Cannot verify constant 30 fps timestamps: {exc}; re-render CFR and probe decoded video frames'


def measure_audio(path):
    result=run([media_binary('ffmpeg'),'-hide_banner','-i',str(path),'-af','loudnorm=I=-14:TP=-1:LRA=11:print_format=json','-f','null','-'])
    match=re.search(r'{\s*"input_i".*?}',result.stderr,re.S)
    if not match:return dict(integrated_lufs=None,true_peak_dbtp=None)
    metrics=json.loads(match.group());values=[]
    for field in ['input_i','input_tp']:
        try:x=float(metrics[field]);values.append(x if math.isfinite(x) else None)
        except (ValueError,KeyError):values.append(None)
    return dict(integrated_lufs=values[0],true_peak_dbtp=values[1])

_DEFAULT=object()
def validate_candidate(ctx,result,spec,*,attempts=None,judge_provider=_DEFAULT):
    report=failure_report(ctx,StageError('QA_FAILED','validator','Technical QA failed'))
    report.update(interpretation=spec['interpretation'],assumptions=list(spec['assumptions']),coverage=copy.deepcopy(spec['coverage']),exclusions=spec['exclusions'],
                  target_duration_ms=spec['target_duration_ms'],attempts=attempts or [],timeline_path=str(result.timeline_path.relative_to(ctx.episode_dir)),limitations=list(result.limitations))
    checks=[]
    def check(name,passed,reason):checks.append(dict(name=name,status='pass' if passed else 'fail',reason=reason if not passed else None,required=True))
    try:validate_narrative(spec);check('narrative_schema',True,None)
    except (StageError,ValueError) as exc:check('narrative_schema',False,f'Narrative contract invalid: {exc}')
    try:
        info=probe(result.candidate_path);data=info['data'];v=info['video'];a=next((s for s in data['streams'] if s['codec_type']=='audio'),{})
        report['actual_duration_ms']=info['duration_ms'];check('readable_media',True,None)
        size=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
        check('output_encoding',v.get('codec_name')=='h264' and a.get('codec_name')=='aac',f"Expected H.264/AAC; observed video={v.get('codec_name')}, audio={a.get('codec_name')}; re-render delivery codecs")
        check('output_dimensions',(v.get('width'),v.get('height'))==size,f"Expected {size[0]}x{size[1]}; observed {v.get('width')}x{v.get('height')}; re-render requested aspect ratio")
        fps_ok,fps_reason=verify_constant_fps(result.candidate_path,v)
        check('output_fps',fps_ok,fps_reason)
    except Exception as exc:check('readable_media',False,f'Output cannot be probed: {exc}; inspect candidate media');info=None
    try:
        metrics=measure_audio(result.candidate_path);report.update(metrics)
        check('loudness',metrics['integrated_lufs'] is not None and -15<=metrics['integrated_lufs']<=-13,f"Expected finite LUFS in [-15,-13]; observed {metrics['integrated_lufs']}; normalize audio")
        check('true_peak',metrics['true_peak_dbtp'] is not None and metrics['true_peak_dbtp']<=-1,f"Expected true peak <= -1 dBTP; observed {metrics['true_peak_dbtp']}; reduce encoded peak and remeasure")
    except Exception as exc:check('loudness',False,f'Audio measurement unavailable: {exc}; inspect audio stream');check('true_peak',False,f'Peak measurement unavailable: {exc}; inspect audio stream')
    try:
        timeline=otio.adapters.read_from_file(str(result.timeline_path));video=next(t for t in timeline.tracks if t.name=='V1')
        expected_ids=[s['segment_id'] for s in spec['segments']];seen=[];cursor_frames=0;cuts=[];sources={s['source_id']:s for s in spec['sources']}
        words=read_json(ctx.episode_dir/'transcript.json')['words'] if (ctx.episode_dir/'transcript.json').exists() else []
        expected_segments={s['segment_id']:s for s in spec['segments']}
        for clip in video:
            if cursor_frames:cuts.append(round(cursor_frames*1000/30))
            sid=clip.metadata['source_id'];source=sources[sid]
            if Path(clip.media_reference.target_url).resolve()!=Path(source['path']).resolve() or not Path(source['path']).is_file():raise ValueError(f'Source {sid}: expected existing original {source["path"]}; observed {clip.media_reference.target_url}')
            start=clip.source_range.start_time.to_seconds()*1000;end=start+clip.source_range.duration.to_seconds()*1000
            if not (math.isfinite(start) and math.isfinite(end) and -_TIMELINE_TOLERANCE_MS<=start<end<=source['duration_ms']+_TIMELINE_TOLERANCE_MS):raise ValueError(f'Source {sid}: expected range within [0,{source["duration_ms"]}]ms (1ms tolerance); observed [{start},{end}]ms')
            selected_ids=list(clip.metadata['selection_ids']);seen.extend(selected_ids)
            for segment_id in selected_ids:
                segment=expected_segments[segment_id]
                if segment['source_id']!=sid or not start-_TIMELINE_TOLERANCE_MS<=segment['render_start_ms']<segment['render_end_ms']<=end+_TIMELINE_TOLERANCE_MS:raise ValueError(f'Selection {segment_id}: expected source {segment["source_id"]} range [{segment["render_start_ms"]},{segment["render_end_ms"]}]ms contained in clip; observed source {sid} range [{start},{end}]ms (1ms tolerance)')
                chosen=[w for w in words if w['source_id']==sid and w['start_ms']<=segment['raw_end_ms'] and segment['raw_start_ms']<=w['end_ms']]
                if chosen:
                    x,y=pad_word_interval(min(w['start_ms'] for w in chosen),max(w['end_ms'] for w in chosen),source['duration_ms'],[w for w in words if w['source_id']==sid])
                    if start>x+_TIMELINE_TOLERANCE_MS or end<y-_TIMELINE_TOLERANCE_MS:raise ValueError(f'Selection {segment_id}: expected protected speech range [{x},{y}]ms; observed clip [{start},{end}]ms (1ms tolerance)')
                    if min(w['start_ms'] for w in chosen)<80 or max(w['end_ms'] for w in chosen)+120>source['duration_ms']:
                        report['limitations'].append('Full word padding unavailable at source boundary')
            cursor_frames+=output_frames(clip.source_range.duration.to_seconds())
        cuts_ok=len(cuts)==len(result.cut_times_ms) and all(abs(a-b)<=_TIMELINE_TOLERANCE_MS for a,b in zip(cuts,result.cut_times_ms))
        check('timeline_and_padding',seen==expected_ids and cuts_ok,f'Expected selection order {expected_ids} and 30fps cuts {cuts}ms; observed order {seen} and cuts {result.cut_times_ms}ms (1ms tolerance)')
        cursor=cursor_frames*1000/30
        check('render_duration',info is not None and abs(info['duration_ms']-cursor)<=150,f"Expected saved timeline on 30fps grid {cursor:.6f}ms ±150ms; observed {info['duration_ms'] if info else None}ms; inspect render cuts")
        check('goal_duration',info is not None and (abs(info['duration_ms']-spec['target_duration_ms'])<=1000 or (spec['coverage']['status']=='partial' and bool(spec['coverage']['limitations']))),f"Expected target {spec['target_duration_ms']}ms ±1000ms or documented partial coverage; observed {info['duration_ms'] if info else None}ms")
    except Exception as exc:check('timeline_and_padding',False,f'Invalid timeline/source/order/padding: {exc}; repair saved timeline');check('render_duration',False,f'Timeline duration unavailable: {exc}')
    try:
        if spec.get('editing_policy'):build_contact_sheet(result.candidate_path,result.cut_times_ms,ctx.episode_dir/'contact-sheet.jpg',production=True)
        else:build_contact_sheet(result.candidate_path,result.cut_times_ms,ctx.episode_dir/'contact-sheet.jpg')
        report['contact_sheet_path']='contact-sheet.jpg';check('contact_sheet',True,None)
    except Exception as exc:check('contact_sheet',False,f'Cut-boundary evidence unavailable: {exc}; regenerate contact sheet')
    from .overlays import verify
    if spec.get('overlays'):
        try:checks.extend(verify(spec,result.candidate_path,result.cut_times_ms,result.duration_ms))
        except Exception:checks.append(dict(name='overlay_render_binding',status='fail',reason='Overlay verification unavailable',required=True))
        report['limitations'].extend(spec['overlays']['limitations'])
    else:report['limitations'].append('Overlays disabled; overlay checks not applicable')
    from .production import verify as verify_policy
    checks.extend(verify_policy(spec,{},root=ctx.episode_dir))
    report['checks']=checks;technical=all(c['status']=='pass' for c in checks)
    if technical:
        judge=judge_goal(ctx,spec,result,words=words) if judge_provider is _DEFAULT else judge_goal(ctx,spec,result,provider=judge_provider,words=words)
        report['goal_judge']=judge
        if judge['status']=='scored' and judge['score']<60:
            report['coverage']['status']='partial';report['coverage']['limitations'].append('AI goal-coverage score below assumed 60/100 threshold')
        report.update(passed=True,result='partial' if report['coverage']['status']=='partial' else 'passed',error=None,output_path=str(result.candidate_path.relative_to(ctx.episode_dir)))
    calls,cost=usage_summary(ctx.episode_dir);report.update(ai_usage=calls,cost=cost,elapsed_seconds=time.monotonic()-ctx.started)
    validate_report(report)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--episode-dir',type=Path,required=True);args=parser.parse_args(argv);root=args.episode_dir.resolve()
    previous=read_json(root/'qa-report.json');spec=read_json(root/'narrative-spec.json')
    ctx=EpisodeContext(spec['episode_id'],root,spec['goal'],fixture_mode=previous.get('fixture_mode',False))
    timeline=otio.adapters.read_from_file(str(root/'timeline.otio'));track=next(t for t in timeline.tracks if t.name=='V1');cursor_frames=0;cuts=[]
    for clip in track:
        if cursor_frames:cuts.append(round(cursor_frames*1000/30))
        cursor_frames+=output_frames(clip.source_range.duration.to_seconds())
    result=AssemblyResult(ctx.episode_id,root/'finished.mp4',root/'timeline.otio',root/'narrative-spec.json',cuts,round(cursor_frames*1000/30))
    report=validate_candidate(ctx,result,spec,attempts=previous['attempts'],judge_provider=None)
    report['goal_judge']=previous.get('goal_judge',unavailable_judge());report['elapsed_seconds']=previous['elapsed_seconds']
    if report['passed'] and report['goal_judge']['status']=='scored' and report['goal_judge']['score']<60:
        report['result']='partial';report['coverage']['status']='partial';report['coverage']['limitations'].append('AI goal-coverage score below assumed 60/100 threshold')
    validate_report(report);atomic_write_json(root/'qa-report.json',report)
    print(json.dumps(dict(passed=report['passed'],result=report['result'])));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
