"""Independent hard media gates. Advisory AI cannot make a failed gate pass."""
from __future__ import annotations
# Support the repository guardrail's direct-script entrypoint as well as -m.
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='pipelines.ffmpeg'
from pathlib import Path
import argparse,copy,json,math,re,time
import opentimelineio as otio
from .contracts import *
from .proxy import probe,run
from .contact_sheet import build_contact_sheet
from .assembler import pad_word_interval
from .goal_judge import judge_goal

def measure_audio(path):
    result=run(['ffmpeg','-hide_banner','-i',str(path),'-af','loudnorm=I=-14:TP=-1:LRA=11:print_format=json','-f','null','-'])
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
    def check(name,passed,reason=None):checks.append(dict(name=name,status='pass' if passed else 'fail',reason=reason if not passed else None,required=True))
    try:validate_narrative(spec);check('narrative_schema',True)
    except (StageError,ValueError):check('narrative_schema',False,'Narrative contract invalid')
    try:
        info=probe(result.candidate_path);data=info['data'];v=info['video'];a=next((s for s in data['streams'] if s['codec_type']=='audio'),{})
        report['actual_duration_ms']=info['duration_ms'];check('readable_media',True)
        size=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
        check('output_encoding',v['codec_name']=='h264' and a.get('codec_name')=='aac')
        check('output_dimensions',(v['width'],v['height'])==size)
        from fractions import Fraction
        check('output_fps',Fraction(v.get('avg_frame_rate','0/1'))==30 and Fraction(v.get('r_frame_rate','0/1'))==30)
    except Exception:check('readable_media',False,'Output cannot be probed');info=None
    try:
        metrics=measure_audio(result.candidate_path);report.update(metrics)
        check('loudness',metrics['integrated_lufs'] is not None and -15<=metrics['integrated_lufs']<=-13,'LUFS must be finite and within [-15,-13]')
        check('true_peak',metrics['true_peak_dbtp'] is not None and metrics['true_peak_dbtp']<=-1,'True peak must be <= -1 dBTP')
    except Exception:check('loudness',False,'Audio measurement unavailable');check('true_peak',False,'Peak measurement unavailable')
    try:
        timeline=otio.adapters.read_from_file(str(result.timeline_path));video=next(t for t in timeline.tracks if t.name=='V1')
        expected_ids=[s['segment_id'] for s in spec['segments']];seen=[];cursor=0;cuts=[];sources={s['source_id']:s for s in spec['sources']}
        words=read_json(ctx.episode_dir/'transcript.json')['words'] if (ctx.episode_dir/'transcript.json').exists() else []
        expected_segments={s['segment_id']:s for s in spec['segments']}
        for clip in video:
            if cursor:cuts.append(round(cursor))
            sid=clip.metadata['source_id'];source=sources[sid]
            if Path(clip.media_reference.target_url).resolve()!=Path(source['path']).resolve() or not Path(source['path']).is_file():raise ValueError('Source mismatch')
            start=clip.source_range.start_time.to_seconds()*1000;end=start+clip.source_range.duration.to_seconds()*1000
            if not 0<=start<end<=source['duration_ms']+1:raise ValueError('Range')
            selected_ids=list(clip.metadata['selection_ids']);seen.extend(selected_ids)
            for segment_id in selected_ids:
                segment=expected_segments[segment_id]
                if segment['source_id']!=sid or not start<=segment['render_start_ms']<segment['render_end_ms']<=end+1:raise ValueError('Selection mapping')
                chosen=[w for w in words if w['source_id']==sid and w['start_ms']<=segment['raw_end_ms'] and segment['raw_start_ms']<=w['end_ms']]
                if chosen:
                    x,y=pad_word_interval(min(w['start_ms'] for w in chosen),max(w['end_ms'] for w in chosen),source['duration_ms'],[w for w in words if w['source_id']==sid])
                    if start>x+1 or end<y-1:raise ValueError('Unprotected spoken cut')
                    if min(w['start_ms'] for w in chosen)<80 or max(w['end_ms'] for w in chosen)+120>source['duration_ms']:
                        report['limitations'].append('Full word padding unavailable at source boundary')
            cursor+=end-start
        check('timeline_and_padding',seen==expected_ids and cuts==result.cut_times_ms,'Timeline ranges/order/padding mismatch')
        check('render_duration',info is not None and abs(info['duration_ms']-cursor)<=150,'Render disagrees with actual saved timeline')
        check('goal_duration',info is not None and (abs(info['duration_ms']-spec['target_duration_ms'])<=1000 or (spec['coverage']['status']=='partial' and bool(spec['coverage']['limitations']))),'Unexplained duration deviation')
    except Exception:check('timeline_and_padding',False,'Invalid timeline or source bounds/padding');check('render_duration',False,'Timeline unavailable')
    try:
        build_contact_sheet(result.candidate_path,result.cut_times_ms,ctx.episode_dir/'contact-sheet.jpg')
        report['contact_sheet_path']='contact-sheet.jpg';check('contact_sheet',True)
    except Exception:check('contact_sheet',False,'Cut-boundary evidence unavailable')
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
    timeline=otio.adapters.read_from_file(str(root/'timeline.otio'));track=next(t for t in timeline.tracks if t.name=='V1');cursor=0;cuts=[]
    for clip in track:
        if cursor:cuts.append(round(cursor))
        cursor+=clip.source_range.duration.to_seconds()*1000
    result=AssemblyResult(ctx.episode_id,root/'finished.mp4',root/'timeline.otio',root/'narrative-spec.json',cuts,round(cursor))
    report=validate_candidate(ctx,result,spec,attempts=previous['attempts'],judge_provider=None)
    report['goal_judge']=previous.get('goal_judge',unavailable_judge());report['elapsed_seconds']=previous['elapsed_seconds']
    if report['passed'] and report['goal_judge']['status']=='scored' and report['goal_judge']['score']<60:
        report['result']='partial';report['coverage']['status']='partial';report['coverage']['limitations'].append('AI goal-coverage score below assumed 60/100 threshold')
    validate_report(report);atomic_write_json(root/'qa-report.json',report)
    print(json.dumps(dict(passed=report['passed'],result=report['result'])));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
