"""Source discovery, proxy evidence and explicitly local WhisperX transcription."""
from pathlib import Path
from fractions import Fraction
import contextlib, os, socket
from PIL import Image, ImageStat, ImageFilter, ImageChops
from .contracts import StageError,atomic_write_json
from .proxy import probe,create_proxy,sample_keyframes,run

def discover_sources(footage_dir):
    return sorted((p for p in Path(footage_dir).iterdir() if p.is_file()),key=lambda p:p.name.casefold())

@contextlib.contextmanager
def _offline_models():
    original=socket.socket.connect
    def blocked(*args,**kwargs):raise RuntimeError('Model assets must be provisioned offline')
    socket.socket.connect=blocked
    try:yield
    finally:socket.socket.connect=original

def transcribe_words(source,*,model_dir,device='cpu'):
    root=Path(model_dir)
    if not (root/'transcription').is_dir() or not (root/'alignment').is_dir():
        raise StageError('DEPENDENCY_MISSING','refinery','Local WhisperX transcription/alignment assets are required',source['source_id'])
    try:
        import whisperx
        with _offline_models():
            model=whisperx.load_model(str(root/'transcription'),device,compute_type='int8')
            audio=whisperx.load_audio(source['path']); result=model.transcribe(audio,batch_size=4)
            if not result['segments']:return []
            align,metadata=whisperx.load_align_model(language_code=result['language'],device=device,model_name=str(root/'alignment'))
            aligned=whisperx.align(result['segments'],align,metadata,audio,device)
        return [dict(source_id=source['source_id'],text=w['word'],start_ms=round(w['start']*1000),end_ms=round(w['end']*1000),confidence=w.get('score')) for w in aligned['word_segments'] if 'start' in w and 'end' in w]
    except StageError:raise
    except Exception as e:raise StageError('ANALYSIS_FAILED','refinery','Local WhisperX analysis failed; check cached assets',source['source_id']) from e

def run_refinery(ctx,footage_dir,*,transcriber=None):
    if not ctx.goal.strip():raise StageError('GOAL_EMPTY','refinery','Goal must not be blank')
    paths=discover_sources(footage_dir);media=[p for p in paths if p.suffix.lower() in ('.mp4','.mov')]
    if not media:raise StageError('INPUT_EMPTY','refinery','No supported source files')
    if len(media)>40:raise StageError('INPUT_LIMIT','refinery','At most 40 media files are supported')
    sources=[];excluded=[];total=0
    for index,path in enumerate(paths,1):
        sid=f'source-{index:04d}'
        try:
            if path.suffix.lower() not in ('.mp4','.mov'):raise ValueError('Unsupported input')
            info=probe(path);v=info['video'];fps=Fraction(v.get('avg_frame_rate') or v.get('r_frame_rate','30/1'))
            if fps<=0:fps=Fraction(30)
            total+=info['duration_ms']
            sources.append(dict(source_id=sid,path=str(path.resolve()),duration_ms=info['duration_ms'],fps_num=fps.numerator,fps_den=fps.denominator,
                width=v['width'],height=v['height'],has_audio=any(s['codec_type']=='audio' for s in info['data']['streams']),
                creation_time=info['data']['format'].get('tags',{}).get('creation_time'),proxy_path=None,proxy_duration_ms=None,
                proxy_time_offset_ms=0,rotation_degrees=info['rotation'],keyframes=[]))
        except (StageError,ValueError,KeyError) as e:excluded.append(dict(source_id=sid,path=str(path.resolve()),reason=str(e)))
    if total>1800000:raise StageError('INPUT_LIMIT','refinery','At most 30 minutes of source video are supported')
    if not sources:raise StageError('NO_USABLE_VIDEO','refinery','No usable video remains')
    scenes=[];words=[];maps={}
    for source in sources:
        sid=source['source_id']; proxy=ctx.episode_dir/'proxies'/f'{sid}.mp4'
        try:maps[sid]=create_proxy(source['path'],proxy)
        except StageError as e:raise StageError('PROXY_FAILED','refinery',str(e),sid) from e
        source.update(proxy_path=str(proxy.relative_to(ctx.episode_dir)),proxy_duration_ms=maps[sid]['proxy_duration_ms'])
        frames=sample_keyframes(proxy,ctx.episode_dir/'frames'/sid)
        for f in frames:f['path']=str(Path(f['path']).relative_to(ctx.episode_dir));f['source_time_ms']=min(f['source_time_ms'],source['duration_ms']-1)
        source['keyframes']=frames
        with Image.open(ctx.episode_dir/frames[0]['path']) as first, Image.open(ctx.episode_dir/frames[-1]['path']) as last:
            motion=ImageStat.Stat(ImageChops.difference(first.convert('L'),last.convert('L').resize(first.size))).mean[0]/255
        if source['has_audio']:
            if transcriber is not None:ws=transcriber.words(source) if hasattr(transcriber,'words') else transcriber(source)
            else:ws=transcribe_words(source,model_dir=os.environ.get('FRAMEFOUNDRY_MODEL_DIR','models'))
        else:ws=[]
        for w in ws:
            if w['source_id']!=sid or not 0<=w['start_ms']<w['end_ms']<=source['duration_ms']:raise StageError('WORD_RANGE_INVALID','refinery','Invalid transcript word range',sid)
        words.extend(ws)
        # Scene boundaries are detected on the proxy and mapped to source time.
        proc=run(['ffmpeg','-hide_banner','-i',str(proxy),'-vf',"select='gt(scene,0.30)',showinfo",'-an','-f','null','-'])
        import re
        cuts=[0]+[round(float(t)*1000) for t in re.findall(r'pts_time:([0-9.]+)',proc.stderr)]+[source['duration_ms']]
        cuts=sorted(set(t for t in cuts if 0<=t<=source['duration_ms']))
        for j,(a,b) in enumerate(zip(cuts,cuts[1:])):
            if a>=b:continue
            frame=min(frames,key=lambda f:abs(f['source_time_ms']-(a+b)/2))
            with Image.open(ctx.episode_dir/frame['path']) as image:
                gray=image.convert('L');brightness=ImageStat.Stat(gray).mean[0]/255
                sharpness=ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
            tokens=[w['text'].lower() for w in ws if w['start_ms']<b and w['end_ms']>a]
            scenes.append(dict(scene_id=f'{sid}-scene-{j}',source_id=sid,start_ms=a,end_ms=b,thumbnail_path=frame['path'],brightness=brightness,sharpness=sharpness,motion=motion,tokens=tokens))
    analysis=dict(schema_version='1.0',episode_id=ctx.episode_id,goal=ctx.goal,sources=sources,scenes=scenes,words=words,exclusions=excluded)
    from .candidates import extract_candidates
    analysis['candidates']=extract_candidates(analysis,ctx.episode_dir)
    atomic_write_json(ctx.episode_dir/'transcript.json',dict(schema_version='1.0',episode_id=ctx.episode_id,words=words))
    atomic_write_json(ctx.episode_dir/'analysis.json',analysis);atomic_write_json(ctx.episode_dir/'proxy-map.json',maps)
    return analysis
