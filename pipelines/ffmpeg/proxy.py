"""Low-resolution, orientation-corrected analysis proxies; originals stay untouched."""
from pathlib import Path
import json, subprocess
from .contracts import StageError

def run(cmd, timeout=180):
    try: return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=timeout)
    except (subprocess.SubprocessError,OSError) as e:
        raise StageError('MEDIA_PROCESS_FAILED','refinery','Media process failed') from e

def probe(path):
    data=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]).stdout)
    videos=[s for s in data['streams'] if s['codec_type']=='video']
    if not videos: raise StageError('NO_VIDEO','refinery','Source contains no video')
    v=videos[0]; fmt=data['format']
    duration=float(v.get('duration') or fmt['duration'])
    rotation=float(v.get('tags',{}).get('rotate',0))
    for side in v.get('side_data_list',[]): rotation=float(side.get('rotation',rotation))
    return dict(data=data,video=v,duration_ms=round(duration*1000),rotation=rotation,
                start_ms=round(float(v.get('start_time',fmt.get('start_time',0)))*1000))

def create_proxy(source_path, output_path, *, max_side=640, fps=15):
    source_path=Path(source_path); output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    original=probe(source_path)
    run(['ffmpeg','-y','-v','error','-threads','2','-i',str(source_path),'-map','0:v:0','-map','0:a:0?',
         '-vf',f'scale={max_side}:{max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps={fps}',
         '-r',str(fps),'-c:v','libx264','-preset','ultrafast','-crf','28','-pix_fmt','yuv420p',
         '-threads','2','-filter_threads','1','-c:a','aac','-ar','48000','-metadata:s:v:0','rotate=0',str(output_path)])
    proxy=probe(output_path)
    if abs(proxy['duration_ms']-original['duration_ms'])>150:
        raise StageError('PROXY_TIME_DRIFT','refinery','Proxy duration differs from source')
    return dict(source_duration_ms=original['duration_ms'],proxy_duration_ms=proxy['duration_ms'],
        source_start_time_ms=original['start_ms'],proxy_start_time_ms=proxy['start_ms'],
        rotation_degrees=original['rotation'],proxy_time_offset_ms=0)

def sample_keyframes(proxy_path,out_dir,*,positions=(.15,.5,.85)):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True); info=probe(proxy_path); frames=[]
    for index,position in enumerate(positions):
        ms=min(max(0,round(info['duration_ms']*position)),max(0,info['duration_ms']-100))
        path=out_dir/f'frame-{index}.jpg'
        run(['ffmpeg','-y','-v','error','-ss',str(ms/1000),'-i',str(proxy_path),'-frames:v','1',
             '-vf','scale=640:640:force_original_aspect_ratio=decrease','-threads','1',str(path)])
        if not path.exists():raise StageError('FRAME_MISSING','refinery','Unable to sample clip')
        frames.append(dict(path=str(path),source_time_ms=ms,proxy_time_ms=ms))
    return frames
