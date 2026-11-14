"""Disposable synthetic footage; tones and sidecars are not real speech."""
from pathlib import Path
import argparse,shutil,subprocess,json

def generate_footage(out_dir,*,seed=8090,profile='trip'):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    hevc=profile=='iphone26';count=26 if hevc else 3;duration=1 if hevc else 4;sources=[]
    for index in range(count):
        path=out_dir/f'{index+1:02d}-trip.{"MOV" if hevc else "mp4"}'
        if hevc and index:shutil.copy2(out_dir/'01-trip.MOV',path)
        else:
            cmd=['ffmpeg','-y','-v','error','-f','lavfi','-i',f'testsrc2=size={"3840x2160" if hevc else "320x180"}:rate={2 if hevc else 30}']
            if profile!='no_audio':cmd+=['-f','lavfi','-i','anullsrc=r=48000:cl=mono' if profile=='silent' else f'sine=frequency={440+index*110}:sample_rate=48000']
            cmd+=['-t',str(duration),'-c:v','libx265' if hevc else 'libx264','-preset','ultrafast','-threads','2','-filter_threads','1']
            if hevc:cmd+=['-x265-params','pools=1:frame-threads=1','-tag:v','hvc1']
            cmd+=['-pix_fmt','yuv420p']
            if profile!='no_audio':cmd+=['-c:a','aac']
            subprocess.run(cmd+[str(path)],check=True,capture_output=True,timeout=120)
        sources.append(dict(source_id=f'source-{index+1:04d}',filename=path.name,duration_ms=duration*1000))
    fixture=dict(schema_version='1.0',sources=sources,words=[])
    (out_dir/'transcript-fixture.json').write_text(json.dumps(fixture,indent=2))
    if profile=='mixed':(out_dir/'99-corrupt.mov').write_text('not a video')
    return fixture
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',default='trip');a=p.parse_args();generate_footage(a.out,profile=a.profile)
