"""Goal-derived cards, source-bound associations and zero-token render checks."""
from __future__ import annotations
import copy, hashlib, json, math, os, re, subprocess, tempfile, textwrap
from pathlib import Path
from PIL import ImageFont
from .contracts import StageError

STYLE=Path(__file__).resolve().parents[2]/'contracts/overlay-style.json'
CUES={'halloween':{'halloween','costume','costumes','parade','shop','shops','candy','uptown','trick'},
      'resort':{'pool','resort','hotel','hilton','bell rock','red rocks'}}
VISUAL=('costumes','parade','shops','pool','resort','red_rocks')
CHECKS=('overlay_safe_area','overlay_duration','overlay_text_limits','overlay_cut_clearance','overlay_render_binding')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(text):
    text=str(text)
    if any(ord(c)<32 and not c.isspace() for c in text):raise StageError('OVERLAY_TEXT','planner','Control characters are not supported')
    return re.sub(r'\s+',' ',text).strip()
def requested(ctx):
    goal=ctx.goal; sections=list(getattr(ctx,'overlay_sections',[]) or [])
    if not sections and ';' in goal:
        pieces=goal.split(';');goal=pieces[0];sections=[clean(x) for x in pieces[1:] if clean(x)]
    if len(sections)>8:raise StageError('OVERLAY_SECTIONS','planner','At most eight sections are supported')
    return clean(goal),[clean(x) for x in sections]
def active(ctx):return bool(getattr(ctx,'overlay_mode',None) if getattr(ctx,'overlay_mode',None) is not None else requested(ctx)[1])
def cue_set(section):
    text=section.lower()
    if any(x in text for x in ('halloween','uptown')):return CUES['halloween']
    if any(x in text for x in ('hilton','bell rock','resort')):return CUES['resort']
    return set(re.findall(r'\w+',text))-{'the','a','in','at','with','and','to','of'}

def attach(ctx,spec,data=None,rows=()):
    if not active(ctx):return spec
    title,sections=requested(ctx);proofs=[]
    for s in (data or {}).get('segments',[]):
        i=s.get('overlay_section');cue=s.get('overlay_evidence')
        if type(i) is not int or not 0<=i<len(sections) or cue not in VISUAL:continue
        if cue.replace('_',' ') not in cue_set(sections[i]):continue
        for row in rows:
            t=row.get('frame_time_ms')
            if (row.get('frame_in_window') and 'tile_index' in row and type(t) is int
                and row['source_id']==s['source_id'] and s['raw_start_ms']<=t<s['raw_end_ms']
                and row.get('frame_path') and row.get('frame_sha256')):
                proofs.append(dict(section=i,source_id=s['source_id'],start_ms=s['raw_start_ms'],end_ms=s['raw_end_ms'],
                    cue=cue,frame_time_ms=t,frame_path=row['frame_path'],frame_sha256=row['frame_sha256']))
                break
    spec['overlays']=dict(version='goal-overlays-1',title=title,sections=sections,proofs=proofs,
                         cards=[],limitations=[],render_sha256=None,plan_sha256=None,font_sha256=None,style_sha256=None)
    return spec

def font_path():
    paths=[os.environ.get('FRAMEFOUNDRY_OVERLAY_FONT',''),'/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf','/Library/Fonts/Arial Bold.ttf']
    for path in paths:
        if path and Path(path).is_file():return Path(path)
    raise StageError('OVERLAY_FONT','assembler','No bold sans font; set FRAMEFOUNDRY_OVERLAY_FONT')

def layout(text,width,height,position='top'):
    lines=textwrap.wrap(clean(text),width=42,break_long_words=True,break_on_hyphens=False) or ['']
    if len(lines)>2:lines=lines[:2];lines[-1]=lines[-1][:41]+'…'
    text='\n'.join(lines);size=max(24,round(height*.034));path=font_path()
    while size>=18:
        font=ImageFont.truetype(str(path),size)
        missing=bytes(font.getmask('\uffff'))
        if any(not c.isspace() and bytes(font.getmask(c))==missing for c in text):
            raise StageError('OVERLAY_GLYPH','assembler','Configured font cannot render the overlay text')
        tw=math.ceil(max(font.getlength(line) for line in lines))+12
        th=math.ceil(size*1.35*len(lines))+12
        if tw<=width*.88:break
        size-=1
    x=(width-tw)//2;y=round(height*.18) if position=='top' else round(height*.75)-th-8
    return dict(text=text,x=x,y=y,width=tw,height=th,fontsize=size,position=position)

def intersects(a,b):return a[0]<b[0]+b[2] and b[0]<a[0]+a[2] and a[1]<b[1]+b[3] and b[1]<a[1]+a[3]
def choose_position(text,w,h,boxes):
    top=layout(text,w,h,'top');low=layout(text,w,h,'lower')
    count=lambda c:sum(intersects([c['x'],c['y'],c['width'],c['height']+8],b) for b in boxes)
    return low if count(low)<count(top) else top

def face_boxes(source,source_time,w,h):
    """Same local Haar detector, one crop-matched frame; not full face tracking."""
    try:
        import cv2, numpy as np
        filt=f'scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}'
        p=subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(source_time/1000),'-i',source,
            '-vf',filt,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],capture_output=True,timeout=30)
        if p.returncode or len(p.stdout)!=w*h*3:return None
        pixels=np.frombuffer(p.stdout,dtype=np.uint8).reshape(h,w,3)
        gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
        detector=cv2.CascadeClassifier(str(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'))
        return [list(map(int,b)) for b in detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(24,24))]
    except Exception:return None  # Optional CPU cues cannot defeat safe text rendering

def matches(spec,analysis,clip,section,root):
    plan=spec['overlays'];sid=clip.metadata['source_id'];a=clip.source_range.start_time.to_seconds()*1000
    b=a+clip.source_range.duration.to_seconds()*1000
    for p in plan['proofs']:
        if p['section']==section and p['source_id']==sid and a<=p['frame_time_ms']<b:
            path=(Path(root)/p['frame_path']).resolve()
            if path.is_relative_to(Path(root).resolve()) and path.is_file() and filehash(path)==p['frame_sha256']:
                return dict(kind='planner_frame',source_id=sid,start_ms=p['start_ms'],end_ms=p['end_ms'],cue=p['cue'])
    words=[word for word in analysis.get('words',[]) if word['source_id']==sid and a<=word['start_ms'] and word['end_ms']<=b]
    text=' '.join(word['text'] for word in words).lower()
    # A negated/requested future visit is not evidence of the retained scene.
    if re.search(r'\b(no|not|never|wish|maybe|would|could|should)\b',text):return None
    for cue in sorted(cue_set(plan['sections'][section])):
        if re.search(r'(?<!\w)'+re.escape(cue)+r'(?!\w)',text):
            return dict(kind='retained_transcript',source_id=sid,start_ms=round(a),end_ms=round(b),cue=cue)
    return None

def compile_cards(spec,analysis,timeline,root):
    from .render import output_frames
    if not spec.get('overlays'):return spec
    plan=copy.deepcopy(spec['overlays']);plan.update(cards=[],limitations=[],render_sha256=None,plan_sha256=None,font_sha256=None,style_sha256=None)
    spec=copy.deepcopy(spec);spec['overlays']=plan
    clips=list(timeline.tracks[0]);w,h=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
    starts=[];cursor=0
    for c in clips:starts.append((cursor,output_frames(c.source_range.duration.to_seconds())));cursor+=starts[-1][1]
    occupied={}
    def add(kind,text,index,hold,proof):
        start,length=starts[index];first=max(start+(1 if index else 0),occupied.get(index,start))
        end=min(start+length-(1 if index<len(clips)-1 else 0),first+round((hold+.35)*30))
        if end-first<47:return False  # ceil((1.2+.2+.15)*30)
        source=next(s for s in spec['sources'] if s['source_id']==clips[index].metadata['source_id'])
        boxes=face_boxes(source['path'],round(clips[index].source_range.start_time.to_seconds()*1000+(first-start)*1000/30),w,h)
        card=choose_position(text,w,h,boxes or [])
        card.update(kind=kind,shot_id=f'shot-{index+1:02d}',start_frame=first,end_frame=end,evidence=proof)
        plan['cards'].append(card);occupied[index]=end+1
        if boxes is None:plan['limitations'].append('Face geometry unavailable; conservative top placement, no verified face avoidance')
        else:plan['limitations'].append('Face placement uses one crop-matched CPU sample, not full interval tracking')
        return True
    if not add('title',plan['title'],0,1.5,dict(kind='operator_goal',source_id='',start_ms=0,end_ms=0,cue='')):
        plan['limitations'].append('Title omitted: first shot lacks a cut-safe readable interval')
    for section,name in enumerate(plan['sections']):
        found=False
        for i,clip in enumerate(clips):
            proof=matches(spec,analysis,clip,section,root)
            if proof and add('section',f'{section+1}. {name}',i,1.8,proof):found=True;break
        if not found:plan['limitations'].append(f'Requested section {section+1} omitted: no grounded cut-safe readable interval')
    for note in plan['limitations']:
        if 'omitted:' in note and note not in spec['coverage']['limitations']:
            spec['coverage']['status']='partial';spec['coverage']['limitations'].append(note)
    plan['style_sha256']=filehash(STYLE);plan['font_sha256']=filehash(font_path())
    plan['plan_sha256']=digest({k:v for k,v in plan.items() if k not in ('render_sha256','plan_sha256')})
    return spec

def burn(video,output,plan):
    from .render import _run,_ffmpeg
    font=font_path();filters=[]
    with tempfile.TemporaryDirectory(prefix='overlay-',dir=Path(output).parent) as scratch:
        for i,c in enumerate(plan['cards']):
            text=Path(scratch)/f'text-{i}.txt';text.write_text(c['text'])
            s=c['start_frame']/30;e=c['end_frame']/30
            # User text is never filter syntax, %{expansion}, ASS tags or shell input.
            escape=lambda p:str(p).replace('\\','\\\\').replace(':','\\:').replace("'", "\\'")
            alpha=f'if(lt(t,{s}),0,if(lt(t,{s+.2}),(t-{s})/.2,if(lt(t,{e-.15}),1,if(lt(t,{e}),({e}-t)/.15,0))))'
            y=f'{c["y"]}+8*(1-min(1,max(0,(t-{s})/.2)))'
            filters.append(f"drawtext=fontfile='{escape(font)}':textfile='{escape(text)}':expansion=none:fontsize={c['fontsize']}:fontcolor=white:shadowcolor=black@0.65:shadowx=2:shadowy=2:line_spacing=4:x={c['x']}:y='{y}':alpha='{alpha}'")
        if filters:
            _run(_ffmpeg()+['-i',str(video),'-vf',','.join(filters),'-map','0:v:0','-map','0:a:0','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2','-c:a','copy',str(output)])
        else:
            import shutil;shutil.copyfile(video,output)
    plan['render_sha256']=filehash(output)

def verify(spec,video,cuts,duration_ms):
    plan=spec.get('overlays')
    if not plan:return []
    w,h=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
    cards=plan['cards'];results={name:True for name in CHECKS}
    for c in cards:
        results['overlay_safe_area'] &= (0<=c['x'] and c['x']+c['width']<=w and c['y']>=math.ceil(h*.12) and c['y']+c['height']+8<=math.floor(h*.8))
        results['overlay_duration'] &= (c['end_frame']-c['start_frame'])/30-.35>=1.2 and 0<=c['start_frame']<c['end_frame']<=round(duration_ms*30/1000)+1
        results['overlay_text_limits'] &= 1<=len(c['text'].splitlines())<=2 and all(len(x)<=42 for x in c['text'].splitlines())
        results['overlay_cut_clearance'] &= all(not c['start_frame']/30*1000<=cut<c['end_frame']/30*1000 for cut in cuts)
        expected=layout(c['text'],w,h,c['position'])
        results['overlay_render_binding'] &= all(c[k]==expected[k] for k in ('text','x','y','width','height','fontsize','position'))
    results['overlay_render_binding'] &= (plan['render_sha256']==filehash(video) and plan['style_sha256']==filehash(STYLE)
        and plan['font_sha256']==filehash(font_path()) and plan['plan_sha256']==digest({k:v for k,v in plan.items() if k not in ('render_sha256','plan_sha256')}))
    return [dict(name=name,status='pass' if ok else 'fail',reason=None if ok else 'Overlay geometry, timing or render binding failed',required=True) for name,ok in results.items()]
