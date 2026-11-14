"""Goal-derived cards, source-bound associations and zero-token render checks."""
from __future__ import annotations
from .media_tools import media_binary
import copy, hashlib, json, math, os, re, subprocess, tempfile, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .contracts import StageError

STYLE=Path(__file__).resolve().parents[2]/'contracts/overlay-style.json'
CUES={'halloween':{'halloween','costume','costumes','parade','shop','shops','candy','uptown','trick'},
      'resort':{'pool','resort','hotel','hilton','bell rock','red rocks'}}
VISUAL=('costumes','parade','shops','pool','resort','red_rocks')
CHECKS=('overlay_safe_area','overlay_duration','overlay_text_limits','overlay_cut_clearance','overlay_render_binding','overlay_legibility')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(text):
    text=str(text)
    if any(ord(c)<32 and not c.isspace() for c in text):raise StageError('OVERLAY_TEXT','planner','Control characters are not supported')
    return re.sub(r'\s+',' ',text).strip()
def requested(ctx):
    goal=ctx.goal; sections=list(getattr(ctx,'overlay_sections',[]) or [])
    listed=list(re.finditer(r'(?m)^\s*(?:[0-9]+[.)]|[-*])\s+(.+)$',goal))
    if not sections and listed:
        sections=[m.group(1) for m in listed];goal=goal[:listed[0].start()]
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
    spec['overlays']=dict(version='goal-overlays-2',title=title,sections=sections,proofs=proofs,
                         cards=[],limitations=[],render_sha256=None,plan_sha256=None,font_sha256=None,style_sha256=None)
    return spec

def font_path():
    paths=[os.environ.get('FRAMEFOUNDRY_OVERLAY_FONT',''),'/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf','/Library/Fonts/Arial Bold.ttf']
    for path in paths:
        if path and Path(path).is_file():return Path(path)
    raise StageError('OVERLAY_FONT','assembler','No bold sans font; set FRAMEFOUNDRY_OVERLAY_FONT')

def layout(text,width,height,position='top',production=False):
    # Never shrink cap height to fit width: use a bold condensed raster instead.
    text=clean(text);text=text if len(text)<=42 else text[:41]+'…';path=font_path();size=max(24,round(height*.06))
    while ImageFont.truetype(str(path),size).getbbox('H')[3]-ImageFont.truetype(str(path),size).getbbox('H')[1]<math.ceil(height*.06):size+=1
    font=ImageFont.truetype(str(path),size)
    words=text.split();options=[]
    if len(text)<=42:options.append([text])
    for i in range(1,len(words)):
        lines=[' '.join(words[:i]),' '.join(words[i:])]
        if all(len(line)<=42 for line in lines):options.append(lines)
    if not options:
        text=text[:82]+'…';options=[[text[:41],text[41:83]]]
    fitting=[lines for lines in options if sum(font.getbbox(line)[3]-font.getbbox(line)[1] for line in lines)+20*(len(lines)-1)+64+8<=math.floor(height/3)-math.ceil(height*.12)]
    lines=min(fitting or options,key=lambda lines:max(font.getlength(line) for line in lines))
    missing=bytes(font.getmask('\uffff'))
    if any(not c.isspace() and bytes(font.getmask(c))==missing for c in '\n'.join(lines)):
        raise StageError('OVERLAY_GLYPH','assembler','Configured font cannot render the overlay text')
    natural=math.ceil(max(font.getlength(line) for line in lines))
    padding=32;tw=min(natural,math.floor(width*(.80 if production and height>width else .80 if production else .88))-2*padding)
    if tw/natural<.4:
        if len(text)<4:raise StageError('OVERLAY_FIT','assembler','Text cannot fit a legible card')
        return layout(text[:-2]+'…',width,height,production=production)
    heights=[font.getbbox(line)[3]-font.getbbox(line)[1] for line in lines]
    th=sum(heights)+20*(len(lines)-1)+2*padding
    if th+8>math.floor(height/3)-math.ceil(height*.12):raise StageError('OVERLAY_FIT','assembler','Large text does not fit the upper-third safe area')
    return dict(text='\n'.join(lines),x=math.floor((width*(.90 if production and height>width else 1)-tw-2*padding)/2),y=math.ceil(height*.12),
        width=tw+2*padding,height=th,fontsize=size,position='top',
        cap_height=font.getbbox('H')[3]-font.getbbox('H')[1],pill_opacity=.6,padding=padding)

def intersects(a,b):return a[0]<b[0]+b[2] and b[0]<a[0]+a[2] and a[1]<b[1]+b[3] and b[1]<a[1]+a[3]
def choose_position(text,w,h,boxes):
    # Current operator requirement fixes placement to the safe upper third.
    return layout(text,w,h,'top')

def face_boxes(source,source_time,w,h):
    """Same local Haar detector, one crop-matched frame; not full face tracking."""
    try:
        import cv2, numpy as np
        filt=(f'scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}' if h>w
              else f'scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2')
        p=subprocess.run([media_binary('ffmpeg'),'-nostdin','-v','error','-ss',str(source_time/1000),'-i',source,
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

def readable_frames(first,end,hidden):
    """Full-strength hold on the 30fps grid; boundary masks in fades are not lost twice."""
    # Conservatively round each 250ms fade up to eight frames.
    lo,hi=first+8,end-8
    return max(0,hi-lo)-len({f for f in hidden if lo<=f<hi})


def free_intervals(first,end,occupied):
    """Earliest unoccupied pieces, without jumping past a later reserved card."""
    cursor=first
    for lo,hi in sorted(occupied):
        if hi<=cursor:continue
        if lo>=end:break
        if cursor<lo:yield cursor,min(lo,end)
        cursor=max(cursor,hi+1)
        if cursor>=end:return
    if cursor<end:yield cursor,end


def compile_cards(spec,analysis,timeline,root):
    from .render import output_frames
    if not spec.get('overlays'):return spec
    plan=copy.deepcopy(spec['overlays']);plan.update(cards=[],limitations=[],render_sha256=None,plan_sha256=None,font_sha256=None,style_sha256=None)
    old_omissions={n for n in spec['overlays'].get('limitations',[]) if n.startswith('Requested section ') and ' omitted:' in n}
    spec=copy.deepcopy(spec);spec['overlays']=plan
    spec['coverage']['limitations']=[n for n in spec['coverage']['limitations'] if n not in old_omissions]
    production=spec.get('editing_policy',{}).get('mode')=='production-v1'
    if production:plan['production']=True
    clips=list(timeline.tracks[0]);w,h=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
    starts=[];cursor=0
    for c in clips:starts.append((cursor,output_frames(c.source_range.duration.to_seconds())));cursor+=starts[-1][1]
    cuts=[start for start,length in starts[1:]]
    occupied=[]
    def clear_frames(first,end):
        return sorted({frame for cut in cuts for frame in (cut-1,cut,cut+1) if first<=frame<end})
    def add(kind,text,index,proof,section=None):
        start,length=starts[index];first=start+(1 if index else 0)
        end_limit=cursor
        if section is not None:
            # Labels can bridge only a contiguous run of independently grounded clips.
            end_limit=start+length
            for following in range(index+1,len(clips)):
                if not matches(spec,analysis,clips[following],section,root):break
                end_limit=starts[following][0]+starts[following][1]
        for first,available_end in free_intervals(first,end_limit,occupied):
            target=105 if kind=='title' else 91
            end=min(available_end,first+target)
            while end<available_end and readable_frames(first,end,clear_frames(first,end))<75:
                if kind=='title' and end-first>=105:break
                end+=1
            hidden=clear_frames(first,end)
            if readable_frames(first,end,hidden)>=75:break
        else:return False
        card=layout(text,w,h,production=production)
        if production:
            source=next(s for s in analysis['sources'] if s['source_id']==clips[index].metadata['source_id'])
            at=(clips[index].source_range.start_time.to_seconds()+clips[index].source_range.duration.to_seconds()/2)*1000
            boxes=face_boxes(source['path'],at,w,h)
            collision=bool(boxes and any(intersects((card['x'],card['y'],card['width'],card['height']),box) for box in boxes))
            spec['editing_policy']['visual'].append(dict(shot_id=f'shot-{index+1:02d}',status='sampled' if boxes is not None else 'unavailable',boxes=boxes or [],collision=collision))
            if boxes is None:spec['editing_policy']['limitations'].append('Subject geometry unavailable; no verified face/eye tracking claim')
        if clean(card['text'])!=clean(text):plan['limitations'].append('Overlay text shortened to keep the required large two-line cap height')
        card.update(kind=kind,shot_id=f'shot-{index+1:02d}',start_frame=first,end_frame=end,evidence=proof,cut_frames=hidden)
        plan['cards'].append(card);occupied.append((first,end))
        plan['limitations'].append('Large upper-third scrim placement; cut frames hidden; no verified face tracking')
        return True
    if not add('title',plan['title'],0,dict(kind='operator_goal',source_id='',start_ms=0,end_ms=0,cue='')):
        raise StageError('OVERLAY_DURATION','assembler','Title lacks a 2.5s readable interval within 3.5s')
    for section,name in enumerate(plan['sections']):
        found=False;grounded=False
        for i,clip in enumerate(clips):
            proof=matches(spec,analysis,clip,section,root)
            grounded |= bool(proof)
            if proof and add('section',f'{section+1}. {name}',i,proof,section):found=True;break
        if not found:
            reason='grounded footage found, but no free 2.5s full-strength hold after fades/cut masks' if grounded else 'no retained transcript or verified in-window frame matched'
            plan['limitations'].append(f'Requested section {section+1} omitted: no grounded cut-safe readable interval ({reason})')
    for note in plan['limitations']:
        if 'omitted:' in note and note not in spec['coverage']['limitations']:
            spec['coverage']['status']='partial';spec['coverage']['limitations'].append(note)
    plan['style_sha256']=filehash(STYLE);plan['font_sha256']=filehash(font_path())
    plan['plan_sha256']=digest({k:v for k,v in plan.items() if k not in ('render_sha256','plan_sha256')})
    return spec

def preflight():
    """Portable PNG renderer: no drawtext, libass, or subtitle dependency."""
    from .render import _run, _ffmpeg
    result=_run(_ffmpeg()+['-filters'])
    available=set(re.findall(r'^\s*[A-Z.|]{2,3}\s+(\w+)\s',result.stdout+'\n'+result.stderr,re.M))
    missing={'overlay','fade'}-available
    if missing:
        raise StageError('OVERLAY_FILTERS','assembler','PNG text compositing requires FFmpeg filters: '+', '.join(sorted(missing)))
    return 'png'


def card_png(card,path):
    """Large bold two-line raster, 60% black pill and literal UTF-8 glyphs."""
    image=Image.new('RGBA',(card['width'],card['height']),(0,0,0,0))
    draw=ImageDraw.Draw(image);font=ImageFont.truetype(str(font_path()),card['fontsize'])
    draw.rounded_rectangle((0,0,card['width']-1,card['height']-1),radius=32,fill=(0,0,0,round(255*card['pill_opacity'])))
    lines=card['text'].splitlines();heights=[font.getbbox(line)[3]-font.getbbox(line)[1] for line in lines]
    natural=math.ceil(max(font.getlength(line) for line in lines));target=card['width']-2*card['padding']
    tile=Image.new('RGBA',(natural+8,sum(heights)+20*(len(lines)-1)+8),(0,0,0,0));glyphs=ImageDraw.Draw(tile);y=0
    for line,h in zip(lines,heights):
        x=(natural-font.getlength(line))/2
        glyphs.text((x+2,y+3),line,font=font,fill=(0,0,0,166),anchor='lt')
        glyphs.text((x,y),line,font=font,fill=(255,255,255,255),anchor='lt');y+=h+20
    # Horizontal condensation preserves measured cap height and bold vertical strokes.
    tile=tile.resize((target,tile.height),Image.Resampling.LANCZOS)
    image.alpha_composite(tile,(card['padding'],card['padding']))
    image.save(path,format='PNG')

def burn(video,output,plan):
    from .render import _run,_ffmpeg
    if not plan['cards']:
        import shutil;shutil.copyfile(video,output)
        plan['render_sha256']=filehash(output)
        return
    preflight()
    with tempfile.TemporaryDirectory(prefix='overlay-',dir=Path(output).parent) as scratch:
        args=_ffmpeg()+['-i',str(video)]
        filters=['[0:v:0]setpts=PTS-STARTPTS[base]'];last='base'
        for i,c in enumerate(plan['cards']):
            png=Path(scratch)/f'card-{i}.png';card_png(c,png)
            args+=['-loop','1','-framerate','30','-i',str(png)]
            s=c['start_frame']/30;e=c['end_frame']/30
            # The inclusive between() upper bound is the LAST visible frame,
            # never the exclusive end/cut frame. Tolerance is far below a frame.
            a=s-1e-6;b=(c['end_frame']-1)/30+1e-6
            filters.append(f'[{i+1}:v:0]format=rgba,setpts=PTS-STARTPTS,fade=t=in:st={s}:d=0.25:alpha=1,fade=t=out:st={e-.25}:d=0.25:alpha=1[card{i}]')
            y=f'{c["y"]}+8*(1-min(1,max(0,(t-{s})/.25)))'
            if plan.get('production'):
                #14/120/1 damped spring normalized at250ms; steady hold stays fixed.
                q=f'min(.25,max(0,t-{s}))'
                spring=f'(1-exp(-7*({q}))*(cos(8.426149773*({q}))+.830747161*sin(8.426149773*({q}))))/.964636'
                y=f'{c["y"]}+8*(1-min(1,max(0,{spring})))'
            enabled=f'between(t,{a},{b})'+''.join(f'*not(between(t,{(n-.25)/30},{(n+.25)/30}))' for n in c.get('cut_frames',[]))
            filters.append(f"[{last}][card{i}]overlay=x={c['x']}:y='{y}':enable='{enabled}':shortest=1:eof_action=pass[v{i}]")
            last=f'v{i}'
        _run(args+['-filter_complex',';'.join(filters),'-map',f'[{last}]','-map','0:a:0',
            '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',
            '-c:a','copy','-shortest',str(output)])
    plan['render_sha256']=filehash(output)

def verify(spec,video,cuts,duration_ms):
    plan=spec.get('overlays')
    if not plan:return []
    w,h=(1080,1920) if spec['aspect_ratio']=='9:16' else (1920,1080)
    cards=plan['cards'];results={name:True for name in CHECKS};results['overlay_legibility']=bool(cards)
    for c in cards:
        results['overlay_safe_area'] &= (0<=c['x'] and c['x']+c['width']<=w and c['y']>=math.ceil(h*.12) and c['y']+c['height']+8<=math.floor(h/3))
        results['overlay_duration'] &= readable_frames(c['start_frame'],c['end_frame'],c.get('cut_frames',[]))>=75 and 0<=c['start_frame']<c['end_frame']<=round(duration_ms*30/1000)+1
        if c['kind']=='title':results['overlay_duration'] &= (c['end_frame']-c['start_frame'])/30<=3.5
        results['overlay_text_limits'] &= 1<=len(c['text'].splitlines())<=2 and all(len(x)<=42 for x in c['text'].splitlines())
        results['overlay_cut_clearance'] &= all(frame in c.get('cut_frames',[]) for cut in cuts for frame in (round(cut*30/1000)-1,round(cut*30/1000),round(cut*30/1000)+1) if c['start_frame']<=frame<c['end_frame'])
        luminance=((1-c.get('pill_opacity',0)+.055)/1.055)**2.4
        results['overlay_legibility'] &= c.get('cap_height',0)>=.05*h and .55<=c.get('pill_opacity',0)<=.65 and 1.05/(luminance+.05)>=4.5
        expected=layout(c['text'],w,h,c['position'],production=plan.get('production',False))
        results['overlay_render_binding'] &= all(c.get(k)==expected[k] for k in ('text','x','y','width','height','fontsize','position','cap_height','pill_opacity','padding'))
    results['overlay_render_binding'] &= (plan['render_sha256']==filehash(video) and plan['style_sha256']==filehash(STYLE)
        and plan['font_sha256']==filehash(font_path()) and plan['plan_sha256']==digest({k:v for k,v in plan.items() if k not in ('render_sha256','plan_sha256')}))
    return [dict(name=name,status='pass' if ok else 'fail',reason=None if ok else 'Overlay geometry, timing or render binding failed',required=True) for name,ok in results.items()]
