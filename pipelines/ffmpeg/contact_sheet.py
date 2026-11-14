"""Cut-boundary evidence from decoded output frames, not plan thumbnails."""
from .media_tools import media_binary
from pathlib import Path
from PIL import Image,ImageDraw
from .proxy import probe,run
from .contracts import atomic_write_json

def build_contact_sheet(video,cut_times_ms,output,*,production=False):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    info=probe(video);duration=info['duration_ms']; samples=[]
    # Evidence keeps the actual output aspect, with labels outside the image.
    video_stream=info['video']
    w,h=video_stream['width'],video_stream['height']
    scale=min(256/w,240/h);tw,th=max(1,round(w*scale)),max(1,round(h*scale))
    label_h=36
    for index,t in enumerate(cut_times_ms):
        for side,offset in [('before',-34),('after',34)]+([('after_three',100)] if production else []):samples.append((index,side,max(0,min(duration-50,t+offset))))
    if not samples:samples=[(0,'representative',duration//2)]
    tiles=[];rows=[]
    for index,(cut,side,ms) in enumerate(samples):
        path=output.parent/f'qa-frame-{index}.jpg'
        run([media_binary('ffmpeg'),'-y','-v','error','-ss',str(ms/1000),'-i',str(video),'-frames:v','1','-vf',f'scale={tw}:{th}','-threads','1',str(path)])
        with Image.open(path) as image:
            tile=Image.new('RGB',(tw,th+label_h),'black');tile.paste(image,(0,0))
            if production and side in ('after_three','representative'):
                zones=Image.new('RGBA',(tw,th),(0,0,0,0));z=ImageDraw.Draw(zones)
                if h>w:
                    for box in [(0,0,tw,round(th*.10)),(0,round(th*.78),tw,th),(round(tw*.85),0,tw,th)]:z.rectangle(box,fill=(255,0,0,64))
                else:
                    for box in [(0,0,tw,round(th*.10)),(0,round(th*.90),tw,th),(0,0,round(tw*.10),th),(round(tw*.90),0,tw,th)]:z.rectangle(box,fill=(255,0,0,64))
                tile.paste(Image.alpha_composite(image.convert('RGBA'),zones).convert('RGB'),(0,0))
            draw=ImageDraw.Draw(tile);draw.text((4,th+3),f'cut {cut} {side}',fill='white')
            draw.text((4,th+18),f'{ms}ms',fill='white')
        tiles.append(tile);rows.append(dict(cut_index=cut,side=side,time_ms=ms))
        path.unlink()
    cols=min(4,len(tiles));tile_h=th+label_h
    sheet=Image.new('RGB',(cols*tw,((len(tiles)+cols-1)//cols)*tile_h),'black')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%cols)*tw,(i//cols)*tile_h))
    sheet.save(output,quality=75)
    atomic_write_json(output.parent/'contact-sheet-index.json',dict(cut_times_ms=cut_times_ms,samples=rows))
    return dict(path=str(output),samples=rows)
