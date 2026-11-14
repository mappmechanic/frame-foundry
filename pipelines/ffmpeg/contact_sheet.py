"""Cut-boundary evidence from decoded output frames, not plan thumbnails."""
from pathlib import Path
from PIL import Image,ImageDraw
from .proxy import probe,run
from .contracts import atomic_write_json

def build_contact_sheet(video,cut_times_ms,output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    info=probe(video);duration=info['duration_ms']; samples=[]
    for index,t in enumerate(cut_times_ms):
        for side,offset in [('before',-34),('after',34)]:samples.append((index,side,max(0,min(duration-50,t+offset))))
    if not samples:samples=[(0,'representative',duration//2)]
    tiles=[];rows=[]
    for index,(cut,side,ms) in enumerate(samples):
        path=output.parent/f'qa-frame-{index}.jpg'
        run(['ffmpeg','-y','-v','error','-ss',str(ms/1000),'-i',str(video),'-frames:v','1','-vf','scale=240:160:force_original_aspect_ratio=decrease','-threads','1',str(path)])
        with Image.open(path) as image:
            tile=Image.new('RGB',(256,192),'black');tile.paste(image,((256-image.width)//2,0));ImageDraw.Draw(tile).text((5,170),f'cut {cut} {side} {ms}ms',fill='white')
        tiles.append(tile);rows.append(dict(cut_index=cut,side=side,time_ms=ms))
        path.unlink()
    cols=min(5,len(tiles));sheet=Image.new('RGB',(cols*256,((len(tiles)+cols-1)//cols)*192),'black')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%cols)*256,(i//cols)*192))
    sheet.save(output,quality=75)
    atomic_write_json(output.parent/'contact-sheet-index.json',dict(cut_times_ms=cut_times_ms,samples=rows))
    return dict(path=str(output),samples=rows)
