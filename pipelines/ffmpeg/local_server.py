"""Loopback-only, no-auth demo adapter. Browsing cannot call a model."""
from pathlib import Path,PurePosixPath
from contextlib import asynccontextmanager
from decimal import Decimal,DecimalException
import argparse,hashlib,os,shutil,tempfile,uuid
from urllib.parse import urlsplit
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import HTMLResponse,FileResponse
from pydantic import BaseModel,ConfigDict,StrictBool,field_validator
from .local_jobs import JobManager
from .local_media import artifact,ARTIFACTS
from .contracts import StageError

REPO=Path(__file__).resolve().parents[2]
MAX_UPLOAD=32*1024**3

class Generate(BaseModel):
    model_config=ConfigDict(extra='forbid')
    goal:str
    footage_path:str|None=None
    upload_id:str|None=None
    target_seconds:float|None=None
    aspect:str='9:16'
    voiceover:StrictBool=False
    @field_validator('goal')
    @classmethod
    def goal_check(cls,value):
        if not value.strip() or len(value)>10000:raise ValueError('Goal must contain 1–10000 characters')
        return value
    @field_validator('target_seconds',mode='before')
    @classmethod
    def duration_check(cls,value):
        if value is not None:
            if isinstance(value,bool):raise ValueError('Duration must be numeric')
            try:number=Decimal(str(value))
            except DecimalException:raise ValueError('Duration must be numeric') from None
            if not number.is_finite() or not 5<=number<=120:raise ValueError('Duration must be 5–120 seconds')
        return value
    @field_validator('aspect')
    @classmethod
    def aspect_check(cls,value):
        if value not in ('9:16','16:9'):raise ValueError('Aspect must be 9:16 or 16:9')
        return value

class Feedback(BaseModel):
    model_config=ConfigDict(extra='forbid')
    feedback:str
    @field_validator('feedback')
    @classmethod
    def check(cls,value):
        if not value.strip() or len(value)>2000:raise ValueError('Feedback must contain 1–2000 characters')
        return value.strip()

class BodyLimit:
    def __init__(self,app,limit):self.app=app;self.limit=limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        count=0
        limit=self.limit if scope.get('path')=='/api/uploads' else 64*1024
        async def bounded():
            nonlocal count
            message=await receive();count+=len(message.get('body',b''))
            if count>limit:raise HTTPException(413,'Upload/request exceeds configured limit')
            return message
        await self.app(scope,bounded,send)


def create_app(*,episodes_root=REPO/'episodes',evals_root=None,data_root=None,spawn=None,max_upload_bytes=MAX_UPLOAD):
    if max_upload_bytes<=0:raise ValueError('Upload limit must be positive')
    root=Path(episodes_root).expanduser().resolve()
    evals=Path(evals_root or os.environ.get('FRAMEFOUNDRY_EVALS_ROOT',REPO/'evals')).expanduser().resolve()
    data=Path(data_root or Path.home()/'.local/share/framefoundry/ui'/hashlib.sha256(str(root).encode()).hexdigest()[:12]).expanduser().resolve()
    manager=JobManager(root,evals,data,**({'spawn':spawn} if spawn else {}))
    @asynccontextmanager
    async def lifecycle(app):
        manager.start()
        try:yield
        finally:manager.close()
    app=FastAPI(lifespan=lifecycle,docs_url=None,redoc_url=None,openapi_url=None)
    app.state.manager=manager
    app.add_middleware(BodyLimit,limit=max_upload_bytes+1024*1024)
    @app.middleware('http')
    async def safety(request,call_next):
        host=request.headers.get('host','')
        try:parsed=urlsplit('http://'+host);hostname=parsed.hostname
        except ValueError:return HTMLResponse('Invalid Host',status_code=403)
        if hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:
            return HTMLResponse('Localhost access only',status_code=403)
        origin=request.headers.get('origin')
        if request.headers.get('sec-fetch-site')=='cross-site' or (origin and origin!=f'{request.url.scheme}://{host}'):
            return HTMLResponse('Same-origin access only',status_code=403)
        length=request.headers.get('content-length')
        if length:
            try:
                if int(length)<0 or int(length)>(max_upload_bytes+1024*1024 if request.url.path=='/api/uploads' else 64*1024):return HTMLResponse('Upload limit exceeded',status_code=413)
            except ValueError:return HTMLResponse('Invalid request length',status_code=400)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; media-src 'self'; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'none'"
        return response
    def failed(error):
        if isinstance(error,FileNotFoundError):return HTTPException(404,'Episode or artifact unavailable')
        if isinstance(error,StageError):return HTTPException(409 if error.code=='RUN_ACTIVE' else 400,str(error))
        return HTTPException(400,str(error) if isinstance(error,ValueError) else 'Local operation failed')
    @app.get('/',response_class=HTMLResponse)
    def index():return (Path(__file__).with_name('local_ui')/'index.html').read_text()
    @app.get('/api/config')
    def config():
        available=False
        try:
            from .voiceover import configuration
            configuration();available=True
        except StageError:pass
        return dict(voiceover_available=available,max_upload_bytes=max_upload_bytes,max_files=40,poll_interval_ms=1000)
    @app.get('/api/episodes')
    def history():return dict(episodes=manager.history())
    @app.get('/api/episodes/{eid}')
    def episode(eid:str):
        try:return manager.get(eid)
        except (StageError,ValueError,FileNotFoundError,OSError) as e:raise failed(e)
    @app.post('/api/jobs',status_code=202)
    def generate(payload:Generate):
        try:return manager.launch(payload.model_dump())
        except (StageError,ValueError,OSError) as e:raise failed(e)
    @app.post('/api/episodes/{eid}/regenerate',status_code=202)
    def regenerate(eid:str,payload:Feedback):
        try:
            old=manager.get(eid);source=manager.source(eid)
            if not source:raise ValueError('Original footage is unavailable')
            return manager.launch(dict(goal=old['goal']+'\nFeedback hint: '+payload.feedback,footage_path=source,upload_id=None,**old['settings']),parent=eid)
        except (StageError,ValueError,OSError) as e:raise failed(e)
    @app.post('/api/uploads',status_code=201)
    async def upload(request:Request):
        staging=None
        try:
            # Starlette spools to disk; then copy in bounded chunks, never read all media into RAM.
            async with request.form(max_files=40,max_fields=0) as form:
                files=form.getlist('files')
                if not 1<=len(files)<=40 or len(files)!=len(form.multi_items()):raise ValueError('Upload 1–40 video files')
                names=[]
                for file in files:
                    name=getattr(file,'filename','') or ''
                    parts=name.split('/')
                    if (not name or '\\' in name or '\x00' in name or any(p in ('','.','..') for p in parts)
                        or PurePosixPath(name).is_absolute() or PurePosixPath(name).suffix.lower() not in ('.mp4','.mov')
                        or len(name)>512 or name in names):raise ValueError('Unsafe, unsupported or duplicate upload filename')
                    names.append(name)
                token=uuid.uuid4().hex;staging=Path(tempfile.mkdtemp(prefix='.upload-',dir=data/'uploads'));total=0
                for index,(file,name) in enumerate(zip(files,names)):
                    # Flatten immutable selected videos for the existing nonrecursive Refinery.
                    base=PurePosixPath(name).name
                    if len(base.encode())>200:raise ValueError('Upload filename too long')
                    with (staging/(f'{index:03d}-'+base)).open('xb') as output:
                        while chunk:=await file.read(1024*1024):
                            total+=len(chunk)
                            if total>max_upload_bytes:raise HTTPException(413,'Upload limit exceeded')
                            output.write(chunk)
                staging.rename(data/'uploads'/token);staging=None
                return dict(upload_id=token,file_count=len(files),bytes=total)
        except HTTPException:raise
        except (ValueError,OSError) as e:raise failed(e)
        finally:
            if staging is not None:shutil.rmtree(staging,ignore_errors=True)
    @app.get('/api/episodes/{eid}/artifacts/{name}')
    def media(eid:str,name:str,download:bool=False):
        try:
            path=artifact(root,eid,name)
            if name=='finished.mp4' and not manager.get(eid)['video_url']:raise FileNotFoundError()
            return FileResponse(path,filename=name if download else None,content_disposition_type='attachment' if download else 'inline')
        except (ValueError,StageError,FileNotFoundError,OSError) as e:raise failed(e)
    @app.get('/scoreboard',response_class=HTMLResponse)
    def scoreboard():
        from .eval_tools import load_reports
        from .scoreboard_html import render_html
        from .factory_history import collect_factory_history
        try:
            source=evals/'scoreboard.jsonl'
            if source.is_symlink():raise ValueError('Unsafe scoreboard')
            records,bad=load_reports(evals)
            # Factory REST metadata is refreshed only here, never by a model; existing HTML is reused.
            return render_html(records,episodes_root=root,unreadable_lines=bad,factory=collect_factory_history())
        except (OSError,ValueError,AttributeError):
            return render_html([],episodes_root=root,input_unavailable=True)
    return app


def main(argv=None):
    p=argparse.ArgumentParser(description='Local FrameFoundry UI (loopback only)')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--episodes-root',type=Path,default=REPO/'episodes')
    p.add_argument('--evals-root',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--max-upload-gib',type=float,default=32)
    args=p.parse_args(argv)
    if not 1<=args.port<=65535 or not 0<args.max_upload_gib<=1024:p.error('Invalid port or upload limit')
    import uvicorn
    print(f'FrameFoundry UI: http://127.0.0.1:{args.port} (one job; no automatic resume)')
    app=create_app(episodes_root=args.episodes_root,evals_root=args.evals_root,data_root=args.data_root,max_upload_bytes=int(args.max_upload_gib*1024**3))
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
    return 0
