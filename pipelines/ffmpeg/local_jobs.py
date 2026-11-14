"""One owned CLI subprocess; persistent identities, no paid automatic resume."""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,DecimalException
import os,sys,subprocess,threading,uuid,time,fcntl,signal
from .contracts import atomic_write_json,StageError
from .episode import episode_lock
from .goal_parser import parse_goal
from .local_media import load,detail

TERMINAL={'completed','failed','interrupted'}


class JobManager:
    def __init__(self,root,evals,data,spawn=subprocess.Popen):
        self.root=Path(root).resolve();self.evals=Path(evals).resolve();self.data=Path(data).resolve()
        for p in (self.root,self.data,self.data/'jobs',self.data/'uploads',self.data/'logs'):p.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.spawn=spawn;self.lock=threading.RLock();self.jobs={};self.processes={};self.handles={};self.owner=None
        for path in (self.data/'jobs').glob('*.json'):
            try:
                import json
                if path.is_symlink():continue
                job=json.loads(path.read_text());eid=job['episode_id']
                if path.name!=eid+'.json':continue
                from .contracts import resolve_episode
                resolve_episode(self.root,eid)
                self.jobs[eid]=job
            except (OSError,ValueError,KeyError,StageError):continue

    def start(self):
        self.owner=(self.root/'.ui.lock').open('a')
        try:fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.owner.close();self.owner=None;raise RuntimeError('Another UI server owns this Episode root') from None
        for job in self.jobs.values():
            if job['stage'] not in TERMINAL:
                job.update(stage='interrupted',error='Server restarted; no automatic resume')
                self.persist(job)

    def persist(self,job):atomic_write_json(self.data/'jobs'/(job['episode_id']+'.json'),job)

    def finish(self,eid):
        process=self.processes[eid];code=process.poll()
        if code is None:return True
        job=self.jobs[eid];state=load(self.root,eid,'state.json') or {}
        stage=state.get('stage')
        job['stage']=stage if stage in TERMINAL else 'failed'
        if job['stage']=='failed':job['error']='Pipeline failed; see Episode report' if stage=='failed' else f'Pipeline exited ({code}) before a terminal report'
        job['elapsed_seconds']=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(job['created_at'])).total_seconds())
        self.persist(job);self.handles.pop(eid).close();del self.processes[eid]
        return False

    def active(self):
        with self.lock:return self._active()

    def _active(self):
        for eid in list(self.processes):
            if self.finish(eid):return eid
        return None

    @staticmethod
    def footage(path):
        path=Path(path).expanduser().resolve()
        if not path.is_dir():raise ValueError('Footage folder is unavailable')
        media=[p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4','.mov')]
        if not media:raise ValueError('Folder contains no supported MP4/MOV videos')
        if len(media)>40:raise ValueError('At most 40 media files are supported')
        return path

    def launch(self,payload,parent=None):
        with self.lock:
            if self.active():raise StageError('RUN_ACTIVE','refinery','Another Generation is active')
            goal=payload['goal'].strip()
            if not goal or len(goal)>10000:raise ValueError('Goal must contain 1–10000 characters')
            parse_goal(goal,target_seconds=payload.get('target_seconds'),aspect=payload['aspect'])
            if payload['voiceover']:
                from .voiceover import configuration
                configuration()
            if bool(payload.get('footage_path'))==bool(payload.get('upload_id')):raise ValueError('Choose exactly one folder or upload')
            if payload.get('upload_id'):
                token=payload['upload_id']
                if not len(token)==32 or any(c not in '0123456789abcdef' for c in token):raise ValueError('Invalid upload identity')
                folder=self.data/'uploads'/token
                if folder.is_symlink():raise ValueError('Unsafe upload')
            else:folder=payload['footage_path']
            folder=self.footage(folder)
            # Probe existing CLI ownership; the child also takes the same lock.
            with episode_lock(self.root):pass
            try:budget=Decimal(os.environ.get('FRAMEFOUNDRY_EPISODE_API_BUDGET_USD','0.10'))
            except DecimalException:raise ValueError('Invalid configured API budget') from None
            if not budget.is_finite() or budget<=0:raise ValueError('Invalid configured API budget')
            budget=min(budget,Decimal('0.10'))
            eid='ui-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:6]
            settings=dict(target_seconds=payload.get('target_seconds'),aspect=payload['aspect'],voiceover=payload['voiceover'])
            job=dict(episode_id=eid,goal=goal,footage_path=str(folder),settings=settings,parent_episode=parent,
                     stage='starting',created_at=datetime.now(timezone.utc).isoformat(),error=None)
            self.jobs[eid]=job;self.persist(job)
            args=[sys.executable,'-m','pipelines.ffmpeg.cli','--footage-dir',str(folder),'--goal='+goal,
                  '--episode',eid,'--episodes-root',str(self.root),'--evals-root',str(self.evals),'--no-planner-retry','--aspect',payload['aspect']]
            if payload.get('target_seconds') is not None:args+=['--duration-seconds',str(payload['target_seconds'])]
            if payload['voiceover']:args+=['--voiceover']
            env=os.environ.copy();env['FRAMEFOUNDRY_EPISODE_API_BUDGET_USD']=str(budget)
            log=(self.data/'logs'/(eid+'.log')).open('wb')
            try:
                process=self.spawn(args,cwd=str(Path(__file__).resolve().parents[2]),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except Exception:
                log.close();job.update(stage='failed',error='Could not start pipeline process');self.persist(job);raise ValueError(job['error']) from None
            self.processes[eid]=process;self.handles[eid]=log
            return self.get(eid)

    def get(self,eid):
        with self.lock:return self._get(eid)

    def _get(self,eid):
        active=eid in self.processes and self.finish(eid)
        job=self.jobs.get(eid)
        if not job and not (self.root/eid).is_dir():raise FileNotFoundError('Episode unavailable')
        result=detail(self.root,eid,job,active)
        if active:
            from datetime import datetime
            result['elapsed_seconds']=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(job['created_at'])).total_seconds())
        if not active and result['elapsed_seconds'] is None and job:
            result['elapsed_seconds']=job.get('elapsed_seconds')
        result['can_regenerate']=bool(self.source(eid))
        return result

    def source(self,eid):
        job=self.jobs.get(eid)
        if job:
            try:return str(self.footage(job['footage_path']))
            except (OSError,ValueError):return None
        a=load(self.root,eid,'analysis.json') or {};sources=a.get('sources',[])
        parents={str(Path(s['path']).parent) for s in sources if isinstance(s,dict) and s.get('path')}
        if len(parents)!=1:return None
        try:return str(self.footage(parents.pop()))
        except (ValueError,OSError):return None

    def history(self):
        ids=set(self.jobs)
        ids.update(p.name for p in self.root.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith('.'))
        rows=[]
        for eid in ids:
            try:rows.append(self.get(eid))
            except (OSError,ValueError,KeyError,TypeError,StageError):continue
        return sorted(rows,key=lambda r:(r['created_at'],r['episode_id']),reverse=True)

    def close(self):
        for eid,process in list(self.processes.items()):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                except ProcessLookupError:pass
            self.jobs[eid].update(stage='interrupted',error='Server shutdown; no automatic resume');self.persist(self.jobs[eid])
            self.handles[eid].close()
        self.processes.clear();self.handles.clear()
        if self.owner:self.owner.close();self.owner=None
