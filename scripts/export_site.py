#!/usr/bin/env python3
"""Local, read-only Episode -> public static bundle. No providers or deployment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from PIL import Image
from pipelines.ffmpeg.contracts import read_json, validate_report
from pipelines.ffmpeg.evaluator import validate_eval_report, sha256_file
from pipelines.ffmpeg.eval_tools import read_jsonl
from pipelines.ffmpeg.scoreboard_html import episode_metrics
from pipelines.ffmpeg.factory_history import collect_factory_history, PR_URL
from pipelines.ffmpeg.disk_safety import require_free_disk
from pipelines.ffmpeg.eval_gates import CHECK_NAMES
from pipelines.ffmpeg.eval_judge import CRITERIA

MAX_BYTES = 8_000_000
ID = re.compile(r'[a-z0-9][a-z0-9-]{0,63}\Z')
SHOT = re.compile(r'shot-[0-9]{2,6}\Z')
MARKER = '.framefoundry-export.json'
PUBLIC_CHECKS=set(CHECK_NAMES)|{'voiceover_integrity','narrative_schema','output_encoding','output_dimensions','output_fps','timeline_and_padding','render_duration','goal_duration','goal_coverage','overlay_safe_area','overlay_duration','overlay_text_limits','overlay_cut_clearance','overlay_render_binding'}

class ExportError(RuntimeError): pass


def timestamp(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None: raise ValueError('Timezone required')
    return stamp


def regular(directory, name):
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise ValueError('Artifact absent or unsafe')
    return path


def command(args):
    try: result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError): raise ExportError('Local FFmpeg/ffprobe failed; previous bundle retained') from None
    if result.returncode: raise ExportError('Local FFmpeg/ffprobe failed; previous bundle retained')
    return result.stdout


def probe(path):
    data = json.loads(command(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))
    duration = float(data['format']['duration'])
    video = next(s for s in data['streams'] if s['codec_type'] == 'video')
    if not math.isfinite(duration) or not 0 < duration <= 180: raise ExportError('Invalid teaser duration')
    if not any(s.get('codec_type') == 'audio' for s in data['streams']): raise ExportError('Verified clip lacks audio')
    return data, video, duration


def encode_video(source, destination, scratch):
    """Bounded two-pass bitrate, metadata stripped; hard size/dimension checks."""
    _, _, duration = probe(source)
    filters = 'scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30'
    for attempt, fraction in enumerate((.93, .82)):
        bitrate = int(MAX_BYTES * fraction * 8 / duration - 96000)
        if bitrate < 16000: raise ExportError('Clip cannot fit the web-video size bound')
        log = scratch / f'pass-{attempt}'
        base = ['ffmpeg','-nostdin','-y','-v','error','-threads','2','-filter_threads','1','-i',str(source),
                '-map','0:v:0','-vf',filters,'-c:v','libx264','-preset','fast','-pix_fmt','yuv420p',
                '-b:v',str(bitrate),'-passlogfile',str(log),'-map_metadata','-1','-map_chapters','-1']
        command(base + ['-pass','1','-an','-f','null',os.devnull])
        command(base + ['-pass','2','-map','0:a:0','-c:a','aac','-b:a','96k','-movflags','+faststart',str(destination)])
        if destination.stat().st_size <= MAX_BYTES: break
    _, video, encoded_duration = probe(destination)
    if (destination.stat().st_size > MAX_BYTES or video['codec_name'] != 'h264'
        or (video['width'],video['height']) != (720,1280) or video.get('pix_fmt') != 'yuv420p'
        or abs(encoded_duration-duration) > .15):
        raise ExportError('Encoded video failed size, format or complete-duration verification')
    return dict(bytes=destination.stat().st_size,width=720,height=1280,codec='h264',
                duration_seconds=encoded_duration,sha256=sha256_file(destination))


def checks(items):
    """Never publish free-form paths, measurement payloads, or provider diagnostics."""
    return [dict(name=c['name'],status=c['status'],reason=None if c['status']=='pass' else 'Recorded check did not pass')
            for c in items if isinstance(c,dict) and c.get('name') in PUBLIC_CHECKS
            and c.get('status') in ('pass','fail','not_run','unavailable')]


def passed_evaluation(report):
    names={c['name'] for c in report['gates']['checks']}
    return (report['scores']['G']==1 and report['gates']['passed']
            and set(CHECK_NAMES).issubset(names)
            and all(c['status']=='pass' for c in report['gates']['checks']))


def public_eval(report):
    scores = {k:report['scores'].get(k) for k in ('J','Q','D','G')}
    cost = report['model_cost']; total = cost['usd']; q = scores['Q']
    public_cost = dict(usd=total,known_usd=cost.get('known_usd'),status=cost['status'],
                       quality_per_dollar=q/total if q is not None and total is not None and total>0 else None)
    judge = report['judge']
    rubric = {k:v for k,v in (judge.get('scores') or {}).items() if k in CRITERIA} or None
    evidence = {k:[s for s in ids if isinstance(s,str) and SHOT.fullmatch(s)]
                for k,ids in (judge.get('evidence') or {}).items() if k in CRITERIA} or None
    return dict(episode_id=report['episode_id'],created_at=report['created_at'],scores=scores,
                model_cost=public_cost,gates=dict(passed=report['gates']['passed'],checks=checks(report['gates']['checks'])),
                judge=dict(status=judge['status'],scores=rubric,evidence=evidence,
                           weakest_shot=judge['weakest_shot'] if isinstance(judge['weakest_shot'],str) and SHOT.fullmatch(judge['weakest_shot']) else None,
                           verdict=judge['verdict']),first_try_passed=report['first_try_passed'],
                timing=dict(elapsed_seconds=report['timing']['elapsed_seconds']))


def public_qa(qa):
    return dict(episode_id=qa['episode_id'],passed=qa['passed'],result=qa['result'],
                integrated_lufs=qa['integrated_lufs'],true_peak_dbtp=qa['true_peak_dbtp'],checks=checks(qa['checks']))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def collect(episodes, evals):
    reports, bad = read_jsonl(evals/'scoreboard.jsonl')
    directories = []
    if episodes.is_dir():
        for directory in sorted(episodes.iterdir()):
            if not ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir(): continue
            directories.append(directory)
            try: reports.append(read_json(regular(directory,'eval-report.json')))
            except (ValueError,OSError): bad += 1
    latest = {}
    for report in reports:
        try:
            validate_eval_report(report)
            if not ID.fullmatch(report['episode_id']): raise ValueError()
            key = timestamp(report['created_at'])
            previous = latest.get(report['episode_id'])
            if previous is None or key >= timestamp(previous['created_at']): latest[report['episode_id']] = report
        except (ValueError,TypeError,KeyError,RuntimeError): bad += 1
    return directories, latest, bad


def export_site(*, episodes_root=REPO/'episodes',evals_root=REPO/'evals',site_root=REPO/'site',factory=None):
    episodes = Path(episodes_root).expanduser().resolve(); evals = Path(evals_root).expanduser().resolve()
    site = Path(site_root).expanduser().resolve(); public = site/'public'
    if site == episodes or site.is_relative_to(episodes) or episodes.is_relative_to(site):
        raise ExportError('Site output and Episode storage must not overlap')
    for name in ('index.html','studio.html','vercel.json'): regular(site,name)
    if public.is_symlink(): raise ExportError('Public output symlink is not allowed')
    if public.exists():
        try:
            marker=read_json(regular(public,MARKER))
            if not public.is_dir() or not isinstance(marker,dict) or marker.get('owner')!='framefoundry-static-export':raise ValueError()
        except (OSError,ValueError):raise ExportError('Refusing to replace unowned public output') from None
    require_free_disk(site)
    with (site/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage = Path(tempfile.mkdtemp(prefix='.public-stage-',dir=site)); backup = site/'.public-backup'
        try:
            runs_root=stage/'runs'; runs_root.mkdir()
            for name in ('index.html','studio.html'): shutil.copyfile(regular(site,name),stage/name)
            if (site/'pitch.html').exists() or (site/'pitch.html').is_symlink():
                shutil.copyfile(regular(site,'pitch.html'),stage/'pitch.html')
            (stage/'robots.txt').write_text('User-agent: *\nDisallow: /\n')
            write_json(stage/MARKER,dict(schema_version='1.0',owner='framefoundry-static-export'))
            directories, latest, bad = collect(episodes,evals)
            runs=[]; omitted=[]
            for directory in directories:
                eid=directory.name; report=latest.get(eid)
                if report is None: omitted.append(dict(id=eid,reason_code='evaluation_unavailable'));continue
                try:
                    canonical=read_json(regular(directory,'eval-report.json'));validate_eval_report(canonical)
                    if canonical['episode_id']!=eid or not passed_evaluation(canonical):raise ValueError()
                    qa=read_json(regular(directory,'qa-report.json'));validate_report(qa)
                    if qa['episode_id']!=eid or not qa['passed']: raise ValueError()
                    if not (-15<=qa['integrated_lufs']<=-13 and qa['true_peak_dbtp']<=-1): raise ValueError()
                    if not passed_evaluation(report):raise ValueError()
                    source=regular(directory,'finished.mp4');sheet=regular(directory,'contact-sheet.jpg')
                    digest=sha256_file(source)
                    if not source.stat().st_size or digest!=report['render_hash'] or digest!=canonical['render_hash']: raise ValueError()
                    with Image.open(sheet) as image: image.verify()
                except (ValueError,KeyError,TypeError,OSError,RuntimeError):
                    omitted.append(dict(id=eid,reason_code='unverified_or_stale_artifacts'));continue
                dest=runs_root/eid; dest.mkdir()
                with tempfile.TemporaryDirectory(prefix='.encode-',dir=site) as scratch:
                    output=encode_video(source,dest/'finished.mp4',Path(scratch))
                # Check that a concurrent local render did not change the accepted source.
                if sha256_file(source)!=digest: raise ExportError('Source changed during export; previous bundle retained')
                # Re-save pixels, dropping EXIF/comment metadata from the public contact sheet.
                with Image.open(sheet) as image: image.convert('RGB').save(dest/'contact-sheet.jpg',quality=88)
                ev=public_eval(report); q=public_qa(qa)
                write_json(dest/'eval-report.json',ev);write_json(dest/'qa-report.json',q)
                url='/runs/'+eid+'/'
                run=dict(id=eid,created_at=report['created_at'],scores=ev['scores'],cost=ev['model_cost'],qa=q,
                         gates=ev['gates'],judge=ev['judge'],first_try_passed=ev['first_try_passed'],
                         elapsed_seconds=ev['timing']['elapsed_seconds'],video_url=url+'finished.mp4',
                         contact_sheet_url=url+'contact-sheet.jpg',output=output,source_render_sha256=digest)
                write_json(dest/'run.json',run)
                runs.append({k:run[k] for k in ('id','created_at','scores','cost','first_try_passed','elapsed_seconds','video_url','contact_sheet_url')} |
                            dict(data_url=url+'run.json',qa_passed=True))
            runs.sort(key=lambda r:(timestamp(r['created_at']),r['id']),reverse=True)
            now=datetime.now(timezone.utc).isoformat(); lead=runs[0]['id'] if runs else None
            write_json(runs_root/'index.json',dict(schema_version='1.0',generated_at=now,mode='recorded-demo',lead_run_id=lead,
                latest_run_id=lead,runs=runs,omitted=omitted,requested_ids=['az-0700','az-0835']))
            metrics=episode_metrics(list(latest.values()),episodes_root=episodes,unreadable_lines=bad)
            metrics['shipped']=len(runs);metrics['cost_per_shipped']=metrics['cohort_subtotal']/len(runs) if runs else None
            if bad: metrics['cost_complete']=False
            cohort={k:metrics[k] for k in ('qa_known','qa_passed','shipped','cost_per_shipped','cohort_subtotal','cost_complete','estimated','known_subtotal','first_known','first_passed','pipeline_median')}
            cohort.update(episode_count=len(metrics['rows']),qa_pass_rate=metrics['qa_passed']/metrics['qa_known'] if metrics['qa_known'] else None,
                          first_try_pass_rate=metrics['first_passed']/metrics['first_known'] if metrics['first_known'] else None)
            history=collect_factory_history() if factory is None else factory
            timings=[w['coverage_minutes'] for w in history['work_orders'] if w.get('coverage_minutes') is not None]
            public_factory=dict(ticket_to_pr_minutes=statistics.median(timings) if timings else None,
                scope='median first recorded PR-title coverage; not completion or delivery',sample_count=len(timings),
                observed_at=history['fetched_at'],pr_url=PR_URL,
                work_order_count=len(history['work_orders']),
                pr_count=1 if isinstance(history.get('pull_request'),dict) else None,
                commit_count=None,branch=None,base_sha=None,head_sha=None,git_observed_at=None)
            # Local read-only Git scope; no auth, remote calls, or total-history count.
            try:
                def git(*args):
                    result=subprocess.run(['git',*args],cwd=REPO,capture_output=True,text=True,timeout=5,check=True)
                    return result.stdout.strip()
                branch=git('branch','--show-current');base=git('rev-parse','main');head=git('rev-parse','HEAD')
                commits=git('rev-list','--count',base+'..'+head)
                if (re.fullmatch(r'[a-zA-Z0-9_./-]{1,160}',branch) and re.fullmatch(r'[a-f0-9]{40}',base)
                    and re.fullmatch(r'[a-f0-9]{40}',head) and commits.isascii() and commits.isdecimal()):
                    public_factory.update(commit_count=int(commits),branch=branch,base_sha=base,head_sha=head,git_observed_at=now)
            except (OSError,subprocess.SubprocessError):
                pass
            trend=[dict(id=r['episode_id'],J=r['scores']['J'],Q=r['scores']['Q'],D=r['scores']['D'],quality_per_dollar=r['qpd']) for r in metrics['rows']]
            write_json(runs_root/'scoreboard.json',dict(schema_version='1.0',generated_at=now,cohort=cohort,latest=runs[0] if runs else None,
                                                     trend=trend,factory=public_factory))
            # Recovery directory is never followed/deleted if it is not ours.
            if backup.exists() or backup.is_symlink(): raise ExportError('Previous export backup exists; inspect before retrying')
            if public.exists():public.rename(backup)
            try:stage.rename(public)
            except OSError:
                if backup.exists():backup.rename(public)
                raise ExportError('Bundle replacement failed; previous bundle retained') from None
            if backup.exists():shutil.rmtree(backup)
            return dict(exported=len(runs),omitted=len(omitted),lead_run_id=lead)
        finally:
            if stage.exists():shutil.rmtree(stage)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-root',type=Path,default=REPO/'episodes')
    parser.add_argument('--evals-root',type=Path,default=REPO/'evals')
    parser.add_argument('--site-root',type=Path,default=REPO/'site')
    args=parser.parse_args(argv)
    try:
        result=export_site(**vars(args));print(json.dumps(result))
        if not result['exported']:print('No verified recorded runs available; pages show N/A. Export on the Mac before the demo.',file=sys.stderr)
        return 0
    except Exception as error:
        print('Static export failed: '+(str(error) if isinstance(error,ExportError) else type(error).__name__)+'. No deployment performed.',file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())
