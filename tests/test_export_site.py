"""Static publication uses strict synthetic records; real FFmpeg proof is separate."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import pytest
from PIL import Image
from scripts import export_site as exporter
from pipelines.ffmpeg.contracts import EpisodeContext,StageError,failure_report,validate_report
from tests.test_scoreboard_html import report

FACTORY=dict(work_orders=[dict(coverage_minutes=12),dict(coverage_minutes=20)],fetched_at='2026-11-14T00:00:00Z')
REQUIRED={'narrative_schema','readable_media','output_encoding','output_dimensions','output_fps','loudness','true_peak','timeline_and_padding','render_duration','goal_duration','contact_sheet'}

@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path/'episodes';root.mkdir();site=tmp_path/'site';site.mkdir();evals=tmp_path/'evals';evals.mkdir()
    for name in ('index.html','studio.html','vercel.json'):shutil.copyfile(exporter.REPO/'site'/name,site/name)
    monkeypatch.setattr(exporter,'require_free_disk',lambda *a:None)
    return root,evals,site


def run_fixture(setup,report,eid='az-0900',*,created='2026-11-14T17:00:00Z',passed=True):
    root,_,_=setup;directory=root/eid;directory.mkdir()
    (directory/'finished.mp4').write_bytes(b'export')
    Image.new('RGB',(40,60),'teal').save(directory/'contact-sheet.jpg')
    ev=report(eid,created=created,passed=passed)
    ev['goal']='/Users/rahat/private.mp4 token=SECRET'
    ev['limitations']=['private@example.org SECRET']
    ev['gates']['checks']=[dict(name=name,status='pass' if passed else 'fail',reason=None) for name in exporter.CHECK_NAMES]
    qa=failure_report(EpisodeContext(eid,directory,'/Users/rahat/private.mp4'),StageError('FIXTURE','validator','SECRET'))
    if passed:
        qa.update(stage='delivery',passed=True,result='passed',coverage=dict(status='full',limitations=[]),error=None,output_path='finished.mp4',timeline_path='timeline.otio',
                  contact_sheet_path='contact-sheet.jpg',integrated_lufs=-14,true_peak_dbtp=-2,
                  checks=[dict(name=n,status='pass',reason=None,required=True) for n in sorted(REQUIRED)])
    validate_report(qa)
    (directory/'qa-report.json').write_text(json.dumps(qa));(directory/'eval-report.json').write_text(json.dumps(ev))
    (directory/'transcript.json').write_text('{"secret":"SECRET"}')
    return directory,ev,qa


def fake_encoder(source,destination,scratch):
    destination.write_bytes(b'public copy')
    return dict(bytes=11,width=720,height=1280,codec='h264',duration_seconds=2,sha256=exporter.sha256_file(destination))


def export(setup,**extra):
    root,evals,site=setup
    return exporter.export_site(episodes_root=root,evals_root=evals,site_root=site,factory=FACTORY,**extra)


def test_empty_bundle_is_deployable_and_truthful(setup):
    result=export(setup);site=setup[2]
    assert result==dict(exported=0,omitted=0,lead_run_id=None)
    assert (site/'public/index.html').exists() and (site/'public/studio.html').exists()
    index=json.loads((site/'public/runs/index.json').read_text())
    assert index['runs']==[] and index['lead_run_id'] is None
    board=json.loads((site/'public/runs/scoreboard.json').read_text())
    assert board['cohort']['cost_per_shipped'] is None and board['cohort']['qa_pass_rate'] is None
    assert board['factory']['ticket_to_pr_minutes']==16


def test_real_ids_latest_picker_and_public_projections_exclude_private_data(setup,report,monkeypatch):
    originals={}
    for eid,time in [('az-0700','2026-11-14T15:00:00Z'),('az-0835','2026-11-14T16:00:00Z'),('az-0900','2026-11-14T17:00:00Z')]:
        d,_,_=run_fixture(setup,report,eid,created=time)
        originals[eid]={p.name:p.read_bytes() for p in d.iterdir()}
    monkeypatch.setattr(exporter,'encode_video',fake_encoder)
    result=export(setup);public=setup[2]/'public'
    assert result['exported']==3 and result['lead_run_id']=='az-0900'
    index=json.loads((public/'runs/index.json').read_text())
    assert [r['id'] for r in index['runs']]==['az-0900','az-0835','az-0700']
    for eid,files in originals.items():
        assert {p.name:p.read_bytes() for p in (setup[0]/eid).iterdir()}==files
        dest=public/'runs'/eid
        assert set(p.name for p in dest.iterdir())=={'run.json','eval-report.json','qa-report.json','finished.mp4','contact-sheet.jpg'}
        text=''.join(p.read_text() for p in dest.glob('*.json'))
        assert 'SECRET' not in text and '/Users/' not in text and 'private@example' not in text and 'transcript' not in text
    board=json.loads((public/'runs/scoreboard.json').read_text())
    assert board['cohort']['shipped']==3 and board['cohort']['qa_pass_rate']==1


@pytest.mark.parametrize('kind',['qa-fail','qa-invalid','hash-stale','eval-missing','sheet-missing','sheet-corrupt','media-symlink','qa-symlink','eval-symlink','directory-symlink','eval-failed','eval-missing-gate'])
def test_unverified_or_unsafe_artifacts_are_never_exported(setup,report,monkeypatch,kind):
    d,ev,qa=run_fixture(setup,report,passed=kind!='eval-failed')
    if kind=='qa-fail':qa.update(passed=False,result='failed',output_path=None,error=StageError('BAD','validator','failed').as_dict());(d/'qa-report.json').write_text(json.dumps(qa))
    if kind=='qa-invalid':qa['checks']=[];(d/'qa-report.json').write_text(json.dumps(qa))
    if kind=='hash-stale':(d/'finished.mp4').write_bytes(b'new render')
    if kind=='eval-missing':(d/'eval-report.json').unlink()
    if kind=='eval-missing-gate':ev['gates']['checks'].pop();(d/'eval-report.json').write_text(json.dumps(ev))
    if kind=='sheet-missing':(d/'contact-sheet.jpg').unlink()
    if kind=='sheet-corrupt':(d/'contact-sheet.jpg').write_bytes(b'bad')
    mapping={'media-symlink':'finished.mp4','qa-symlink':'qa-report.json','eval-symlink':'eval-report.json'}
    if kind in mapping:
        path=d/mapping[kind];outside=d.parent/(mapping[kind]+'.outside');path.rename(outside);path.symlink_to(outside)
    if kind=='directory-symlink':
        outside=d.parent/'outside';d.rename(outside);d.symlink_to(outside,target_is_directory=True)
    monkeypatch.setattr(exporter,'encode_video',lambda *a:pytest.fail('No encoding invalid data'))
    result=export(setup)
    assert result['exported']==0
    assert not (setup[2]/'public/runs/az-0900').exists()


def test_failed_encoder_preserves_previous_bundle(setup,report,monkeypatch):
    export(setup);previous=(setup[2]/'public/runs/index.json').read_bytes()
    run_fixture(setup,report)
    monkeypatch.setattr(exporter,'encode_video',lambda *a:(_ for _ in ()).throw(exporter.ExportError('encode failed')))
    with pytest.raises(exporter.ExportError):export(setup)
    assert (setup[2]/'public/runs/index.json').read_bytes()==previous
    assert not list(setup[2].glob('.public-stage-*')) and not list(setup[2].glob('.encode-*'))


def test_stale_exported_runs_are_removed_by_successful_rebuild(setup,report,monkeypatch):
    d,_,_=run_fixture(setup,report);monkeypatch.setattr(exporter,'encode_video',fake_encoder);export(setup)
    (d/'finished.mp4').write_bytes(b'changed')
    result=export(setup)
    assert result['exported']==0 and not (setup[2]/'public/runs/az-0900').exists()


def test_history_cumulative_cost_includes_failed_overhead_once(setup,report,monkeypatch):
    _,valid,_=run_fixture(setup,report)
    _,failed,_=run_fixture(setup,report,'failed',passed=False)
    failed['model_cost'].update(usd=3,known_usd=3)
    (setup[0]/'failed/eval-report.json').write_text(json.dumps(failed))
    old=deepcopy(valid);old['created_at']='2026-11-13T00:00:00Z';old['model_cost'].update(usd=99,known_usd=99)
    (setup[1]/'scoreboard.jsonl').write_text(json.dumps(old)+'\n'+json.dumps(valid)+'\n')
    monkeypatch.setattr(exporter,'encode_video',fake_encoder);export(setup)
    board=json.loads((setup[2]/'public/runs/scoreboard.json').read_text())
    assert board['cohort']['episode_count']==2 and board['cohort']['shipped']==1
    assert board['cohort']['cost_per_shipped']==5 and board['cohort']['qa_pass_rate']==.5


@pytest.mark.parametrize('kind',['public-symlink','unowned-output','overlap'])
def test_output_safety_preserves_unowned_files(setup,kind):
    root,evals,site=setup
    if kind=='public-symlink':(site/'public').symlink_to(root,target_is_directory=True)
    if kind=='unowned-output':(site/'public').mkdir();(site/'public/keep').write_text('safe')
    with pytest.raises(exporter.ExportError):
        exporter.export_site(episodes_root=root,evals_root=evals,site_root=root if kind=='overlap' else site,factory=FACTORY)
    if kind=='unowned-output':assert (site/'public/keep').read_text()=='safe'


def test_vercel_is_static_clean_noindex_and_uploads_gitignored_media():
    site=exporter.REPO/'site';config=json.loads((site/'vercel.json').read_text())
    assert config['outputDirectory']=='public' and config['buildCommand'] is None and config['framework'] is None
    assert config['cleanUrls'] is True
    assert any(h['source']=='/(.*)' and dict((v['key'],v['value']) for v in h['headers']).get('X-Robots-Tag')=='noindex' for h in config['headers'])
    assert '*.mp4' not in (site/'.vercelignore').read_text()
    output=subprocess.run(['git','check-ignore','site/public/runs/az-0900/finished.mp4'],cwd=exporter.REPO,capture_output=True)
    assert output.returncode==0


def test_real_ffmpeg_reencode_is_bounded_complete_and_source_unchanged(tmp_path):
    source=tmp_path/'original.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30:duration=2',
                    '-f','lavfi','-i','sine=frequency=650:sample_rate=48000:duration=2','-c:v','libx264','-pix_fmt','yuv420p',
                    '-c:a','aac','-shortest',str(source)],check=True,capture_output=True)
    original=source.read_bytes();output=tmp_path/'web.mp4'
    metadata=exporter.encode_video(source,output,tmp_path)
    assert (metadata['width'],metadata['height'],metadata['codec'])==(720,1280,'h264')
    assert output.stat().st_size<=8_000_000 and metadata['duration_seconds']==pytest.approx(2,abs=.15)
    assert source.read_bytes()==original and metadata['sha256']==exporter.sha256_file(output)

@pytest.mark.parametrize('always_oversized',[False,True])
def test_encoder_retries_oversize_once_and_hard_fails_if_still_too_large(tmp_path,monkeypatch,always_oversized):
    source=tmp_path/'source.mp4';source.write_bytes(b'source');destination=tmp_path/'web.mp4';calls=[]
    monkeypatch.setattr(exporter,'probe',lambda p:({},dict(codec_name='h264',width=720,height=1280,pix_fmt='yuv420p'),45))
    def command(args):
        calls.append(args)
        if '-pass' in args and args[args.index('-pass')+1]=='2':
            passes=sum(a[a.index('-pass')+1]=='2' for a in calls)
            destination.write_bytes(b'0'*(8_000_001 if always_oversized or passes==1 else 7_000_000))
        return ''
    monkeypatch.setattr(exporter,'command',command)
    if always_oversized:
        with pytest.raises(exporter.ExportError,match='verification'):exporter.encode_video(source,destination,tmp_path)
    else:assert exporter.encode_video(source,destination,tmp_path)['bytes']==7_000_000
    assert len(calls)==4
    assert int(calls[2][calls[2].index('-b:v')+1])<int(calls[0][calls[0].index('-b:v')+1])
    for args in calls:assert args[args.index('-map_metadata')+1]=='-1' and args[args.index('-map_chapters')+1]=='-1'


def test_concurrent_source_change_aborts_and_keeps_previous_bundle(setup,report,monkeypatch):
    export(setup);previous=(setup[2]/'public/runs/index.json').read_bytes();d,_,_=run_fixture(setup,report)
    def mutate(source,destination,scratch):
        output=fake_encoder(source,destination,scratch);source.write_bytes(b'concurrent rerender');return output
    monkeypatch.setattr(exporter,'encode_video',mutate)
    with pytest.raises(exporter.ExportError,match='Source changed'):export(setup)
    assert (setup[2]/'public/runs/index.json').read_bytes()==previous


def test_replacement_failure_restores_previous_bundle(setup,report,monkeypatch):
    export(setup);previous=(setup[2]/'public/runs/index.json').read_bytes();run_fixture(setup,report)
    monkeypatch.setattr(exporter,'encode_video',fake_encoder);original=Path.rename
    def rename(source,destination):
        if source.name.startswith('.public-stage-'):raise OSError('injected swap failure')
        return original(source,destination)
    monkeypatch.setattr(Path,'rename',rename)
    with pytest.raises(exporter.ExportError,match='replacement failed'):export(setup)
    assert (setup[2]/'public/runs/index.json').read_bytes()==previous
    assert not (setup[2]/'.public-backup').exists()


@pytest.mark.parametrize('marker',['[]','{"owner":"someone-else"}','{broken'])
def test_unowned_or_malformed_marker_cannot_authorize_deletion(setup,marker):
    public=setup[2]/'public';public.mkdir();(public/exporter.MARKER).write_text(marker);(public/'keep').write_text('safe')
    with pytest.raises(exporter.ExportError,match='unowned'):export(setup)
    assert (public/'keep').read_text()=='safe'


def test_optional_pitch_page_is_copied_without_requiring_old_fixtures(setup):
    site=setup[2]
    # Existing two-page fixtures continue to export without pitch.html.
    export(setup)
    assert not (site/'public/pitch.html').exists()
    shutil.copyfile(exporter.REPO/'site/pitch.html',site/'pitch.html')
    export(setup)
    assert (site/'public/pitch.html').read_bytes()==(site/'pitch.html').read_bytes()


def test_pitch_symlink_is_not_followed_and_previous_bundle_survives(setup,tmp_path):
    export(setup);site=setup[2];previous=(site/'public/runs/index.json').read_bytes()
    outside=tmp_path/'private.html';outside.write_text('SECRET')
    (site/'pitch.html').symlink_to(outside)
    with pytest.raises(ValueError):export(setup)
    assert (site/'public/runs/index.json').read_bytes()==previous


def test_overlay_checks_are_allowlisted_but_private_payloads_are_not():
    names={'overlay_safe_area','overlay_duration','overlay_text_limits','overlay_cut_clearance','overlay_render_binding'}
    records=[dict(name=n,status='pass',reason='SECRET',measurements={'source':'/Users/private'}) for n in names]
    records.append(dict(name='private_extra',status='pass',reason='SECRET'))
    public=exporter.checks(records)
    assert {r['name'] for r in public}==names
    assert all(set(r)=={'name','status','reason'} and r['reason'] is None for r in public)


def test_factory_commit_provenance_is_local_scoped_and_safe(setup):
    export(setup)
    factory=json.loads((setup[2]/'public/runs/scoreboard.json').read_text())['factory']
    assert factory['work_order_count']==2 and factory['pr_count'] is None
    assert factory['commit_count'] is None or isinstance(factory['commit_count'],int)
    if factory['commit_count'] is not None:
        assert len(factory['base_sha'])==len(factory['head_sha'])==40
        expected=subprocess.run(['git','rev-list','--count',factory['base_sha']+'..'+factory['head_sha']],cwd=exporter.REPO,capture_output=True,text=True,check=True)
        assert factory['commit_count']==int(expected.stdout)
    assert 'SECRET' not in json.dumps(factory)
