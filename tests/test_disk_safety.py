from types import SimpleNamespace
import pytest
from pipelines.ffmpeg.disk_safety import require_free_disk,cleanup_work_media,MIN_FREE_BYTES
from pipelines.ffmpeg.contracts import StageError

@pytest.mark.parametrize('free',[0,120_000_000,MIN_FREE_BYTES-1])
def test_low_disk_fails_before_processing(tmp_path,monkeypatch,free):
    from pipelines.ffmpeg.cli import run_episode
    folder=tmp_path/'raw';folder.mkdir()
    monkeypatch.setattr('pipelines.ffmpeg.disk_safety.shutil.disk_usage',lambda p:SimpleNamespace(free=free))
    monkeypatch.setattr('pipelines.ffmpeg.cli.run_refinery',lambda *a,**k:pytest.fail('Processing dispatched'))
    monkeypatch.setattr('pipelines.ffmpeg.cli.LLMProvider.from_env',lambda:pytest.fail('Provider initialized'))
    with pytest.raises(StageError,match='at least 1.000 GB') as e:
        run_episode(folder,'teaser',episodes_root=tmp_path/'new'/'episodes')
    assert e.value.code=='DISK_SPACE_LOW' and not (tmp_path/'new').exists()

@pytest.mark.parametrize('free',[MIN_FREE_BYTES,1_300_000_000])
def test_exact_boundary_and_mac_alias_accepted(tmp_path,monkeypatch,free):
    real=tmp_path/'real';real.mkdir();alias=tmp_path/'tmp';alias.symlink_to(real,target_is_directory=True)
    monkeypatch.setattr('pipelines.ffmpeg.disk_safety.shutil.disk_usage',lambda p:SimpleNamespace(free=free))
    assert require_free_disk(alias/'missing'/'episodes')==free


def test_only_owned_work_media_removed_and_sidecars_retained(tmp_path):
    episode=tmp_path/'episode';work=episode/'work'/'attempt-1';work.mkdir(parents=True)
    for name in ('cut-001.mov','candidate.mp4','joined.mov','voiced.mov','audio.wav','narrative-spec.json','timeline.otio','concat.txt'):
        (work/name).write_text('fixture')
    for name in ('finished.mp4','diagnostic.mp4','contact-sheet.jpg'):(episode/name).write_text('keep')
    source=tmp_path/'raw.mov';source.write_text('original')
    (work/'source.mov').symlink_to(source)
    outside=tmp_path/'outside';outside.mkdir();(outside/'cut.mov').write_text('original')
    (work/'alias').symlink_to(outside,target_is_directory=True)
    result=cleanup_work_media(episode)
    assert result=={'deleted':5,'failed':0}
    for name in ('narrative-spec.json','timeline.otio','concat.txt'):assert (work/name).is_file()
    assert (episode/'finished.mp4').is_file() and (episode/'diagnostic.mp4').is_file()
    assert source.read_text()=='original' and (outside/'cut.mov').read_text()=='original'


@pytest.mark.parametrize('keep_work',[False,True])
def test_real_successful_pipeline_default_cleanup_and_opt_out(tmp_path,monkeypatch,keep_work):
    from tests.fixtures.generate_footage import generate_footage
    from pipelines.ffmpeg.cli import run_episode
    from pipelines.ffmpeg.evaluator import evaluate_episode
    for key in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','ELEVENLABS_API_KEY'):monkeypatch.delenv(key,raising=False)
    raw=tmp_path/'raw';generate_footage(raw)
    report=run_episode(raw,'10 second teaser',episodes_root=tmp_path/'episodes',episode_id='test-cleanup',
        transcript_fixture=raw/'transcript-fixture.json',provider=None,planner_retry=False,keep_work=keep_work,evals_root=tmp_path/'evals')
    assert report['passed'] and -15<=report['integrated_lufs']<=-13
    episode=tmp_path/'episodes/test-cleanup'
    media=[p for p in (episode/'work').rglob('*') if p.suffix.lower() in ('.mov','.mp4','.wav')]
    assert bool(media)==keep_work
    for name in ('finished.mp4','contact-sheet.jpg','timeline.otio','analysis.json','eval-report.json'):
        assert (episode/name).stat().st_size>0
    assert (episode/'work/attempt-1/narrative-spec.json').is_file()
    assert (episode/'work/attempt-1/concat.txt').is_file()
    # Removing intermediate media does not prevent independent final-artifact evaluation.
    assert evaluate_episode(episode,provider=None,persist=False)['render_hash']


def test_keep_work_cli_flag_forwarded(tmp_path,monkeypatch,capsys):
    from pipelines.ffmpeg.cli import _run_main
    captured={}
    def fake(*args,**kwargs):
        captured.update(kwargs);raise StageError('TEST','refinery','Offline dispatch sentinel')
    monkeypatch.setattr('pipelines.ffmpeg.cli.run_episode',fake)
    assert _run_main(['--footage-dir',str(tmp_path),'--goal','teaser','--keep-work'])!=0
    assert captured['keep_work'] is True
