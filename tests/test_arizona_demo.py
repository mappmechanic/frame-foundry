"""Real private sample integration is opt-in; never fetch media during unit CI."""
import os
from pathlib import Path
import pytest
from pipelines.ffmpeg.cli import run_episode
from pipelines.ffmpeg.contracts import read_json
from tests.fixtures.generate_footage import generate_footage

GOAL = 'Create a teaser from this raw family-trip footage that highlights the important moments of the trip'


def test_explicit_visual_only_is_labeled(tmp_path):
    footage=tmp_path/'footage'; generate_footage(footage)
    report=run_episode(footage,'10 second family-trip teaser',episodes_root=tmp_path/'episodes',episode_id='visual',provider=None,visual_only=True)
    assert report['passed'] and not report['fixture_mode']
    assert any('Speech' in x for x in report['coverage']['limitations'])
    assert read_json(tmp_path/'episodes'/'visual'/'analysis.json')['transcription_status']=='visual_only_unavailable'


@pytest.mark.skipif(not os.environ.get('FRAMEFOUNDRY_ARIZONA_FOOTAGE'), reason='Set FRAMEFOUNDRY_ARIZONA_FOOTAGE to downloaded private release clips')
def test_real_arizona_pipeline(tmp_path):
    footage=Path(os.environ['FRAMEFOUNDRY_ARIZONA_FOOTAGE'])
    assert len(list(footage.glob('*.mp4')))==26
    report=run_episode(footage,GOAL,episodes_root=tmp_path,episode_id='arizona',visual_only=os.environ.get('FRAMEFOUNDRY_VISUAL_ONLY')=='1')
    assert report['passed'], report
    assert -15 <= report['integrated_lufs'] <= -13 and report['true_peak_dbtp'] <= -1
    assert (tmp_path/'arizona'/'contact-sheet.jpg').stat().st_size > 0
    print({'qa':report,'footage_mode':'visual-only' if os.environ.get('FRAMEFOUNDRY_VISUAL_ONLY')=='1' else 'local-asr'})
