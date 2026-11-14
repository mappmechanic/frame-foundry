"""Mock tests prove bounds/contracts only, never detector accuracy."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError
from PIL import Image, ImageChops, ImageStat

from pipelines.ffmpeg.contracts import EpisodeContext, StageError
from pipelines.ffmpeg.vision import analyze_visual_cues


class Gray:
    def __init__(self, image):
        self.image = image
        self.shape = (image.height, image.width)

    def mean(self):
        return ImageStat.Stat(self.image).mean[0]


@pytest.fixture
def fake_cv2(monkeypatch):
    calls = SimpleNamespace(faces=[], people=[], threads=[], opencl=[])

    class CVError(Exception):
        pass

    class Face:
        def empty(self):
            return False

        def detectMultiScale(self, gray, **kwargs):
            calls.faces.append(gray.shape)
            return [(1, 2, 3, 4)]  # Fabricated output; not accuracy evidence.

    class HOG:
        def setSVMDetector(self, weights):
            pass

        def detectMultiScale(self, pixels, **kwargs):
            calls.people.append(pixels.size)
            return [], []

    cv = SimpleNamespace(
        error=CVError, data=SimpleNamespace(haarcascades='/bundled/'),
        CascadeClassifier=lambda path: Face(), HOGDescriptor=HOG,
        HOGDescriptor_getDefaultPeopleDetector=lambda: [1],
        setNumThreads=calls.threads.append,
        ocl=SimpleNamespace(setUseOpenCL=calls.opencl.append),
        COLOR_RGB2GRAY=1, COLOR_RGB2BGR=2,
        cvtColor=lambda image, code: Gray(image.convert('L')) if code == 1 else image,
        resize=lambda gray, size: Gray(gray.image.resize(size)),
        absdiff=lambda a, b: Gray(ImageChops.difference(a.image, b.image)))
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    monkeypatch.setitem(sys.modules, 'numpy', SimpleNamespace(asarray=lambda image: image.copy()))
    return cv, calls


def source(tmp_path, colors=(0, 128, 255), size=(640, 360), sid='source-0001'):
    frames = []
    for i, color in enumerate(colors):
        path = Path('frames') / sid / f'{i}.jpg'
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', size, (color, color, color)).save(tmp_path / path)
        frames.append(dict(path=str(path), source_time_ms=100 + 1000*i,
                           proxy_time_ms=1000*i))
    return dict(source_id=sid, duration_ms=10000, keyframes=frames)


def test_bounded_sampling_source_time_and_sparse_measured_motion(tmp_path, fake_cv2):
    s = source(tmp_path, colors=(0, 30, 128, 200, 255), size=(900, 1600))
    cues = analyze_visual_cues([s], tmp_path)[s['source_id']]
    assert len(cues) == 3
    assert [c['time_ms'] for c in cues] == [100, 2100, 4100]
    assert [c['frame_path'] for c in cues] == [s['keyframes'][i]['path'] for i in (0, 2, 4)]
    assert [c['face_count'] for c in cues] == [1, 1, 1]
    assert [c['person_count'] for c in cues] == [0, 0, 0]
    assert cues[0]['motion'] is None
    assert math.isclose(cues[1]['motion'], 128/255)
    assert math.isclose(cues[2]['motion'], 127/255)
    assert all('camera_lighting_subject_confounded' in c['method'] for c in cues)
    _, calls = fake_cv2
    assert calls.threads == [1] and calls.opencl == [False]
    assert len(calls.faces) == len(calls.people) == 3
    assert all(max(shape) <= 320 for shape in calls.faces + calls.people)


def test_missing_opencv_is_unavailable_not_zero(tmp_path, monkeypatch):
    s = source(tmp_path)
    monkeypatch.setitem(sys.modules, 'cv2', None)
    cues = analyze_visual_cues([s], tmp_path)[s['source_id']]
    assert len(cues) == 3
    assert all(c['method'] == 'unavailable:opencv_missing' for c in cues)
    assert all(c[k] is None for c in cues for k in ('face_count', 'person_count', 'motion'))


def test_decode_failure_resets_motion_and_keeps_provenance(tmp_path, fake_cv2):
    s = source(tmp_path)
    (tmp_path / s['keyframes'][1]['path']).write_text('corrupt')
    cues = analyze_visual_cues([s], tmp_path)[s['source_id']]
    assert cues[1]['method'] == 'unavailable:frame_decode_failed'
    assert cues[1]['time_ms'] == 1100 and cues[1]['face_count'] is None
    assert cues[2]['motion'] is None


def test_detection_error_not_measured_zero(tmp_path, fake_cv2):
    cv, _ = fake_cv2
    s = source(tmp_path)

    def fail(*args, **kwargs):
        raise cv.error('broken detector')

    cv.CascadeClassifier = lambda path: SimpleNamespace(empty=lambda: False, detectMultiScale=fail)
    cues = analyze_visual_cues([s], tmp_path)[s['source_id']]
    assert all(c['face_count'] is None and c['method'].startswith('unavailable:face_detection_failed') for c in cues)
    assert cues[1]['motion'] is not None  # Independent measured motion is retained.


def test_missing_bundled_asset_truthful(tmp_path, fake_cv2):
    cv, _ = fake_cv2
    cv.CascadeClassifier = lambda path: SimpleNamespace(empty=lambda: True)
    cues = analyze_visual_cues([source(tmp_path)], tmp_path)['source-0001']
    assert all(c['method'] == 'unavailable:haar_asset_missing' for c in cues)
    assert all(c['person_count'] is None for c in cues)


def test_tiny_image_does_not_call_hog_or_pretend_zero(tmp_path, fake_cv2):
    _, calls = fake_cv2
    cues = analyze_visual_cues([source(tmp_path, size=(32, 32))], tmp_path)['source-0001']
    assert not calls.people
    assert all(c['person_count'] is None and 'person_window_too_small' in c['method'] for c in cues)


@pytest.mark.parametrize('path,time_ms', [('../outside.jpg', 100), ('/tmp/outside.jpg', 100),
                                         ('frames/a.jpg', -1), ('frames/a.jpg', 10000)])
def test_invalid_provenance_is_typed(tmp_path, path, time_ms):
    s = dict(source_id='s', duration_ms=10000,
             keyframes=[dict(path=path, source_time_ms=time_ms)])
    with pytest.raises(StageError) as error:
        analyze_visual_cues([s], tmp_path)
    assert error.value.code == 'VISUAL_PROVENANCE_INVALID'


def test_motion_does_not_cross_sources_or_duplicate_times(tmp_path, fake_cv2):
    s = source(tmp_path, sid='one')
    s['keyframes'][1]['source_time_ms'] = 100
    other = source(tmp_path, sid='two')
    cues = analyze_visual_cues([s, other], tmp_path)
    assert cues['one'][0]['motion'] is cues['one'][1]['motion'] is cues['two'][0]['motion'] is None
    assert cues['two'][1]['motion'] is not None


def test_visual_schema_bounds_and_availability(tmp_path, fake_cv2):
    path = Path(__file__).resolve().parents[1] / 'schemas/episode-artifacts.schema.json'
    schema = json.loads(path.read_text())['$defs']['analysis']['properties']['visual_cues']
    validator = Draft202012Validator(schema)
    cues = analyze_visual_cues([source(tmp_path)], tmp_path)
    validator.validate(cues)
    cues['source-0001'][1]['motion'] = 1.1
    with pytest.raises(ValidationError):
        validator.validate(cues)
    cues['source-0001'][1]['motion'] = .5
    cues['source-0001'][1]['face_count'] = None
    with pytest.raises(ValidationError):
        validator.validate(cues)
    cues['source-0001'][1]['method'] = 'unavailable:face_detection_failed'
    validator.validate(cues)


def test_refinery_persists_visual_cues_before_candidates(tmp_path, fake_cv2, monkeypatch):
    from pipelines.ffmpeg import refinery, candidates
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'test.mp4').touch()
    s = source(tmp_path, sid='source-0001')
    monkeypatch.setattr(refinery, 'probe', lambda path: dict(
        video=dict(avg_frame_rate='30/1', width=640, height=360), duration_ms=10000,
        rotation=0, data=dict(streams=[], format={})))
    monkeypatch.setattr(refinery, 'create_proxy', lambda *args: dict(proxy_duration_ms=10000))
    monkeypatch.setattr(refinery, 'sample_keyframes', lambda *args: [
        dict(f, path=str(tmp_path / f['path'])) for f in s['keyframes']])
    monkeypatch.setattr(refinery, 'run', lambda *args: SimpleNamespace(stderr=''))
    seen = []
    monkeypatch.setattr(candidates, 'extract_candidates', lambda analysis, root: seen.append(analysis['visual_cues']) or [])
    analysis = refinery.run_refinery(EpisodeContext('visual-test', tmp_path, 'teaser'), inputs)
    assert seen == [analysis['visual_cues']]
    assert json.loads((tmp_path / 'analysis.json').read_text())['visual_cues'] == analysis['visual_cues']


def test_optional_genuine_pixels_measure_difference_not_detector_accuracy(tmp_path):
    cv = pytest.importorskip('cv2')
    pytest.importorskip('numpy')
    cues = analyze_visual_cues([source(tmp_path, colors=(0, 255, 255))], tmp_path)['source-0001']
    assert cv.getNumThreads() == 1 and not cv.ocl.useOpenCL()
    assert all(not c['method'].startswith('unavailable:') for c in cues)
    assert cues[0]['motion'] is None
    assert cues[1]['motion'] == pytest.approx(1.0)
    assert cues[2]['motion'] == pytest.approx(0.0)
    assert all(isinstance(c['face_count'], int) and isinstance(c['person_count'], int) for c in cues)
