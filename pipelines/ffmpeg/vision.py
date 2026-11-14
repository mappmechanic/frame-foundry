"""Bounded, offline CPU cues from existing shared-origin sampled keyframes.

Counts are detector outputs, not reliable subject counts or semantic recognition.
Motion is sparse mean absolute grayscale difference (0..1), confounded by camera
movement, lighting, and subject movement; it is not optical flow or action speed.
Null means unavailable, never a measured zero. Consumers should ignore unavailable
methods and handle the first frame's null motion (no preceding measurement).
"""
from pathlib import Path

from PIL import Image

from .contracts import StageError

MAX_FRAMES = 3
MAX_SIDE = 320
METHOD = ('opencv_cpu_haar_face_hog_person;'
          'sparse_gray_difference_camera_lighting_subject_confounded')


def _selected_frames(source, root):
    """Pick temporal first/middle/last, preserving original source timestamps."""
    frames = sorted(source['keyframes'], key=lambda f: f['source_time_ms'])
    if len(frames) > MAX_FRAMES:
        frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    selected = []
    seen = set()
    for frame in frames:
        path = Path(frame['path'])
        time_ms = frame['source_time_ms']
        if (path.is_absolute() or not (root / path).resolve().is_relative_to(root)
                or type(time_ms) is not int or not 0 <= time_ms < source['duration_ms']):
            raise StageError('VISUAL_PROVENANCE_INVALID', 'refinery',
                             'Visual keyframe must be episode-relative with source time',
                             source['source_id'])
        key = (str(path), time_ms)
        if key not in seen:
            selected.append(dict(frame_path=str(path), time_ms=time_ms))
            seen.add(key)
    return selected


def _unavailable(frame, reason):
    return dict(frame, face_count=None, person_count=None, motion=None,
                method=f'unavailable:{reason}')


def analyze_visual_cues(sources, episode_dir):
    """Return SOURCE_ID -> <=3 cue records; never read videos or fetch assets.

    Uses bundled OpenCV Haar/HOG assets with one CPU thread and OpenCL disabled.
    Source times are copied, not recomputed from proxy/container timestamps.
    Missing OpenCV/assets/images produce explicit unavailable records. Invalid
    provenance raises typed StageError rather than silently reading outside episode.
    """
    root = Path(episode_dir).resolve()
    selected = {s['source_id']: _selected_frames(s, root) for s in sources}
    try:
        import cv2
        import numpy as np
    except (ImportError, OSError):
        return {sid: [_unavailable(f, 'opencv_missing') for f in frames]
                for sid, frames in selected.items()}
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        face = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) /
                                        'haarcascade_frontalface_default.xml'))
        if face.empty():
            return {sid: [_unavailable(f, 'haar_asset_missing') for f in frames]
                    for sid, frames in selected.items()}
        person = cv2.HOGDescriptor()
        person.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except (cv2.error, OSError):
        return {sid: [_unavailable(f, 'detector_initialization_failed') for f in frames]
                for sid, frames in selected.items()}
    cues = {}
    for sid, frames in selected.items():
        records = cues[sid] = []
        previous = None
        previous_time = None
        for frame in frames:
            try:
                # Downscale before either detector, without enlarging small frames.
                with Image.open(root / frame['frame_path']) as image:
                    image = image.convert('RGB')
                    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
                    pixels = np.asarray(image)
                gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            except (OSError, ValueError, cv2.error):
                records.append(_unavailable(frame, 'frame_decode_failed'))
                previous = previous_time = None
                continue
            record = dict(frame, face_count=None, person_count=None, motion=None, method=METHOD)
            reasons = []
            try:
                record['face_count'] = len(face.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))
            except cv2.error:
                reasons.append('face_detection_failed')
            # OpenCV HOG requires its default 64x128 window to fit the image.
            if gray.shape[0] < 128 or gray.shape[1] < 64:
                reasons.append('person_window_too_small')
            else:
                try:
                    boxes, _ = person.detectMultiScale(
                        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                        winStride=(8, 8), padding=(8, 8), scale=1.05)
                    record['person_count'] = len(boxes)
                except cv2.error:
                    reasons.append('person_detection_failed')
            if previous is not None and frame['time_ms'] > previous_time:
                try:
                    aligned = cv2.resize(gray, (previous.shape[1], previous.shape[0]))
                    record['motion'] = min(1.0, max(0.0, float(
                        cv2.absdiff(previous, aligned).mean()) / 255.0))
                except cv2.error:
                    reasons.append('motion_measurement_failed')
            else:
                record['method'] += ';motion_unavailable:no_previous_distinct_time'
            if reasons:
                record['method'] = 'unavailable:' + ','.join(reasons) + ';' + record['method']
            records.append(record)
            previous, previous_time = gray, frame['time_ms']
    return cues
