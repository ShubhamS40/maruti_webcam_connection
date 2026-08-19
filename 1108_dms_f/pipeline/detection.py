"""
Face detection pipeline: MediaPipe Face Mesh and landmark extraction.
"""

import cv2
import mediapipe as mp

from config import (
    FACE_MESH_DETECTION_CONF,
    FACE_MESH_MAX_FACES,
    FACE_MESH_TRACKING_CONF,
)


class FaceDetector:
    """Wrap MediaPipe Face Mesh and normalize landmarks to pixel tuples."""

    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=FACE_MESH_MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=FACE_MESH_DETECTION_CONF,
            min_tracking_confidence=FACE_MESH_TRACKING_CONF,
        )

    def process(self, frame_bgr):
        """Return pixel landmarks list or None."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        mesh = results.multi_face_landmarks[0]
        landmarks = []
        for lm in mesh.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))
        return landmarks

    def close(self):
        self._mesh.close()


def crop_face(frame, landmarks, padding=20):
    """Bounding box crop around the face mesh."""
    if not landmarks:
        return None, (0, 0, 0, 0)

    h, w = frame.shape[:2]
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]

    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)
    x2 = min(max(xs) + padding, w)
    y2 = min(max(ys) + padding, h)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None, (x1, y1, x2, y2)
    return face, (x1, y1, x2, y2)
