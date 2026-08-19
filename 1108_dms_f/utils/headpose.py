"""
Head pose estimation — geometric pitch (stable) + solvePnP yaw/roll.

RQDecomp*360 produced values like -54839°; pitch for head-down uses
nose–chin–eye geometry instead (degrees in a sane ±90 range).
"""

import math

import cv2
import numpy as np

from utils.ear import _as_xy


def _to_pixels(point, width, height):
    x, y = _as_xy(point, width, height)
    return float(x), float(y)


def _clamp_angle(deg, limit=89.0):
    return max(-limit, min(limit, float(deg)))


def _euler_from_rotation_matrix(R):
    """ZYX Euler angles in degrees from a 3x3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    sy = math.hypot(R[0, 0], R[1, 0])
    if sy < 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(-R[1, 2], R[1, 1])
        roll = 0.0
    else:
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    return (
        _clamp_angle(math.degrees(pitch)),
        _clamp_angle(math.degrees(yaw)),
        _clamp_angle(math.degrees(roll)),
    )


def geometric_pitch_deg(landmarks, width, height):
    """
    Pitch proxy from facial geometry (image coordinates).
    Looking down → chin moves toward nose → angle decreases vs neutral.
    """
    nose = _to_pixels(landmarks[1], width, height)
    chin = _to_pixels(landmarks[152], width, height)
    le = _to_pixels(landmarks[33], width, height)
    re = _to_pixels(landmarks[263], width, height)

    eye_y = (le[1] + re[1]) / 2.0
    face_h = chin[1] - eye_y
    if face_h < 8:
        return 0.0

    nose_chin_dy = chin[1] - nose[1]
    ratio = nose_chin_dy / face_h
    angle = math.degrees(math.atan2(ratio, 0.55))
    return _clamp_angle(angle)


def head_pose(landmarks, width, height):
    """Return (pitch, yaw, roll). Calculated geometrically for maximum automotive stability."""
    if landmarks is None or len(landmarks) < 292 or width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0

    # 1. Stable Geometric Pitch
    pitch = geometric_pitch_deg(landmarks, width, height)

    # 2. Stable Geometric Roll (eye corner slope)
    le = _to_pixels(landmarks[33], width, height)
    re = _to_pixels(landmarks[263], width, height)
    dx = abs(le[0] - re[0])
    if dx < 1e-5:
        dx = 1e-5
    dy = le[1] - re[1]
    roll = _clamp_angle(math.degrees(math.atan2(dy, dx)))


    # 3. Stable Geometric Yaw (nose-to-eye ratios)
    nose = _to_pixels(landmarks[1], width, height)
    d_left = math.hypot(nose[0] - le[0], nose[1] - le[1])
    d_right = math.hypot(nose[0] - re[0], nose[1] - re[1])
    if (d_left + d_right) > 1e-5:
        # Ratio scaled to degree range
        raw_yaw = ((d_right - d_left) / (d_left + d_right)) * 95.0
        yaw = _clamp_angle(raw_yaw)
    else:
        yaw = 0.0

    return pitch, yaw, roll



def estimate_head_pose(landmarks, width, height):
    return head_pose(landmarks, width, height)
