"""
Eye Aspect Ratio (EAR) utilities — tuple-safe landmarks.
"""

import math


def _as_xy(point, width=None, height=None):
    if point is None:
        raise ValueError("point is None")
    if hasattr(point, "x") and hasattr(point, "y"):
        x, y = float(point.x), float(point.y)
    else:
        x, y = float(point[0]), float(point[1])
    if width is not None and height is not None:
        if abs(x) <= 1.5 and abs(y) <= 1.5:
            return x * width, y * height
    return x, y


def euclidean(p1, p2, width=None, height=None):
    x1, y1 = _as_xy(p1, width, height)
    x2, y2 = _as_xy(p2, width, height)
    return math.hypot(x1 - x2, y1 - y2)


def compute_EAR(eye_points, width=None, height=None, eye_indices=None):
    if eye_points is None:
        return 0.0
    try:
        if eye_indices is not None and len(eye_indices) == 6:
            points = [_as_xy(eye_points[i], width, height) for i in eye_indices]
        else:
            points = list(eye_points)
            if len(points) != 6:
                return 0.0
            points = [_as_xy(p, width, height) for p in points]
        p1, p2, p3, p4, p5, p6 = points
        vertical1 = euclidean(p2, p6)
        vertical2 = euclidean(p3, p5)
        horizontal = euclidean(p1, p4)
        if horizontal <= 1e-6:
            return 0.0
        return (vertical1 + vertical2) / (2.0 * horizontal)
    except (IndexError, TypeError, ValueError, KeyError):
        return 0.0


def compute_ear(eye_points, width=None, height=None, eye_indices=None):
    return compute_EAR(eye_points, width, height, eye_indices)


def dynamic_ear_threshold(baseline_ear, ratio=0.72, floor=0.08):
    if baseline_ear is None or baseline_ear <= 0:
        return floor
    return max(floor, float(baseline_ear) * ratio)
