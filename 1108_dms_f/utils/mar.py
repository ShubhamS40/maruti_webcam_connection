"""
Mouth Aspect Ratio (MAR) utilities with tuple-safe landmark handling.
"""

import math

from utils.ear import _as_xy, euclidean


def compute_MAR(points, width=None, height=None):
    """Compute mouth aspect ratio from 6 or 8 mouth landmark points."""
    if points is None:
        return 0.0

    try:
        converted = [_as_xy(point, width, height) for point in points]

        if len(converted) >= 8:
            left, right = converted[0], converted[1]
            vertical_pairs = (
                (converted[2], converted[3]),
                (converted[4], converted[5]),
                (converted[6], converted[7]),
            )
        elif len(converted) >= 6:
            left, top1, top2, right, bottom2, bottom1 = converted[:6]
            vertical_pairs = ((top1, bottom1), (top2, bottom2))
        else:
            return 0.0

        horizontal = euclidean(left, right)
        if horizontal <= 1e-6:
            return 0.0

        vertical = sum(euclidean(top, bottom) for top, bottom in vertical_pairs)
        vertical /= len(vertical_pairs)
        return vertical / horizontal
    except (IndexError, TypeError, ValueError):
        return 0.0


def compute_mar(points, width=None, height=None):
    return compute_MAR(points, width, height)


def compute_mouth_area(points, width=None, height=None):
    """Compute mouth opening area = Mouth Width x Mouth Height from landmarks."""
    if points is None:
        return 0.0, 0.0, 0.0

    try:
        converted = [_as_xy(point, width, height) for point in points]

        if len(converted) >= 8:
            left, right = converted[0], converted[1]
            vertical_pairs = (
                (converted[2], converted[3]),
                (converted[4], converted[5]),
                (converted[6], converted[7]),
            )
        elif len(converted) >= 6:
            left, top1, top2, right, bottom2, bottom1 = converted[:6]
            vertical_pairs = ((top1, bottom1), (top2, bottom2))
        else:
            return 0.0, 0.0, 0.0

        horizontal = euclidean(left, right)
        vertical = sum(euclidean(top, bottom) for top, bottom in vertical_pairs) / len(vertical_pairs)
        area = horizontal * vertical
        return area, horizontal, vertical
    except (IndexError, TypeError, ValueError):
        return 0.0, 0.0, 0.0

