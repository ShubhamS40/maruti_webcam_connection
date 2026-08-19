import cv2
import numpy as np


def adjust_brightness(frame):
    """Apply gentle global brightness stabilization without changing frame shape."""
    if frame is None or frame.size == 0:
        return frame

    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_v = float(np.mean(v))

        if mean_v < 85:
            gain = min(1.35, 105.0 / max(mean_v, 1.0))
            v = np.clip(v.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        elif mean_v > 205:
            gain = max(0.80, 190.0 / mean_v)
            v = np.clip(v.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    except cv2.error:
        return frame


def enhance_brightness(face_crop, brightness):

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY
    )

    if brightness < 40:

        clip = 5.0

    elif brightness < 70:

        clip = 4.0

    else:

        clip = 2.0

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR
    )

    return enhanced
