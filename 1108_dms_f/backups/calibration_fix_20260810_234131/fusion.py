"""Risk fusion — eye closure and PERCLOS weighted heavily."""


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def compute_raw_risk(
    ear,
    ear_threshold,
    perclos,
    cnn_score,
    eyes_closed=False,
    eye_closed_sec=0.0,
    head_down=False,
    yawn_fatigue=False,
    yawn_bonus=0.0,
):
    thr = max(ear_threshold, 0.20)
    eye_open = clamp(ear / thr)
    eye_risk = 1.0 - eye_open

    # If eyes are closed, scale risk from 0.50 to 1.00 based on closure duration
    if eyes_closed:
        eye_closed_risk = 0.50 + clamp(eye_closed_sec / 2.0) * 0.50
        eye_risk = max(eye_risk, eye_closed_risk)

    perclos_risk = clamp(perclos)
    cnn_risk = clamp(cnn_score)
    
    # Static additions for dangerous behavioral states (clamped in final sum)
    head_risk = 0.40 if head_down else 0.0
    yawn_risk = 0.20 if yawn_fatigue else 0.0

    raw = (
        eye_risk * 0.40
        + perclos_risk * 0.30
        + cnn_risk * 0.15
        + head_risk
        + yawn_risk
        + float(yawn_bonus)
    )
    return clamp(raw)


