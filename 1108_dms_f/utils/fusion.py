"""Risk fusion — eye closure and PERCLOS weighted heavily.

Eye risk is relative to the personalized close threshold (not a universal EAR cut),
so small-eye attentive drivers are not automatically high-risk.
"""


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
    # Floor must stay below small-eye personal thresholds (was 0.20 → false drowsiness).
    thr = max(float(ear_threshold), 0.05)
    eye_open = clamp(ear / thr)
    # Soften mild dips below threshold; temporal eyes_closed + PERCLOS carry hard evidence.
    eye_risk = clamp(1.0 - eye_open)
    if not eyes_closed:
        eye_risk *= 0.55

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


