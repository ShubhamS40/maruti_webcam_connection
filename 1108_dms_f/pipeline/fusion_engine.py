from config import RISK_EMA_ALPHA
from pipeline.temporal import ExponentialSmoother
from utils.fusion import compute_raw_risk


class FusionEngine:
    def __init__(self, alpha=RISK_EMA_ALPHA):
        self._ema = ExponentialSmoother(alpha=alpha)
        self.raw_risk = 0.0
        self.smooth_risk = 0.0

    def update(
        self,
        ear,
        ear_threshold,
        perclos,
        cnn_score,
        eyes_closed,
        eye_closed_sec,
        head_down,
        yawn_fatigue,
        yawn_bonus=0.0,
    ):
        self.raw_risk = compute_raw_risk(
            ear=ear,
            ear_threshold=ear_threshold,
            perclos=perclos,
            cnn_score=cnn_score,
            eyes_closed=eyes_closed,
            eye_closed_sec=eye_closed_sec,
            head_down=head_down,
            yawn_fatigue=yawn_fatigue,
            yawn_bonus=yawn_bonus,
        )
        self.smooth_risk = max(0.0, min(1.0, self._ema.update(self.raw_risk)))
        return self.smooth_risk


