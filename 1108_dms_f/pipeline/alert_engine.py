"""
Alert engine — audio/UI alerts ONLY for DROWSY / MICROSLEEP after consecutive persistence.

Eval window (Eval Sec trackbar) gates initial fatigue evidence before state machine
can accumulate tier counters.
"""

import threading
import time

try:
    import winsound

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from pipeline.temporal import ConsecutiveCounter


class AlertEngine:
    def __init__(self):
        self.eval_counter = ConsecutiveCounter(1)
        self.alert_counter = ConsecutiveCounter(1)
        self._alarm_playing = False
        self._last_alarm_time = 0.0
        self.eval_satisfied = False
        self.alert_active = False

    def _play_alarm(self, duration_ms):
        if self._alarm_playing:
            return
        self._alarm_playing = True
        try:
            if AUDIO_AVAILABLE:
                winsound.Beep(2500, int(duration_ms))
        except Exception:
            pass
        finally:
            self._alarm_playing = False

    def update_eval(self, fatigue_evidence, fps, eval_seconds):
        """
        Fatigue evidence must be continuously true for eval_seconds (consecutive).
        Resets immediately when evidence stops — fixes sticky-eval bug.
        """
        required = max(1, int(eval_seconds * fps))
        self.eval_counter.set_threshold(required)
        self.eval_satisfied = self.eval_counter.update(fatigue_evidence)
        return self.eval_satisfied

    def update_alert(self, state_alert_eligible, fps, alert_seconds, alarm_duration_sec=3.0, cooldown_sec=5.0):
        """
        Alert only when state is DROWSY/MICROSLEEP AND remains so for alert_seconds.
        """
        if not state_alert_eligible:
            self.alert_counter.reset()
            self.alert_active = False
            return False

        required = max(1, int(alert_seconds * fps))
        self.alert_counter.set_threshold(required)
        self.alert_active = self.alert_counter.update(True)

        now = time.time()
        if self.alert_active and (now - self._last_alarm_time) >= cooldown_sec:
            self._last_alarm_time = now
            duration_ms = int(alarm_duration_sec * 1000)
            threading.Thread(target=self._play_alarm, args=(duration_ms,), daemon=True).start()

        return self.alert_active

