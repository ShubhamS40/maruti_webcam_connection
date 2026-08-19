"""
Alert engine — audio/UI alerts ONLY for DROWSY / MICROSLEEP after consecutive persistence.

Eval window (Eval Sec trackbar) gates initial fatigue evidence before state machine
can accumulate tier counters.
"""

import os
import sys
import threading
import time
import subprocess

import numpy as np

AUDIO_AVAILABLE = False
AUDIO_BACKEND = "none"

try:
    import winsound
    AUDIO_AVAILABLE = True
    AUDIO_BACKEND = "winsound"
except ImportError:
    pass

try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
    AUDIO_BACKEND = "pygame"
except Exception:
    pass

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
            if AUDIO_BACKEND == "winsound":
                winsound.Beep(2500, int(duration_ms))
            elif AUDIO_BACKEND == "pygame":
                try:
                    freq_hz = 2500
                    duration_s = duration_ms / 1000.0
                    sample_rate = 44100
                    n_samples = int(sample_rate * duration_s)
                    t = np.linspace(0, duration_s, n_samples, False)
                    wave = 0.5 * np.sin(2 * np.pi * freq_hz * t)
                    audio_stereo = np.column_stack((wave, wave))
                    sound_array = (audio_stereo * 32767).astype(np.int16)
                    sound = pygame.sndarray.make_sound(sound_array)
                    sound.play()
                    time.sleep(duration_s)
                    sound.stop()
                except Exception:
                    pass
            elif sys.platform == "darwin":
                try:
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Glass.aiff"],
                        check=False, timeout=5,
                    )
                except Exception:
                    pass
            elif sys.platform.startswith("linux"):
                try:
                    subprocess.run(
                        ["play", "-n", "synth", f"{duration_ms/1000}", "sine", "2500"],
                        check=False, timeout=5,
                    )
                except Exception:
                    pass
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

