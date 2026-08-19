"""
Driver state machine with high-accuracy temporal-behavioral filtering.

  ATTENTIVE  — eyes open, head up, not yawning, low risk
  FATIGUED   — yawning frequently / head postures (turns/tilts) active
  DROWSY     — eyes closed >= 3.0s OR big yawning active
  MICROSLEEP — eyes closed >= 5.0s OR head down active

Strictly blocks DROWSY / MICROSLEEP when eyes/face are not detected.
"""

from config import (
    ATTENTION_INITIAL,
    EYE_CLOSED_SEC_DROWSY,
    EYE_CLOSED_SEC_MICROSLEEP,
    STATE_ENTER_DROWSY_FRAMES,
    STATE_ENTER_FATIGUED_FRAMES,
    STATE_ENTER_MICROSLEEP_FRAMES,
    STATE_EXIT_FRAMES,
)
from pipeline.temporal import ConsecutiveCounter
from utils.status import color_for_state


class AttentionEngine:
    def __init__(self, initial=ATTENTION_INITIAL):
        self.score = float(initial)

    def update(self, smooth_risk, eyes_closed, head_down, any_tilt):
        # Attention decays based on active risk, eyes closed, head down, and other tilts
        if smooth_risk > 0.15 or eyes_closed or head_down or any_tilt:
            decay = (
                (smooth_risk * 1.5) +
                (3.0 if eyes_closed else 0.0) +
                (2.0 if head_down else 0.0) +
                (1.0 if any_tilt else 0.0)
            )
            self.score = max(0.0, self.score - decay)
        else:
            # Recover gradually and smoothly when alert/attentive (0.6% per frame = ~12% per second at 20 FPS)
            self.score = min(100.0, self.score + 0.6)
        return self.score


class DriverStateMachine:
    def __init__(self):
        self.state = "ATTENTIVE"
        self.state_color = color_for_state("ATTENTIVE")
        self.attention = AttentionEngine()
        
        # State entrance persistence counters
        self._micro = ConsecutiveCounter(STATE_ENTER_MICROSLEEP_FRAMES)
        self._drowsy = ConsecutiveCounter(STATE_ENTER_DROWSY_FRAMES)
        self._fatigue = ConsecutiveCounter(STATE_ENTER_FATIGUED_FRAMES)
        
        # State exit/downgrade hysteresis counter
        self._exit_counter = ConsecutiveCounter(STATE_EXIT_FRAMES)
        
        self.debug = {}

    def update(self, signals, system_calibrated=True):
        if not system_calibrated:
            self.state = "CALIBRATING"
            self.state_color = color_for_state("CALIBRATING")
            return self.state

        face_lost = signals.get("face_lost", False)
        eyes_closed = signals.get("eyes_closed", False)
        t_closed = signals.get("eye_closed_sec", 0.0)
        head_down = signals.get("head_down", False)
        any_tilt = signals.get("any_tilt", False)
        yawn_fatigue = signals.get("yawn_fatigue", False)
        frequent_yawning = signals.get("frequent_yawning", False)
        repeated_yawning = signals.get("repeated_yawning", False)
        big_yawning = signals.get("big_yawning", False)
        smooth = signals.get("smooth_risk", 0.0)

        # Update attention based on current conditions
        self.attention.update(smooth, eyes_closed, head_down, any_tilt)

        # GATING: If face/eyes are not detected, block DROWSY & MICROSLEEP states completely
        if face_lost:
            micro_cond = False
            drowsy_cond = False
            # Can stay in FATIGUED if the system is still recovering, or transition back to ATTENTIVE
            fatigue_cond = False
        else:
            # 1. MICROSLEEP: Eyes closed for a long time (>= 5.0s) OR Head Down active
            micro_cond = (eyes_closed and t_closed >= EYE_CLOSED_SEC_MICROSLEEP) or head_down
            
            # 2. DROWSY: Eyes closed for at least 3.0s OR sustained high fatigue risk (>= 0.45)
            drowsy_cond = (eyes_closed and t_closed >= EYE_CLOSED_SEC_DROWSY) or (smooth >= 0.45)
            
            # 3. FATIGUED: Frequent/repeated yawning OR posture tilt active OR brief eye closure OR moderate risk
            fatigue_cond = (
                yawn_fatigue or
                frequent_yawning or
                repeated_yawning or
                any_tilt or
                (eyes_closed and t_closed > 0.25) or
                smooth >= 0.15
            )


        # Update entrance counters
        micro_ok = self._micro.update(micro_cond)
        drowsy_ok = self._drowsy.update(drowsy_cond)
        fatigue_ok = self._fatigue.update(fatigue_cond)

        # State Hierarchy: ATTENTIVE < FATIGUED < DROWSY < MICROSLEEP
        state_ranks = {"ATTENTIVE": 1, "FATIGUED": 2, "DROWSY": 3, "MICROSLEEP": 4}
        current_rank = state_ranks.get(self.state, 1)

        # 1. UPGRADE TRANSITIONS: Can jump to a higher state if its persistence counter is satisfied
        if micro_ok and current_rank < 4:
            self.state = "MICROSLEEP"
            self._exit_counter.reset()
        elif drowsy_ok and current_rank < 3:
            self.state = "DROWSY"
            self._exit_counter.reset()
        elif fatigue_ok and current_rank < 2:
            self.state = "FATIGUED"
            self._exit_counter.reset()

        # 2. DOWNGRADE HYSTERESIS: Can only drop to a lower state if current active condition is lost for STATE_EXIT_FRAMES
        else:
            condition_active = False
            if self.state == "MICROSLEEP":
                condition_active = micro_cond
            elif self.state == "DROWSY":
                condition_active = drowsy_cond
            elif self.state == "FATIGUED":
                condition_active = fatigue_cond
            elif self.state == "ATTENTIVE":
                condition_active = True

            if not condition_active:
                if self._exit_counter.update(True):
                    # Exit timer satisfied! Downgrade to the highest active condition
                    if drowsy_cond:
                        self.state = "DROWSY"
                    elif fatigue_cond:
                        self.state = "FATIGUED"
                    else:
                        self.state = "ATTENTIVE"
                    self._exit_counter.reset()
            else:
                self._exit_counter.reset()

        self.state_color = color_for_state(self.state)
        
        self.debug = {
            "micro_ok": micro_ok,
            "drowsy_ok": drowsy_ok,
            "fatigue_ok": fatigue_ok,
            "micro_frames": self._micro.count,
            "drowsy_frames": self._drowsy.count,
            "fatigue_frames": self._fatigue.count,
            "exit_frames": self._exit_counter.count,
        }
        return self.state

    @property
    def alert_eligible(self):
        return self.state in ("DROWSY", "MICROSLEEP")
