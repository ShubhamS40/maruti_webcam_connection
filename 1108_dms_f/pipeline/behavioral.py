"""
Behavioral analyzers — personalized eye closure, multi-posture head analysis, and temporal yawning.

Design principles:
- EAR and MAR are separate signals; eye closure NEVER triggers yawning.
- Absolute EAR/MAR alone do not classify drowsiness/yawns; personal baselines do.
- Hysteresis + temporal confirmation prevent flicker and sticky false positives.
- A yawn is a complete temporal EVENT (open → sustain ≥1.5s → close → reset), not a per-frame flag.
"""

import math
import time
from collections import deque
import numpy as np

from config import (
    BLINK_MAX_CLOSED_SEC,
    BLINK_MIN_CLOSED_FRAMES,
    EAR_BASELINE_EMA_ALPHA,
    EAR_BASELINE_MAX,
    EAR_BASELINE_MIN,
    EAR_CALIBRATION_FRAMES,
    EAR_OPEN_HYSTERESIS,
    EAR_SMOOTH_WINDOW,
    EAR_THRESHOLD_CEILING,
    EAR_THRESHOLD_FLOOR,
    EAR_THRESHOLD_RATIO,
    EYE_CLOSE_CONFIRM_SEC,
    HEAD_CALIBRATION_FRAMES,
    HEAD_DOWN_DELTA_DEG,
    HEAD_DOWN_PERSIST_FRAMES,
    MAR_BASELINE_DELTA_START,
    MAR_BASELINE_HYSTERESIS,
    MAR_BASELINE_MAX,
    MAR_BASELINE_MIN,
    MAR_CALIBRATION_FRAMES,
    MAR_PEAK_MIN_DELTA,
    MAR_TALKING_STD_THRESHOLD,
    MAR_YAWN_END_THRESHOLD,
    MAR_YAWN_END_MAX,
    MAR_YAWN_START_MAX,
    MAR_YAWN_START_THRESHOLD,
    PERCLOS_MIN_CLOSED_FRAMES,
    PERCLOS_WINDOW,
    STATE_EXIT_FRAMES,
    YAWN_BONUS_FREQUENT,
    YAWN_BONUS_REPEATED,
    YAWN_BONUS_SINGLE,
    YAWN_COOLDOWN_SEC,
    YAWN_DISPLAY_SEC,
    YAWN_END_HOLD_SEC,
    YAWN_FATIGUE_COUNT,
    YAWN_MIN_DURATION_SEC,
    YAWN_WINDOW_120S,
    YAWN_WINDOW_5S,
    YAWN_WINDOW_60S,
)
from pipeline.temporal import MovingAverage, ConsecutiveCounter
from utils.ear import compute_EAR
from utils.headpose import head_pose
from utils.mar import compute_MAR, compute_mouth_area


class EyeAnalyzer:
    """
    Personalized eye-state machine with open/close hysteresis and temporal confirmation.

    Why personal baseline:
      Attentive open-EAR varies widely (~0.18–0.35). Absolute EAR alone would falsely
      mark small-eye drivers as closed/drowsy.

    Why hysteresis (OPEN_THR > CLOSE_THR):
      Prevents OPEN↔CLOSED flicker on landmark noise.

    Why temporal confirmation:
      A single low-EAR frame is not EYES CLOSED; short closures remain blinks.
    """

    def __init__(self):
        self.calibrated = False
        self._calibration_samples = []
        self.baseline_ear = None
        self.ear_close_threshold = EAR_THRESHOLD_FLOOR
        self.ear_open_threshold = EAR_THRESHOLD_FLOOR + EAR_OPEN_HYSTERESIS
        # Keep legacy alias used by fusion / UI
        self.ear_threshold = self.ear_close_threshold
        self.raw_ear = 0.0
        self.smooth_ear_ma = MovingAverage(maxlen=EAR_SMOOTH_WINDOW)
        self._eyes_closed = False
        self._close_candidate_frames = 0
        self.closed_frames = 0
        self.blink_count = 0
        self.perclos_window = deque(maxlen=PERCLOS_WINDOW)
        self.eye_state = "OPEN"  # OPEN | PARTIAL | CLOSED

    def _update_threshold(self, smooth_ear):
        # Calibration: collect attentive open-eye samples only (upper EAR values).
        if not self.calibrated:
            self._calibration_samples.append(smooth_ear)
            if len(self._calibration_samples) >= EAR_CALIBRATION_FRAMES:
                arr = np.array(self._calibration_samples, dtype=float)
                # Use upper 60% of samples so blinks during calibration do not pull baseline down.
                cutoff = float(np.percentile(arr, 40))
                open_samples = arr[arr >= cutoff]
                if len(open_samples) < 5:
                    open_samples = arr
                med = float(np.median(open_samples))
                self.baseline_ear = float(np.clip(med, EAR_BASELINE_MIN, EAR_BASELINE_MAX))
                self.calibrated = True
        elif self.baseline_ear is not None and not self._eyes_closed:
            # Slow adaptive update only when clearly open — never learn closed eyes as normal.
            if smooth_ear > self.ear_open_threshold:
                alpha = EAR_BASELINE_EMA_ALPHA
                new_val = (1.0 - alpha) * self.baseline_ear + alpha * smooth_ear
                self.baseline_ear = float(np.clip(new_val, EAR_BASELINE_MIN, EAR_BASELINE_MAX))

        if self.baseline_ear is not None:
            close_thr = self.baseline_ear * EAR_THRESHOLD_RATIO
            close_thr = float(np.clip(close_thr, EAR_THRESHOLD_FLOOR, EAR_THRESHOLD_CEILING))
            open_thr = close_thr + EAR_OPEN_HYSTERESIS
            # Keep open threshold below baseline so a fully open eye always clears it.
            open_thr = min(open_thr, self.baseline_ear * 0.95)
            self.ear_close_threshold = close_thr
            self.ear_open_threshold = max(open_thr, close_thr + 0.01)
            self.ear_threshold = self.ear_close_threshold

    def _update_closed_state(self, smooth_ear, fps):
        fps_val = max(float(fps), 1.0)
        confirm_frames = max(1, int(math.ceil(EYE_CLOSE_CONFIRM_SEC * fps_val)))

        if self._eyes_closed:
            # Re-open only after clearing the higher open threshold (hysteresis).
            if smooth_ear >= self.ear_open_threshold:
                # Closing episode ended — classify blink vs sustained closure.
                if (
                    self.closed_frames >= BLINK_MIN_CLOSED_FRAMES
                    and (self.closed_frames / fps_val) <= BLINK_MAX_CLOSED_SEC
                ):
                    self.blink_count += 1
                self._eyes_closed = False
                self.closed_frames = 0
                self._close_candidate_frames = 0
            else:
                self.closed_frames += 1
        else:
            if smooth_ear < self.ear_close_threshold:
                self._close_candidate_frames += 1
                if self._close_candidate_frames >= confirm_frames:
                    self._eyes_closed = True
                    self.closed_frames = self._close_candidate_frames
            else:
                self._close_candidate_frames = 0
                self.closed_frames = 0

        # Three-way label for debug / robustness (partial ≠ closed).
        if self._eyes_closed:
            self.eye_state = "CLOSED"
        elif smooth_ear < self.ear_open_threshold:
            self.eye_state = "PARTIAL"
        else:
            self.eye_state = "OPEN"

    def update(self, left_eye_pts, right_eye_pts, fps):
        left = compute_EAR(left_eye_pts)
        right = compute_EAR(right_eye_pts)
        self.raw_ear = (left + right) / 2.0
        smooth_ear = self.smooth_ear_ma.update(self.raw_ear)

        self._update_threshold(smooth_ear)
        self._update_closed_state(smooth_ear, fps)

        fps_val = max(float(fps), 1.0)
        sustained = self._eyes_closed and self.closed_frames >= PERCLOS_MIN_CLOSED_FRAMES
        self.perclos_window.append(1 if sustained else 0)
        perclos = float(np.mean(self.perclos_window)) if self.perclos_window else 0.0

        relative_ear = None
        if self.baseline_ear and self.baseline_ear > 1e-6:
            relative_ear = float(smooth_ear / self.baseline_ear)

        return {
            "raw_ear": self.raw_ear,
            "ear": smooth_ear,
            "baseline_ear": self.baseline_ear,
            "ear_threshold": self.ear_threshold,
            "ear_close_threshold": self.ear_close_threshold,
            "ear_open_threshold": self.ear_open_threshold,
            "relative_ear": relative_ear,
            "eyes_closed": self._eyes_closed,
            "eye_state": self.eye_state,
            "perclos": perclos,
            "perclos_window_len": len(self.perclos_window),
            "closed_frames": self.closed_frames,
            "eye_closed_sec": self.closed_frames / fps_val,
            "blink_count": self.blink_count,
            "calibrated": self.calibrated,
        }


class YawnAnalyzer:
    """
    Authoritative temporal yawn state machine.

    Sequence:
      IDLE → CANDIDATE (MAR ≥ start) → CONFIRMED (≥1.5s + peak) → event once
           → stay active while MAR ≥ end → IDLE after end-hold → cooldown

    Critical separation:
      yawn_active     = mouth currently in a confirmed yawn
      yawn_event      = rising edge / one-shot count increment
      yawn_message    = short UI flash ("YAWNING DETECTED")
      yawn_confidence = current evidence only (decays when idle)

    EAR/eyes_closed are NEVER used as yawn evidence.
    """

    def __init__(self):
        self.mar_history = deque(maxlen=100)
        self.smoothed_mar = None
        self.baseline_mar = 0.20
        self._mar_calibration = []
        self.mar_calibrated = False
        self.open_frames = 0
        self._below_end_frames = 0
        self.yawn_count = 0
        self._cooldown = 0
        self.yawn_state = "NO_YAWN"  # NO_YAWN | POSSIBLE_YAWN | CONFIRMED_YAWN | RECOVERY
        self.is_yawning = False
        self.yawn_active = False
        self.yawn_event = False
        self.is_talking = False
        self.fatigue_yawn = False
        self.repeated_yawning = False
        self.frequent_yawning = False
        self.yawn_confidence = 0.0
        self.yawn_timestamps = deque()
        self.yawn_alert_frames = 0
        self.last_yawn_time = None
        self.area_baseline = None
        self.prev_mar = None
        self.mar_velocities = deque(maxlen=15)
        self.yawn_peak_mar = 0.0
        self.big_yawn_frames = 0
        self.big_yawning = False

    def _adaptive_thresholds(self):
        """Personal start/end thresholds with safe absolute clamps."""
        start_thr = self.baseline_mar + MAR_BASELINE_DELTA_START
        start_thr = float(np.clip(start_thr, MAR_YAWN_START_THRESHOLD, MAR_YAWN_START_MAX))
        end_thr = start_thr - MAR_BASELINE_HYSTERESIS
        end_thr = float(np.clip(end_thr, MAR_YAWN_END_THRESHOLD, MAR_YAWN_END_MAX))
        # Enforce hysteresis invariant: START > END
        if end_thr >= start_thr:
            end_thr = start_thr - 0.08
        return start_thr, end_thr

    def _update_baseline(self, mar, start_thr):
        """
        Calibrate / adapt resting MAR only while idle (not during a yawn candidate).
        Prevents the detector from learning a yawn as "normal mouth."
        """
        if self.yawn_state in ("POSSIBLE_YAWN", "CONFIRMED_YAWN"):
            return

        if not self.mar_calibrated:
            # Collect neutral mouth samples (below provisional start).
            if mar < start_thr * 0.85:
                self._mar_calibration.append(mar)
            if len(self._mar_calibration) >= MAR_CALIBRATION_FRAMES:
                arr = np.array(self._mar_calibration, dtype=float)
                self.baseline_mar = float(np.clip(np.median(arr), MAR_BASELINE_MIN, MAR_BASELINE_MAX))
                self.mar_calibrated = True
            return

        # Slow EMA only when mouth is near resting (not talking spikes / yawns).
        if mar < self.baseline_mar + 0.08 and mar < start_thr * 0.80:
            alpha = 0.015
            self.baseline_mar = (1.0 - alpha) * self.baseline_mar + alpha * mar
            self.baseline_mar = float(np.clip(self.baseline_mar, MAR_BASELINE_MIN, MAR_BASELINE_MAX))

    def _compute_confidence(self, mar, start_thr, end_thr, dur_sec, head_pose_consistency):
        """
        Confidence from CURRENT evidence. Idle / closed mouth → near zero.
        Persistence alone must not keep confidence at 1.0 after the mouth closes.
        """
        if self.yawn_state == "NO_YAWN" and self.yawn_alert_frames <= 0:
            return 0.0

        if mar < end_thr and self.yawn_state in ("NO_YAWN", "RECOVERY"):
            return max(0.0, self.yawn_confidence * 0.7)

        mar_span = max(0.15, start_thr - self.baseline_mar)
        mar_norm = min(1.0, max(0.0, (mar - self.baseline_mar) / mar_span))
        peak_norm = min(1.0, max(0.0, (self.yawn_peak_mar - self.baseline_mar) / max(mar_span, MAR_PEAK_MIN_DELTA)))
        persistence_norm = min(1.0, dur_sec / YAWN_MIN_DURATION_SEC) if self.yawn_state != "NO_YAWN" else 0.0

        if self.yawn_state == "CONFIRMED_YAWN":
            raw = 0.45 * mar_norm + 0.30 * peak_norm + 0.15 * persistence_norm + 0.10 * head_pose_consistency
        elif self.yawn_state == "POSSIBLE_YAWN":
            raw = 0.50 * mar_norm + 0.20 * peak_norm + 0.20 * persistence_norm + 0.10 * head_pose_consistency
        elif self.yawn_alert_frames > 0:
            # Display flash may linger briefly; decay with mouth closure.
            raw = 0.35 * mar_norm + 0.35 * peak_norm + 0.30 * (self.yawn_alert_frames / max(1, int(30)))
        else:
            raw = 0.0

        return float(np.clip(raw, 0.0, 1.0))

    def update(
        self,
        mouth_points,
        width,
        height,
        fps,
        ear=None,
        baseline_ear=None,
        pitch=None,
        relative_pitch=None,
        yaw=None,
        roll=None,
        cnn_score=None,
        eyes_closed=False,
        current_time=None,
    ):
        # NOTE: ear / eyes_closed / cnn_score intentionally unused for yawn decision.
        # They remain in the signature for call-site compatibility only.
        if current_time is None:
            current_time = time.time()

        # Use true evaluation FPS (do not floor to 10 — that breaks 1.5s timing at low FPS).
        fps_val = max(float(fps), 1.0)
        required_frames = max(1, int(math.ceil(YAWN_MIN_DURATION_SEC * fps_val)))
        end_hold_frames = max(1, int(math.ceil(YAWN_END_HOLD_SEC * fps_val)))
        cooldown_frames = max(1, int(math.ceil(YAWN_COOLDOWN_SEC * fps_val)))
        display_frames = max(1, int(math.ceil(YAWN_DISPLAY_SEC * fps_val)))

        target_history_len = max(10, int(fps_val * YAWN_WINDOW_5S))
        if self.mar_history.maxlen != target_history_len:
            self.mar_history = deque(self.mar_history, maxlen=target_history_len)

        # 1) Raw + EMA-smoothed MAR
        mar_raw = compute_MAR(mouth_points, width, height)
        if self.smoothed_mar is None:
            self.smoothed_mar = mar_raw
        else:
            self.smoothed_mar = 0.65 * self.smoothed_mar + 0.35 * mar_raw
        mar = self.smoothed_mar
        area_raw, m_width, m_height = compute_mouth_area(mouth_points, width, height)
        self.mar_history.append(mar)

        start_thr, end_thr = self._adaptive_thresholds()
        self._update_baseline(mar, start_thr)
        # Recompute after possible baseline update
        start_thr, end_thr = self._adaptive_thresholds()
        relative_mar = max(0.0, mar - self.baseline_mar)

        # 2) Talking filter: rapid MAR oscillation (speech), not sustained open (yawn)
        if self.prev_mar is not None:
            velocity = (mar - self.prev_mar) * fps_val
        else:
            velocity = 0.0
        self.prev_mar = mar
        self.mar_velocities.append(velocity)

        mar_std = float(np.std(self.mar_history)) if len(self.mar_history) > 10 else 0.0
        sign_changes = 0
        if len(self.mar_velocities) > 3:
            vel_arr = list(self.mar_velocities)
            for i in range(1, len(vel_arr)):
                if (vel_arr[i] * vel_arr[i - 1]) < -0.0002:
                    sign_changes += 1
        self.is_talking = (sign_changes >= 4 or mar_std > MAR_TALKING_STD_THRESHOLD) and (mar < start_thr)

        # 3) Head-pose gate (extreme turn/tilt suppresses yawn candidates)
        head_pose_consistency = 1.0
        if yaw is not None and abs(yaw) > 35.0:
            head_pose_consistency = 0.0
        elif roll is not None and abs(roll) > 35.0:
            head_pose_consistency = 0.0
        elif relative_pitch is not None and abs(relative_pitch) > 35.0:
            head_pose_consistency = 0.0

        if self._cooldown > 0:
            self._cooldown -= 1

        self.yawn_event = False
        peak_needed = self.baseline_mar + MAR_PEAK_MIN_DELTA

        # 4) Temporal state machine
        if self.yawn_state == "NO_YAWN":
            self.open_frames = 0
            self._below_end_frames = 0
            self.yawn_peak_mar = 0.0
            if (
                mar >= start_thr
                and not self.is_talking
                and head_pose_consistency > 0
                and self._cooldown == 0
            ):
                self.yawn_state = "POSSIBLE_YAWN"
                self.open_frames = 1
                self.yawn_peak_mar = mar

        elif self.yawn_state == "POSSIBLE_YAWN":
            if mar >= end_thr and head_pose_consistency > 0 and not self.is_talking:
                self.open_frames += 1
                self.yawn_peak_mar = max(self.yawn_peak_mar, mar)
                self._below_end_frames = 0
                # Confirm only after sustained opening AND meaningful peak (rejects smile/talk spikes)
                if (
                    self.open_frames >= required_frames
                    and self.yawn_peak_mar >= max(start_thr, peak_needed)
                ):
                    self.yawn_state = "CONFIRMED_YAWN"
                    self.yawn_count += 1
                    self.yawn_event = True
                    self.last_yawn_time = current_time
                    self.yawn_timestamps.append(current_time)
                    self.yawn_alert_frames = display_frames
                    # Cooldown starts after yawn ends; keep zero while still open
            else:
                self._below_end_frames += 1
                if self._below_end_frames >= end_hold_frames or self.is_talking:
                    self.yawn_state = "RECOVERY"
                    self.open_frames = 0
                    self.yawn_peak_mar = 0.0
                    self._below_end_frames = 0

        elif self.yawn_state == "CONFIRMED_YAWN":
            if mar >= end_thr and head_pose_consistency > 0:
                self.open_frames += 1
                self.yawn_peak_mar = max(self.yawn_peak_mar, mar)
                self._below_end_frames = 0
            else:
                # Mouth closing — require short hold below END so noise does not re-trigger
                self._below_end_frames += 1
                if self._below_end_frames >= end_hold_frames:
                    self.yawn_state = "RECOVERY"
                    self.open_frames = 0
                    self._below_end_frames = 0
                    self._cooldown = cooldown_frames

        elif self.yawn_state == "RECOVERY":
            if mar < end_thr:
                self.yawn_state = "NO_YAWN"
                self.open_frames = 0
                self.yawn_peak_mar = 0.0
                self._below_end_frames = 0
            elif (
                mar >= start_thr
                and not self.is_talking
                and head_pose_consistency > 0
                and self._cooldown == 0
            ):
                self.yawn_state = "POSSIBLE_YAWN"
                self.open_frames = 1
                self.yawn_peak_mar = mar
                self._below_end_frames = 0
            else:
                # Still elevated but not a new candidate — wait for true close
                if mar < start_thr:
                    self._below_end_frames += 1
                    if self._below_end_frames >= end_hold_frames:
                        self.yawn_state = "NO_YAWN"
                        self.open_frames = 0
                        self.yawn_peak_mar = 0.0

        if self.yawn_alert_frames > 0:
            self.yawn_alert_frames -= 1

        # 5) Frequent-yawn windows → fatigue evidence (not a permanent active yawn)
        while self.yawn_timestamps and (current_time - self.yawn_timestamps[0] > YAWN_WINDOW_120S):
            self.yawn_timestamps.popleft()

        yawns_in_60s = sum(1 for t in self.yawn_timestamps if (current_time - t) <= YAWN_WINDOW_60S)
        yawns_in_120s = len(self.yawn_timestamps)
        self.repeated_yawning = yawns_in_60s >= 2
        self.frequent_yawning = yawns_in_120s >= 3
        self.fatigue_yawn = self.frequent_yawning or (self.yawn_count >= YAWN_FATIGUE_COUNT)

        self.yawn_active = self.yawn_state == "CONFIRMED_YAWN"
        # UI/message uses alert timer only — do NOT keep "YAWNING DETECTED" for the whole open duration
        yawn_message = self.yawn_alert_frames > 0
        # Legacy flag: true for active yawn OR short display flash (call sites / graphs)
        self.is_yawning = self.yawn_active or yawn_message

        # Risk bonus: modest for a recent single event; stronger for repeated/frequent
        if self.yawn_active or yawn_message:
            if self.frequent_yawning:
                yawn_bonus = YAWN_BONUS_FREQUENT
            elif self.repeated_yawning:
                yawn_bonus = YAWN_BONUS_REPEATED
            else:
                yawn_bonus = YAWN_BONUS_SINGLE
        elif self.frequent_yawning:
            yawn_bonus = YAWN_BONUS_FREQUENT * 0.5
        elif self.repeated_yawning:
            yawn_bonus = YAWN_BONUS_REPEATED * 0.4
        else:
            yawn_bonus = 0.0

        dur_sec = self.open_frames / fps_val
        self.yawn_confidence = self._compute_confidence(
            mar, start_thr, end_thr, dur_sec, head_pose_consistency
        )

        # Big yawn: large sustained opening (separate from normal yawn event)
        if mar >= max(0.70, start_thr + 0.12) and relative_mar >= MAR_PEAK_MIN_DELTA and not self.is_talking:
            self.big_yawn_frames += 1
        else:
            self.big_yawn_frames = 0
        self.big_yawning = self.big_yawn_frames >= max(1, int(math.ceil(2.5 * fps_val)))

        last_yawn_sec_ago = (current_time - self.last_yawn_time) if self.last_yawn_time else None

        return {
            "mar": self.smoothed_mar,
            "mar_raw": mar_raw,
            "raw_mar": mar_raw,
            "baseline_mar": self.baseline_mar,
            "relative_mar": relative_mar,
            "yawn_start_threshold": start_thr,
            "yawn_end_threshold": end_thr,
            "yawn_peak_mar": self.yawn_peak_mar,
            "open_frames": self.open_frames,
            "required_frames": required_frames,
            "yawn_confidence": float(self.yawn_confidence),
            "yawn_count": self.yawn_count,
            "is_yawning": self.is_yawning,
            "yawn_active": self.yawn_active,
            "yawn_event": self.yawn_event,
            "yawn_cooldown": self._cooldown,
            "yawn_detected_alert": yawn_message,
            "fatigue_yawn": self.fatigue_yawn,
            "frequent_yawning": self.frequent_yawning,
            "repeated_yawning": self.repeated_yawning,
            "yawn_bonus": yawn_bonus,
            "is_talking": self.is_talking,
            "big_yawning": self.big_yawning,
            "big_yawn_frames": self.big_yawn_frames,
            "last_yawn_sec_ago": last_yawn_sec_ago,
            "opening_area": area_raw,
            "yawn_state": self.yawn_state,
            "open_duration_sec": dur_sec,
            "mar_calibrated": self.mar_calibrated,
        }


class HeadPoseAnalyzer:
    """
    Comprehensive Head Posture Tracking:
    Detects Pitch (head down/up), Yaw (looking left/right), and Roll (tilting left/right).
    Uses strict, independent consecutive counters to avoid rapid flickering.
    """

    def __init__(self):
        self.calibrated = False
        self._calibration = []
        self.pitch_baseline = None
        self.smooth_pitch = MovingAverage(maxlen=6)

        # Posture triggers
        self._head_down_active = False
        self._head_up_active = False
        self._head_left_active = False
        self._head_right_active = False
        self._head_tilt_left_active = False
        self._head_tilt_right_active = False

        # Independent consecutive counters
        self.head_down_counter = ConsecutiveCounter(HEAD_DOWN_PERSIST_FRAMES)
        self.head_down_exit_counter = ConsecutiveCounter(STATE_EXIT_FRAMES)

        self.head_up_counter = ConsecutiveCounter(12)
        self.head_up_exit_counter = ConsecutiveCounter(10)

        self.head_left_counter = ConsecutiveCounter(12)
        self.head_left_exit_counter = ConsecutiveCounter(10)

        self.head_right_counter = ConsecutiveCounter(12)
        self.head_right_exit_counter = ConsecutiveCounter(10)

        self.head_tilt_left_counter = ConsecutiveCounter(12)
        self.head_tilt_left_exit_counter = ConsecutiveCounter(10)

        self.head_tilt_right_counter = ConsecutiveCounter(12)
        self.head_tilt_right_exit_counter = ConsecutiveCounter(10)

        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.relative_pitch = 0.0

    def update(self, landmarks, width, height, eyes_closed=False):
        pitch, self.yaw, self.roll = head_pose(landmarks, width, height)
        self.pitch = self.smooth_pitch.update(pitch)

        if not self.calibrated:
            if not eyes_closed:
                self._calibration.append(self.pitch)
            if len(self._calibration) >= HEAD_CALIBRATION_FRAMES:
                self.pitch_baseline = float(np.median(self._calibration))
                self.calibrated = True
        elif self.pitch_baseline is not None and not eyes_closed:
            if abs(self.pitch - self.pitch_baseline) < 10.0:
                self.pitch_baseline = 0.995 * self.pitch_baseline + 0.005 * self.pitch

        if self.pitch_baseline is not None:
            self.relative_pitch = self.pitch_baseline - self.pitch
        else:
            self.relative_pitch = 0.0

        # --- A) HEAD DOWN DETECTION (Pitch drop) ---
        head_down_candidate = self.calibrated and self.relative_pitch > HEAD_DOWN_DELTA_DEG
        if not self._head_down_active:
            if self.head_down_counter.update(head_down_candidate):
                self._head_down_active = True
                self.head_down_exit_counter.reset()
        else:
            if self.head_down_exit_counter.update(not head_down_candidate):
                self._head_down_active = False
                self.head_down_counter.reset()

        # --- B) HEAD UP DETECTION (Pitch rise) ---
        head_up_candidate = self.calibrated and self.relative_pitch < -HEAD_DOWN_DELTA_DEG - 2.0
        if not self._head_up_active:
            if self.head_up_counter.update(head_up_candidate):
                self._head_up_active = True
                self.head_up_exit_counter.reset()
        else:
            if self.head_up_exit_counter.update(not head_up_candidate):
                self._head_up_active = False
                self.head_up_counter.reset()

        # --- C) HEAD LEFT DETECTION (Yaw left) ---
        head_left_candidate = self.calibrated and self.yaw < -25.0
        if not self._head_left_active:
            if self.head_left_counter.update(head_left_candidate):
                self._head_left_active = True
                self.head_left_exit_counter.reset()
        else:
            if self.head_left_exit_counter.update(not head_left_candidate):
                self._head_left_active = False
                self.head_left_counter.reset()

        # --- D) HEAD RIGHT DETECTION (Yaw right) ---
        head_right_candidate = self.calibrated and self.yaw > 25.0
        if not self._head_right_active:
            if self.head_right_counter.update(head_right_candidate):
                self._head_right_active = True
                self.head_right_exit_counter.reset()
        else:
            if self.head_right_exit_counter.update(not head_right_candidate):
                self._head_right_active = False
                self.head_right_counter.reset()

        # --- E) HEAD TILT LEFT DETECTION (Roll left) ---
        head_tilt_left_candidate = self.calibrated and self.roll < -25.0
        if not self._head_tilt_left_active:
            if self.head_tilt_left_counter.update(head_tilt_left_candidate):
                self._head_tilt_left_active = True
                self.head_tilt_left_exit_counter.reset()
        else:
            if self.head_tilt_left_exit_counter.update(not head_tilt_left_candidate):
                self._head_tilt_left_active = False
                self.head_tilt_left_counter.reset()

        # --- F) HEAD TILT RIGHT DETECTION (Roll right) ---
        head_tilt_right_candidate = self.calibrated and self.roll > 25.0
        if not self._head_tilt_right_active:
            if self.head_tilt_right_counter.update(head_tilt_right_candidate):
                self._head_tilt_right_active = True
                self.head_tilt_right_exit_counter.reset()
        else:
            if self.head_tilt_right_exit_counter.update(not head_tilt_right_candidate):
                self._head_tilt_right_active = False
                self.head_tilt_right_counter.reset()

        any_tilt = (
            self._head_up_active
            or self._head_left_active
            or self._head_right_active
            or self._head_tilt_left_active
            or self._head_tilt_right_active
        )

        return {
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "pitch_baseline": self.pitch_baseline,
            "relative_pitch": self.relative_pitch,
            "head_down": self._head_down_active,
            "head_up": self._head_up_active,
            "head_left": self._head_left_active,
            "head_right": self._head_right_active,
            "head_tilt_left": self._head_tilt_left_active,
            "head_tilt_right": self._head_tilt_right_active,
            "head_down_frames": self.head_down_counter.count if not self._head_down_active else self.head_down_exit_counter.count,
            "calibrated": self.calibrated,
            "any_tilt": any_tilt,
        }
