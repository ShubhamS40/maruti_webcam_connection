"""
Behavioral analyzers — high accuracy eye closure, multi-posture head analysis, and yawning.
"""

import time
from collections import deque
import numpy as np

from config import (
    BLINK_MIN_CLOSED_FRAMES,
    EAR_BASELINE_EMA_ALPHA,
    EAR_CALIBRATION_FRAMES,
    EAR_SMOOTH_WINDOW,
    EAR_THRESHOLD_FLOOR,
    EAR_THRESHOLD_RATIO,
    HEAD_CALIBRATION_FRAMES,
    HEAD_DOWN_DELTA_DEG,
    HEAD_DOWN_PERSIST_FRAMES,
    MAR_SMOOTH_WINDOW,
    MAR_TALKING_STD_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    MAR_YAWN_START_THRESHOLD,
    MAR_YAWN_END_THRESHOLD,
    PERCLOS_MIN_CLOSED_FRAMES,
    PERCLOS_WINDOW,
    STATE_EXIT_FRAMES,
    YAWN_BONUS_FREQUENT,
    YAWN_BONUS_REPEATED,
    YAWN_BONUS_SINGLE,
    YAWN_COOLDOWN_SEC,
    YAWN_DISPLAY_SEC,
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
    Eyes closed when smoothed EAR < personalized threshold.
    Hysteresis only on reopen (prevents flutter). Works with glasses/small eyes.
    Tuned so that even if eyes are open 20% relative to baseline, the driver is ATTENTIVE.
    """

    def __init__(self):
        self.calibrated = False
        self._calibration_samples = []
        self.baseline_ear = None
        self.ear_threshold = EAR_THRESHOLD_FLOOR
        self.raw_ear = 0.0
        self.smooth_ear_ma = MovingAverage(maxlen=EAR_SMOOTH_WINDOW)
        self._eyes_closed = False
        self.closed_frames = 0
        self.blink_count = 0
        self.perclos_window = deque(maxlen=PERCLOS_WINDOW)

    def _update_threshold(self, smooth_ear):
        if not self.calibrated:
            self._calibration_samples.append(smooth_ear)
            if len(self._calibration_samples) >= EAR_CALIBRATION_FRAMES:
                arr = np.array(self._calibration_samples)
                med = float(np.median(arr))
                # Clamp baseline between [0.24, 0.40] to support small eyes
                self.baseline_ear = max(0.24, min(0.40, med))
                self.calibrated = True
        elif self.baseline_ear is not None and not self._eyes_closed:
            # Slow adaptive update when fully open
            if smooth_ear > self.ear_threshold + 0.02:
                alpha = 0.005
                new_val = (1 - alpha) * self.baseline_ear + alpha * smooth_ear
                self.baseline_ear = max(0.24, min(0.40, new_val))

        if self.baseline_ear is not None:
            # Set closed threshold to EAR_THRESHOLD_RATIO (78%) of baseline EAR.
            # Floor at EAR_THRESHOLD_FLOOR (0.20) prevents false positives with noisy low-EAR faces.
            # Ceiling at 0.25 accommodates large-eyed users without over-triggering.
            self.ear_threshold = max(EAR_THRESHOLD_FLOOR, min(0.25, self.baseline_ear * EAR_THRESHOLD_RATIO))


    def _update_closed_state(self, smooth_ear):
        thr = self.ear_threshold
        # Hysteresis decision
        if self._eyes_closed:
            if smooth_ear > thr + 0.02:
                self._eyes_closed = False
        else:
            if smooth_ear < thr:
                self._eyes_closed = True

    def update(self, left_eye_pts, right_eye_pts, fps):
        left = compute_EAR(left_eye_pts)
        right = compute_EAR(right_eye_pts)
        self.raw_ear = (left + right) / 2.0
        smooth_ear = self.smooth_ear_ma.update(self.raw_ear)

        self._update_threshold(smooth_ear)
        self._update_closed_state(smooth_ear)

        if self._eyes_closed:
            self.closed_frames += 1
        else:
            if self.closed_frames > BLINK_MIN_CLOSED_FRAMES:
                self.blink_count += 1
            self.closed_frames = 0

        # Roll perclos window
        sustained = self._eyes_closed and self.closed_frames >= PERCLOS_MIN_CLOSED_FRAMES
        self.perclos_window.append(1 if sustained else 0)
        perclos = float(np.mean(self.perclos_window)) if self.perclos_window else 0.0

        return {
            "raw_ear": self.raw_ear,
            "ear": smooth_ear,
            "baseline_ear": self.baseline_ear,
            "ear_threshold": self.ear_threshold,
            "eyes_closed": self._eyes_closed,
            "perclos": perclos,
            "perclos_window_len": len(self.perclos_window),
            "closed_frames": self.closed_frames,
            "eye_closed_sec": self.closed_frames / max(float(fps), 1.0),
            "blink_count": self.blink_count,
            "calibrated": self.calibrated,
        }


class YawnAnalyzer:
    """
    Multi-Stage Behavior-Aware Yawning Detector.
    States: NO_YAWN -> POSSIBLE_YAWN -> CONFIRMED_YAWN -> RECOVERY.
    Only CONFIRMED_YAWN triggers YAWNING DETECTED.
    Features:
    1. EMA MAR smoothing (smoothed_mar = 0.8 * previous + 0.2 * current)
    2. Continuous mouth opening duration threshold (>= 1.5s using actual FPS)
    3. Hysteresis thresholds (MAR_YAWN_START_THRESHOLD = 0.78, MAR_YAWN_END_THRESHOLD = 0.68)
    4. Anti-talking filter using velocity derivative oscillations and MAR variance
    5. Head pose consistency filter (ignores yawning if head is turned heavily)
    6. Multi-cue confidence score (0.55 MAR + 0.20 persistence + 0.15 CNN + 0.10 head pose)
    7. Strict anti-eye-closure separation (EAR alone NEVER triggers or inflates yawning)
    8. Risk bonus (+0.08) applied ONLY after confirmed yawning
    """

    def __init__(self):
        self.mar_history = deque(maxlen=100)
        self.smoothed_mar = None
        self.baseline_mar = 0.20
        self.open_frames = 0
        self.yawn_count = 0
        self._cooldown = 0
        self.yawn_state = "NO_YAWN"
        self.is_yawning = False
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
        
        # Big yawning tracking
        self.big_yawn_frames = 0
        self.big_yawning = False

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
        if current_time is None:
            current_time = time.time()

        fps_val = max(float(fps), 10.0)
        target_history_len = int(fps_val * YAWN_WINDOW_5S)
        if self.mar_history.maxlen != target_history_len:
            self.mar_history = deque(self.mar_history, maxlen=target_history_len)

        # 1. Compute Raw MAR and EMA Smoothed MAR
        mar_raw = compute_MAR(mouth_points, width, height)
        if self.smoothed_mar is None:
            self.smoothed_mar = mar_raw
        else:
            self.smoothed_mar = 0.7 * self.smoothed_mar + 0.3 * mar_raw

        mar = self.smoothed_mar
        area_raw, m_width, m_height = compute_mouth_area(mouth_points, width, height)
        self.mar_history.append(mar)

        # 2. Dynamic MAR Baseline adaptation when mouth is resting/closed
        if mar < 0.40:
            alpha = 0.02
            self.baseline_mar = (1.0 - alpha) * self.baseline_mar + alpha * mar
            self.baseline_mar = max(0.12, min(0.32, self.baseline_mar))

        relative_mar = max(0.0, mar - self.baseline_mar)

        # 3. Velocity / Derivative Analysis & Anti-Talking Filter
        if self.prev_mar is not None:
            velocity = (mar - self.prev_mar) * fps_val
        else:
            velocity = 0.0
        self.prev_mar = mar
        self.mar_velocities.append(velocity)

        # Adaptive start threshold: baseline_mar + 0.35, clamped to [MAR_YAWN_START_THRESHOLD, 0.75]
        start_thr = max(MAR_YAWN_START_THRESHOLD, min(0.75, self.baseline_mar + 0.35))
        # Adaptive end threshold: start_thr - 0.10, clamped down to MAR_YAWN_END_THRESHOLD
        end_thr = max(MAR_YAWN_END_THRESHOLD, start_thr - 0.10)

        mar_std = float(np.std(self.mar_history)) if len(self.mar_history) > 10 else 0.0
        sign_changes = 0
        if len(self.mar_velocities) > 3:
            vel_arr = list(self.mar_velocities)
            for i in range(1, len(vel_arr)):
                if (vel_arr[i] * vel_arr[i-1]) < -0.0002:
                    sign_changes += 1

        # Talking causes rapid oscillations or high velocity sign changes while MAR is moderate (< start_thr)
        self.is_talking = (sign_changes >= 4 or mar_std > MAR_TALKING_STD_THRESHOLD) and (mar < start_thr)

        # 4. Head Pose Consistency Check (head must be forward relative to baseline)
        head_pose_consistency = 1.0
        if yaw is not None and abs(yaw) > 35.0:
            head_pose_consistency = 0.0
        elif roll is not None and abs(roll) > 35.0:
            head_pose_consistency = 0.0
        elif relative_pitch is not None and abs(relative_pitch) > 35.0:
            head_pose_consistency = 0.0

        # 5. Multi-Cue Yawn Confidence Calculation
        # Weights: 0.60 MAR + 0.25 temporal persistence + 0.15 head pose consistency
        # EAR alone MUST NEVER trigger or inflate yawning (0% weight from EAR)
        mar_norm = min(1.0, max(0.0, (mar - self.baseline_mar) / 0.35))
        dur_sec = self.open_frames / fps_val
        persistence_norm = min(1.0, dur_sec / YAWN_MIN_DURATION_SEC)

        yawn_confidence_raw = (
            0.60 * mar_norm +
            0.25 * persistence_norm +
            0.15 * head_pose_consistency
        )
        self.yawn_confidence = max(0.0, min(1.0, float(yawn_confidence_raw)))

        # 6. Multi-Stage State Machine Logic: NO_YAWN -> POSSIBLE_YAWN -> CONFIRMED_YAWN -> RECOVERY
        required_frames = max(1, int(fps_val * YAWN_MIN_DURATION_SEC))

        if self._cooldown > 0:
            self._cooldown -= 1

        if self.yawn_state == "NO_YAWN":
            if mar >= start_thr and not self.is_talking and head_pose_consistency > 0 and self._cooldown == 0:
                self.yawn_state = "POSSIBLE_YAWN"
                self.open_frames = 1
            else:
                self.open_frames = 0

        elif self.yawn_state == "POSSIBLE_YAWN":
            if mar >= end_thr and head_pose_consistency > 0:
                self.open_frames += 1
                current_dur = self.open_frames / fps_val
                if current_dur >= YAWN_MIN_DURATION_SEC and self.yawn_confidence >= 0.45:
                    self.yawn_state = "CONFIRMED_YAWN"
                    self.yawn_count += 1
                    self.last_yawn_time = current_time
                    self.yawn_timestamps.append(current_time)
                    self.yawn_alert_frames = int(fps_val * YAWN_DISPLAY_SEC)
                    self._cooldown = int(fps_val * YAWN_COOLDOWN_SEC)
            else:
                # MAR dropped before required frames or head turned -> reset candidate
                self.yawn_state = "RECOVERY"
                self.open_frames = 0

        elif self.yawn_state == "CONFIRMED_YAWN":
            if mar >= end_thr and head_pose_consistency > 0:
                self.open_frames += 1
            else:
                self.yawn_state = "RECOVERY"
                self.open_frames = 0

        elif self.yawn_state == "RECOVERY":
            if mar < end_thr - 0.05:
                self.yawn_state = "NO_YAWN"
                self.open_frames = 0
            elif mar >= start_thr and not self.is_talking and head_pose_consistency > 0 and self._cooldown == 0:
                self.yawn_state = "POSSIBLE_YAWN"
                self.open_frames = 1

        if self.yawn_alert_frames > 0:
            self.yawn_alert_frames -= 1

        # 7. Consecutive Yawning Tracking (60s and 120s windows)
        while self.yawn_timestamps and (current_time - self.yawn_timestamps[0] > YAWN_WINDOW_120S):
            self.yawn_timestamps.popleft()

        yawns_in_60s = sum(1 for t in self.yawn_timestamps if (current_time - t) <= YAWN_WINDOW_60S)
        yawns_in_120s = len(self.yawn_timestamps)

        self.repeated_yawning = (yawns_in_60s >= 2)
        self.frequent_yawning = (yawns_in_120s >= 3)

        # 8. Yawn Risk Bonus applied ONLY AFTER CONFIRMATION
        is_confirmed = (self.yawn_state == "CONFIRMED_YAWN") or (self.yawn_alert_frames > 0)
        if is_confirmed:
            if self.frequent_yawning:
                yawn_bonus = YAWN_BONUS_FREQUENT
            elif self.repeated_yawning:
                yawn_bonus = YAWN_BONUS_REPEATED
            else:
                yawn_bonus = YAWN_BONUS_SINGLE
        else:
            yawn_bonus = 0.0

        self.is_yawning = is_confirmed
        self.fatigue_yawn = self.frequent_yawning or (self.yawn_count >= YAWN_FATIGUE_COUNT)

        # 9. Big Yawning Tracking (MAR >= 0.75 and relative_mar >= 0.45 sustained for 2.5 seconds)
        if mar >= 0.75 and relative_mar >= 0.45 and not self.is_talking:
            self.big_yawn_frames += 1
        else:
            self.big_yawn_frames = 0

        big_yawn_threshold_frames = int(2.5 * fps_val)
        self.big_yawning = (self.big_yawn_frames >= big_yawn_threshold_frames)

        last_yawn_sec_ago = (current_time - self.last_yawn_time) if self.last_yawn_time else None

        return {
            "mar": self.smoothed_mar,
            "mar_raw": mar_raw,
            "raw_mar": mar_raw,
            "baseline_mar": self.baseline_mar,
            "relative_mar": relative_mar,
            "yawn_start_threshold": start_thr,
            "yawn_end_threshold": end_thr,
            "open_frames": self.open_frames,
            "required_frames": required_frames,
            "yawn_confidence": float(self.yawn_confidence),
            "yawn_count": self.yawn_count,
            "is_yawning": self.is_yawning,
            "yawn_active": self.is_yawning,
            "yawn_cooldown": self._cooldown,
            "yawn_detected_alert": (self.yawn_alert_frames > 0),
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
            "open_duration_sec": self.open_frames / fps_val,
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

        # Posture Alert: Is any head posture turn or tilt active?
        any_tilt = (
            self._head_up_active or
            self._head_left_active or
            self._head_right_active or
            self._head_tilt_left_active or
            self._head_tilt_right_active
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
