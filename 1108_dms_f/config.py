"""
Global configuration for the Driver Monitoring System (DMS).
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drowsy_model_full.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALARM_PATH = os.path.join(BASE_DIR, "static", "alarm.wav")

VIDEO_SOURCE = 0
FRAME_W = 1280
FRAME_H = 720
WINDOW_NAME = "DMS"
OUTPUT_FPS = 20.0

FACE_MESH_MAX_FACES = 1
FACE_MESH_DETECTION_CONF = 0.5
FACE_MESH_TRACKING_CONF = 0.5

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# 8-point MAR: left, right, then three (top,bottom) pairs.
# Uses true mouth corners (61/291). The previous set put 311/308 in the
# "right/bottom" slots, which inflated resting MAR (~0.6–0.7) and left
# CONFIRMED_YAWN stuck forever on a closed mouth.
MOUTH_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]

CNN_INPUT_SIZE = (224, 224)
CNN_SMOOTH_WINDOW = 40

# EAR — personal open-eye baseline + hysteresis (not a universal absolute cut).
# Why personal calibration: small-eye attentive drivers can have open EAR ~0.18.
# Why hysteresis: prevents OPEN/CLOSED flicker on noisy landmarks.
EAR_CALIBRATION_FRAMES = 45
EAR_BASELINE_EMA_ALPHA = 0.01
EAR_THRESHOLD_RATIO = 0.72          # close_thr ≈ baseline * ratio
EAR_THRESHOLD_FLOOR = 0.08          # safety floor only; must stay below small-eye baselines
EAR_THRESHOLD_CEILING = 0.28
EAR_OPEN_HYSTERESIS = 0.025         # open_thr = close_thr + hysteresis
EAR_BASELINE_MIN = 0.08
EAR_BASELINE_MAX = 0.42
EAR_SMOOTH_WINDOW = 8
EYE_CLOSE_CONFIRM_SEC = 0.12        # temporal confirm before EYES CLOSED (blinks still count)
BLINK_MIN_CLOSED_FRAMES = 2
BLINK_MAX_CLOSED_SEC = 0.45         # longer than this is drowsiness evidence, not a blink

PERCLOS_WINDOW = 150
PERCLOS_MIN_CLOSED_FRAMES = 2

# MAR / yawn — temporal event with start/end hysteresis + personal baseline.
# Why hysteresis: prevents candidate flicker; yawn ends only below END threshold.
# Absolute bounds constrain adaptive thresholds so baseline drift cannot make
# yawns impossible or permanently active.
# Correct 8-pt MAR rests ~0.25–0.45 (old broken landmarks inflated to ~0.6–0.7).
MAR_SMOOTH_WINDOW = 60
MAR_YAWN_THRESHOLD = 0.55
MAR_YAWN_START_THRESHOLD = 0.48     # absolute minimum start (adaptive may be higher)
MAR_YAWN_END_THRESHOLD = 0.36       # absolute minimum end (always < start)
MAR_YAWN_START_MAX = 0.90
MAR_YAWN_END_MAX = 0.75
MAR_BASELINE_DELTA_START = 0.22     # start_thr ≈ baseline + delta
MAR_BASELINE_HYSTERESIS = 0.10      # end_thr ≈ start_thr - hysteresis
MAR_PEAK_MIN_DELTA = 0.18           # peak must rise meaningfully above baseline
MAR_TALKING_STD_THRESHOLD = 0.06
YAWN_MIN_DURATION_SEC = 1.5         # real-time minimum; frames = ceil(1.5 * eval_fps)
YAWN_COOLDOWN_SEC = 1.5             # refractory period after mouth closes
YAWN_END_HOLD_SEC = 0.15            # mouth must stay below END this long to finish yawn
YAWN_FATIGUE_COUNT = 2
MAR_CALIBRATION_FRAMES = 45
MAR_BASELINE_MIN = 0.15
MAR_BASELINE_MAX = 0.50

# Advanced Yawning Behavioral Parameters
YAWN_WINDOW_5S = 5.0
YAWN_WINDOW_60S = 60.0
YAWN_WINDOW_120S = 120.0
YAWN_BONUS_SINGLE = 0.08
YAWN_BONUS_REPEATED = 0.15
YAWN_BONUS_FREQUENT = 0.25
YAWN_DISPLAY_SEC = 2.5              # UI "YAWNING DETECTED" flash only (not continuous state)


HEAD_CALIBRATION_FRAMES = 45
HEAD_DOWN_DELTA_DEG = 14.0
HEAD_DOWN_PERSIST_FRAMES = 12
HEAD_DOWN_DECAY = 2

RISK_EMA_ALPHA = 0.15

# Eye-closure timing (seconds) — primary state drivers
EYE_CLOSED_SEC_DROWSY = 3.0
EYE_CLOSED_SEC_MICROSLEEP = 5.0


STATE_ENTER_FATIGUED_FRAMES = 8
STATE_ENTER_DROWSY_FRAMES = 10
STATE_ENTER_MICROSLEEP_FRAMES = 12
STATE_EXIT_FRAMES = 10

ATTENTION_INITIAL = 100
ATTENTION_RISK_GAIN = 45.0
ATTENTION_RECOVERY = 1.8

DEFAULT_ALERT_SEC = 3
DEFAULT_EVAL_SEC = 2
DEFAULT_SPEED_LIMIT = 20

DEBUG_OVERLAY = True

# ==============================================================================
# SPEED MANAGER — Production-Grade Speed Subsystem Configuration
# ==============================================================================

# Activation threshold: DMS arms when smoothed speed >= this value (minus tolerance)
SPEED_ACTIVATION_KMH = 25.0

# Tolerance band applied to the activation threshold.
# Effective activation threshold = SPEED_ACTIVATION_KMH - SPEED_TOLERANCE_KMH
# e.g. 25 - 5 = 20 km/h  →  DMS activates at 20 km/h
SPEED_TOLERANCE_KMH = 5.0

# Hysteresis band below the effective activation threshold for deactivation.
# Effective deactivation threshold = (SPEED_ACTIVATION_KMH - SPEED_TOLERANCE_KMH) - SPEED_HYSTERESIS_KMH
# e.g. 20 - 5 = 15 km/h  →  DMS deactivates when speed falls to 15 km/h
# Prevents rapid switching (flickering) around the activation boundary.
SPEED_HYSTERESIS_KMH = 5.0

# Seconds the speed must remain above effective threshold before PENDING → ACTIVE
SPEED_PENDING_SEC = 1.0

# Grace period in seconds that speed must stay below deactivation threshold before ACTIVE → INACTIVE
SPEED_LOST_GRACE_SEC = 5.0

# Size of the deque window used for EMA smoothing of raw speed samples
SPEED_SMOOTH_WINDOW = 15

# Exponential Moving Average alpha for speed smoothing (lower = smoother, higher = more responsive)
SPEED_EMA_ALPHA = 0.2

# Seconds before an OkDriver push-injected speed value is considered stale
# and the system falls back to the next available provider
OKDRIVER_STALE_SEC = 5.0

# Provider polling interval in seconds (background thread)
SPEED_POLL_INTERVAL = 0.1   # 10 Hz


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    PURPLE = (180, 0, 255)
    GRAY = (180, 180, 180)
