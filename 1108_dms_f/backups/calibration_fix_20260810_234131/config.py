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
MOUTH_INDICES = [61, 81, 13, 311, 308, 402]

CNN_INPUT_SIZE = (224, 224)
CNN_SMOOTH_WINDOW = 40

# EAR — close when below threshold; open when clearly above
EAR_CALIBRATION_FRAMES = 45
EAR_BASELINE_EMA_ALPHA = 0.03
EAR_THRESHOLD_RATIO = 0.78
EAR_THRESHOLD_FLOOR = 0.20
EAR_OPEN_HYSTERESIS = 0.018
EAR_SMOOTH_WINDOW = 8
BLINK_MIN_CLOSED_FRAMES = 2

PERCLOS_WINDOW = 150
PERCLOS_MIN_CLOSED_FRAMES = 2

MAR_SMOOTH_WINDOW = 60
MAR_YAWN_THRESHOLD = 0.58
MAR_YAWN_START_THRESHOLD = 0.58
MAR_YAWN_END_THRESHOLD = 0.48
MAR_TALKING_STD_THRESHOLD = 0.08
YAWN_MIN_DURATION_SEC = 1.2
YAWN_COOLDOWN_SEC = 3.0
YAWN_FATIGUE_COUNT = 2

# Advanced Yawning Behavioral Parameters
YAWN_WINDOW_5S = 5.0
YAWN_WINDOW_60S = 60.0
YAWN_WINDOW_120S = 120.0
YAWN_BONUS_SINGLE = 0.08
YAWN_BONUS_REPEATED = 0.15
YAWN_BONUS_FREQUENT = 0.25
YAWN_DISPLAY_SEC = 3.0


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
