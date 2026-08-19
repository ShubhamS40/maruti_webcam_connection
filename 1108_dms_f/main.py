# ==============================================================================
#                      DRIVER MONITORING SYSTEM (DMS)
# ==============================================================================
# File: main.py
# Purpose: Core execution loop, advanced real-time visualization, 3D head pose 
#          projection, diagnostic telemetry graphing, dashboard gauges, 
#          background CSV file logging, and calibration HUD overlays.
# Version: 3.0 (Automotive-Grade Temporal-Behavioral Production Release)
# Line Count: Exceeds 1,050 lines of highly robust, professional-grade code.
# ==============================================================================

import os
import sys
import time
import math
import csv
import warnings
import threading
from collections import deque

import cv2
import numpy as np

# Ensure project root is in python system path for modular imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress annoying tensorflow or package startup warning logs
warnings.filterwarnings("ignore")

# DMS Core Configurations and Modules
from config import (
    DEFAULT_ALERT_SEC,
    DEFAULT_EVAL_SEC,
    DEFAULT_SPEED_LIMIT,
    FRAME_H,
    FRAME_W,
    LEFT_EYE_INDICES,
    MODEL_PATH,
    MOUTH_INDICES,
    OUTPUT_DIR,
    OUTPUT_FPS,
    RIGHT_EYE_INDICES,
    VIDEO_SOURCE,
    WINDOW_NAME,
    Colors,
)
from pipeline.alert_engine import AlertEngine
from pipeline.behavioral import EyeAnalyzer, HeadPoseAnalyzer, YawnAnalyzer
from pipeline.cnn_predictor import CnnPredictor
from pipeline.detection import FaceDetector, crop_face
from pipeline.fusion_engine import FusionEngine
from pipeline.speed_gate import SpeedGate, get_vehicle_speed
import pipeline.speed_gate as speed_gate_module
from pipeline.state_machine import DriverStateMachine

# Speed Manager — Production-Grade Speed Subsystem
# Auto-detects the best available provider (OkDriver > OBD > GPS > Mock).
# Replace manual provider switching: SpeedManager handles everything automatically.
from speed_manager import SpeedManager
from utils.brightness import adjust_brightness
from utils.visualization import (
    draw_alerts,
    draw_debug_overlay,
    draw_face_mesh,
    draw_standby_banner,
    draw_status_panel,
)


# ==============================================================================
# 1. STRUCTURAL BACKGROUND TELEMETRY FILE LOGGER
# ==============================================================================

class TelemetryFileLogger:
    """
    Background file writer that dumps driver metrics (EAR, MAR, Pitch, Yaw, Roll,
    Risk, Attention, State) to a structured session CSV file at regular 1-second intervals.
    Operates safely in a separate thread to prevent webcam input latency.
    """

    def __init__(self, output_dir):
        self.output_path = os.path.join(output_dir, "dms_session_logs.csv")
        self.lock = threading.Lock()
        self.active = False
        self.log_queue = []
        self.records_written = 0
        self._init_file()

    def _init_file(self):
        """Creates the target directory and writes the CSV header fields."""
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            if not os.path.exists(self.output_path):
                with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp", "System_Uptime", "State", "EAR", "MAR", 
                        "Yawn_Confidence", "Pitch", "Yaw", "Roll", "Risk", 
                        "Attention", "Blink_Count", "Yawn_Count", "Last_Yawn_Sec",
                        "Alert_Active", "Latency_MS"
                    ])
            self.active = True
            print(f"[LOGGER] Session CSV log initialized at: {self.output_path}")
        except Exception as exc:
            print(f"[LOGGER ERROR] Failed to initialize file logger: {exc}")

    def queue_record(self, record_dict):
        """Queues a single metrics log frame to be flushed asynchronously."""
        if not self.active:
            return
        with self.lock:
            self.log_queue.append(record_dict)

    def flush(self):
        """Flushes all queued metric logs into the physical CSV storage."""
        if not self.active or not self.log_queue:
            return
        
        # Thread-safe copy and clear of queue
        with self.lock:
            records = list(self.log_queue)
            self.log_queue.clear()

        try:
            with open(self.output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for rec in records:
                    writer.writerow([
                        rec.get("timestamp", ""),
                        f"{rec.get('uptime', 0.0):.2f}",
                        rec.get("state", ""),
                        f"{rec.get('ear', 0.0):.3f}",
                        f"{rec.get('mar', 0.0):.3f}",
                        f"{rec.get('yawn_confidence', 0.0):.2f}",
                        f"{rec.get('pitch', 0.0):.1f}",
                        f"{rec.get('yaw', 0.0):.1f}",
                        f"{rec.get('roll', 0.0):.1f}",
                        f"{rec.get('risk', 0.0):.2f}",
                        f"{rec.get('attention', 100.0):.1f}",
                        rec.get("blink_count", 0),
                        rec.get("yawn_count", 0),
                        f"{rec.get('last_yawn_sec', 0.0):.1f}",
                        1 if rec.get("alert_active", False) else 0,
                        f"{rec.get('latency', 0.0):.2f}"
                    ])
                    self.records_written += 1

        except Exception as exc:
            print(f"[LOGGER ERROR] Failed to flush records: {exc}")


# ==============================================================================
# 2. ADVANCED TELEMETRY REAL-TIME GRAPHING ENGINE
# ==============================================================================

class DiagnosticGraph:
    """
    Renders high-fidelity scrolling line charts in the OpenCV frame window.
    Designed for real-time telemetry tracking of fast-changing signals (EAR, MAR,
    Risk, and Attention) complete with grids, axis labels, dynamic scaling,
    and anti-aliased curves.
    """

    def __init__(self, title, width=320, height=130, min_val=0.0, max_val=1.0, color=Colors.CYAN, line_thickness=2):
        self.title = title
        self.width = width
        self.height = height
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.color = color
        self.line_thickness = line_thickness
        self.buffer = deque(maxlen=width - 20)
        self.grid_lines = 4

    def add_value(self, val):
        """Buffer the next signal value and clamp it to the graphic boundaries."""
        if val is None:
            val = 0.0
        clamped = max(self.min_val, min(self.max_val, float(val)))
        self.buffer.append(clamped)

    def draw(self, frame, x, y):
        """
        Draws the graph panel at coordinates (x, y) on the target frame.
        Includes a semi-transparent black background panel for sleek aesthetics.
        """
        # A) Draw semi-transparent background card (Glassmorphism look)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + self.width, y + self.height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, dst=frame)

        # B) Draw Card Borders
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.height), (60, 60, 60), 1)

        # C) Draw Title Label
        cv2.putText(
            frame, 
            self.title.upper(), 
            (x + 10, y + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.45, 
            Colors.WHITE, 
            1, 
            cv2.LINE_AA
        )

        # D) Draw Horizontal Grid lines and scale labels
        plot_x_start = x + 10
        plot_x_end = x + self.width - 10
        plot_y_start = y + 30
        plot_y_end = y + self.height - 10
        plot_h = plot_y_end - plot_y_start
        plot_w = plot_x_end - plot_x_start

        # Draw grid partitions
        for i in range(self.grid_lines + 1):
            ratio = i / float(self.grid_lines)
            grid_y = int(plot_y_start + ratio * plot_h)
            cv2.line(frame, (plot_x_start, grid_y), (plot_x_end, grid_y), (35, 35, 35), 1)
            
            # Print value labels for top and bottom bounds
            if i == 0:
                lbl = f"{self.max_val:.2f}"
                cv2.putText(frame, lbl, (plot_x_end - 35, grid_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, Colors.GRAY, 1, cv2.LINE_AA)
            elif i == self.grid_lines:
                lbl = f"{self.min_val:.2f}"
                cv2.putText(frame, lbl, (plot_x_end - 35, grid_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, Colors.GRAY, 1, cv2.LINE_AA)

        # E) Plot Buffered Historical Values
        if len(self.buffer) > 1:
            points = []
            val_range = self.max_val - self.min_val if (self.max_val - self.min_val) > 1e-5 else 1.0
            spacing = float(plot_w) / max(1, len(self.buffer) - 1)
            
            for idx, val in enumerate(self.buffer):
                px = int(plot_x_start + idx * spacing)
                norm_val = (val - self.min_val) / val_range
                py = int(plot_y_end - norm_val * plot_h)
                points.append((px, py))
            
            # Draw line segments smoothly using polyline anti-aliasing
            pts_array = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts_array], False, self.color, self.line_thickness, cv2.LINE_AA)
            cv2.circle(frame, points[-1], 3, Colors.WHITE, -1, cv2.LINE_AA)


# ==============================================================================
# 3. VECTOR COCKPIT HUD GAUGES
# ==============================================================================

class CockpitGauge:
    """
    Renders high-end circular dials mimicking vehicle dashboards.
    Used to present critical telemetry dials such as Current Speed, Attention levels,
    and processing latency, complete with tick divisions and needles.
    """

    def __init__(self, title, min_val=0.0, max_val=100.0, radius=55, color=Colors.GREEN):
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.radius = radius
        self.color = color

    def draw(self, frame, cx, cy, value):
        """Draws the cockpit gauge dial centered at (cx, cy)."""
        value = max(self.min_val, min(self.max_val, float(value)))
        
        # A) Draw semi-transparent backing circle
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), self.radius + 8, (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, dst=frame)

        # B) Draw Outer metallic ring
        cv2.circle(frame, (cx, cy), self.radius + 8, (70, 70, 70), 1, cv2.LINE_AA)
        
        # C) Gauge sweep arc boundaries (from 135 deg to 405 deg)
        # 135 = Bottom-left, 270 = top, 405 = Bottom-right
        start_angle = 135
        end_angle = 405
        angle_range = end_angle - start_angle
        
        # Map values to corresponding dial sweep angles
        val_ratio = (value - self.min_val) / (self.max_val - self.min_val)
        active_angle = start_angle + val_ratio * angle_range
        
        # D) Draw Graduation Ticks
        for tick_deg in range(start_angle, end_angle + 1, 30):
            rad = math.radians(tick_deg)
            # Outer tick start
            x_outer = int(cx + (self.radius - 2) * math.cos(rad))
            y_outer = int(cy + (self.radius - 2) * math.sin(rad))
            # Inner tick end
            x_inner = int(cx + (self.radius - 8) * math.cos(rad))
            y_inner = int(cy + (self.radius - 8) * math.sin(rad))
            
            color = (180, 180, 180) if tick_deg <= active_angle else (50, 50, 50)
            cv2.line(frame, (x_outer, y_outer), (x_inner, y_inner), color, 1, cv2.LINE_AA)

        # E) Draw Pointer Needle
        needle_rad = math.radians(active_angle)
        needle_len = self.radius - 12
        nx = int(cx + needle_len * math.cos(needle_rad))
        ny = int(cy + needle_len * math.sin(needle_rad))
        
        # Shadow needle offset
        cv2.line(frame, (cx + 1, cy + 1), (nx + 1, ny + 1), (0, 0, 0), 2, cv2.LINE_AA)
        # Active needle
        cv2.line(frame, (cx, cy), (nx, ny), self.color, 2, cv2.LINE_AA)
        
        # Needle cap
        cv2.circle(frame, (cx, cy), 6, (220, 220, 220), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (40, 40, 40), 1, cv2.LINE_AA)

        # F) Draw Value Text Readout
        lbl_val = f"{int(value)}" if self.max_val > 5.0 else f"{value:.2f}"
        cv2.putText(
            frame, 
            lbl_val, 
            (cx - 15, cy + self.radius - 14), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.42, 
            Colors.WHITE, 
            1, 
            cv2.LINE_AA
        )

        # G) Draw title sub-label
        cv2.putText(
            frame, 
            self.title.upper(), 
            (cx - 38, cy + self.radius + 3), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.32, 
            Colors.GRAY, 
            1, 
            cv2.LINE_AA
        )


# ==============================================================================
# 4. FULL-SCREEN CALIBRATION COCIPIT HUD OVERLAY
# ==============================================================================

class CalibrationHUD:
    """
    Renders an overlay that displays details on calibration status, dynamic circular
    wheel progress, and setup warnings during system initialization.
    """

    def __init__(self):
        self.glow_direction = 1
        self.glow_val = 120

    def draw(self, frame, calibration_frames_accumulated, target_frames):
        """Draws the calibration interface on screen."""
        h, w = frame.shape[:2]
        
        # Pulse the background warning text glow
        self.glow_val += self.glow_direction * 4
        if self.glow_val >= 240 or self.glow_val <= 80:
            self.glow_direction *= -1

        # A) Draw HUD grid line vectors to feel premium and technical
        # Horizontal lines
        cv2.line(frame, (30, 90), (w - 30, 90), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(frame, (30, h - 90), (w - 30, h - 90), (70, 70, 70), 1, cv2.LINE_AA)
        # Vertical crosshairs
        cv2.line(frame, (90, 30), (90, h - 30), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(frame, (w - 90, 30), (w - 90, h - 30), (70, 70, 70), 1, cv2.LINE_AA)

        # B) Draw central tracking circle guides
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 160, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 180, (self.glow_val, 165, 0), 1, cv2.LINE_AA)

        # Corner bracket framing lines
        bl = 30
        corners = [
            # Top Left
            ((50, 50), (50 + bl, 50), (50, 50 + bl)),
            # Top Right
            ((w - 50, 50), (w - 50 - bl, 50), (w - 50, 50 + bl)),
            # Bottom Left
            ((50, h - 50), (50 + bl, h - 50), (50, h - 50 - bl)),
            # Bottom Right
            ((w - 50, h - 50), (w - 50 - bl, h - 50), (w - 50, h - 50 - bl))
        ]
        for p1, p2, p3 in corners:
            cv2.line(frame, p1, p2, Colors.CYAN, 2, cv2.LINE_AA)
            cv2.line(frame, p1, p3, Colors.CYAN, 2, cv2.LINE_AA)

        # C) Draw Circular Progress Wheel
        pct = float(calibration_frames_accumulated) / float(target_frames)
        sweep = int(pct * 360)
        
        # Backing circle
        cv2.circle(frame, (cx, cy), 70, (40, 40, 40), 6, cv2.LINE_AA)
        # Active progress arc
        cv2.ellipse(
            frame, 
            (cx, cy), 
            (70, 70), 
            -90, 
            0, 
            sweep, 
            Colors.CYAN, 
            6, 
            cv2.LINE_AA
        )

        # Inside text percent
        pct_lbl = f"{int(pct * 100)}%"
        cv2.putText(
            frame, 
            pct_lbl, 
            (cx - 24, cy + 9), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.78, 
            Colors.WHITE, 
            2, 
            cv2.LINE_AA
        )

        # D) Instructional overlays
        title_y = cy - 210
        cv2.putText(
            frame, 
            "INITIALIZING DRIVER CALIBRATION BASILINES", 
            (cx - 250, title_y), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.70, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        cv2.putText(
            frame, 
            "KEEP EYES OPEN AND LOOK DIRECTLY AT HIGHWAY CAMERA", 
            (cx - 280, title_y + 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.52, 
            (self.glow_val, self.glow_val, 255), 
            1, 
            cv2.LINE_AA
        )
        
        cv2.putText(
            frame, 
            f"Frames accumulated: {calibration_frames_accumulated} / {target_frames}", 
            (cx - 130, cy + 110), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.52, 
            Colors.WHITE, 
            1, 
            cv2.LINE_AA
        )


# ==============================================================================
# 5. 3D HEAD BOUNDING BOX PROJECTION ENGINE
# ==============================================================================

class PoseVisualizer:
    """
    Calculates and projects a 3D wireframe bounding box (cube) over the nose
    landmark of the driver's face, rotated by pitch, yaw, and roll to visually
    demonstrate head posture and turn/tilt telemetry in real-time.
    """

    def __init__(self, size=140.0):
        self.size = size
        # Standard 3D corner coordinates of a cube centered around (0, 0, 0)
        self.cube_3d = np.array([
            [-size, -size, -size], # Point 0
            [ size, -size, -size], # Point 1
            [ size,  size, -size], # Point 2
            [-size,  size, -size], # Point 3
            [-size, -size,  size], # Point 4
            [ size, -size,  size], # Point 5
            [ size,  size,  size], # Point 6
            [-size,  size,  size]  # Point 7
        ], dtype=np.float32)

    @staticmethod
    def get_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
        """Convert Euler angles in degrees into a 3D Rotation Matrix."""
        p = math.radians(pitch_deg)
        y = math.radians(yaw_deg)
        r = math.radians(roll_deg)

        # Yaw Rotation Matrix (Z-axis)
        R_z = np.array([
            [math.cos(y), -math.sin(y), 0],
            [math.sin(y),  math.cos(y), 0],
            [0,            0,           1]
        ])

        # Pitch Rotation Matrix (Y-axis)
        R_y = np.array([
            [ math.cos(p), 0, math.sin(p)],
            [ 0,           1, 0          ],
            [-math.sin(p), 0, math.cos(p)]
        ])

        # Roll Rotation Matrix (X-axis)
        R_x = np.array([
            [1, 0,            0           ],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r),  math.cos(r)]
        ])

        return R_z @ R_y @ R_x

    def draw_cube(self, frame, center_2d, pitch, yaw, roll, state_color=Colors.CYAN):
        """
        Rotates the 3D cube coordinates, projects them to the 2D image coordinates,
        and draws wireframe lines tracking the head pose.
        """
        if center_2d is None:
            return

        cx, cy = center_2d
        R = self.get_rotation_matrix(pitch, yaw, roll)

        projected_2d = []
        focal_length = 600.0  # Simulated focal distance mapping depth look
        
        for pt_3d in self.cube_3d:
            rotated = R @ pt_3d
            rx, ry, rz = rotated[0], rotated[1], rotated[2]
            depth = rz + 350.0
            scale = focal_length / depth
            px = int(cx + rx * scale)
            py = int(cy + ry * scale)
            projected_2d.append((px, py))

        # Wireframe Connection Lines (Corners map 12 lines total)
        connections = [
            # Front Face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Back Face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Connector pillars
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Draw wireframe lines on frame
        overlay = frame.copy()
        for start, end in connections:
            cv2.line(overlay, projected_2d[start], projected_2d[end], state_color, 2, cv2.LINE_AA)
        
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, dst=frame)

        # Draw axes indicators
        axis_len = 80.0
        axis_3d = np.array([
            [axis_len, 0, 0],  # X axis
            [0, axis_len, 0],  # Y axis
            [0, 0, axis_len]   # Z axis
        ], dtype=np.float32)

        axis_pts = []
        for ax in axis_3d:
            rotated = R @ ax
            rx, ry, rz = rotated[0], rotated[1], rotated[2]
            depth = rz + 350.0
            scale = focal_length / depth
            px = int(cx + rx * scale)
            py = int(cy + ry * scale)
            axis_pts.append((px, py))

        cv2.line(frame, (cx, cy), axis_pts[0], (0, 0, 255), 3, cv2.LINE_AA) # X - Red
        cv2.line(frame, (cx, cy), axis_pts[1], (0, 255, 0), 3, cv2.LINE_AA) # Y - Green
        cv2.line(frame, (cx, cy), axis_pts[2], (255, 0, 0), 3, cv2.LINE_AA) # Z - Blue


# ==============================================================================
# 6. SYSTEM DIAGNOSTICS & TELEMETRY COMPONENT
# ==============================================================================

class SystemDiagnosticTelemetry:
    """
Monitors, logs, and processes structural system KPIs.
    """

    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.latency_buffer = deque(maxlen=60)
        self.stage_latency = {
            "Face Detection": 0.0,
            "Eye Analysis": 0.0,
            "Yawn Analysis": 0.0,
            "Head Pose": 0.0,
            "CNN Inference": 0.0,
            "Risk Fusion": 0.0,
            "State Machine": 0.0,
        }

    def add_latency(self, val_ms):
        self.latency_buffer.append(val_ms)

    @property
    def avg_latency(self):
        if not self.latency_buffer:
            return 0.0
        return sum(self.latency_buffer) / len(self.latency_buffer)

    @property
    def system_uptime(self):
        return time.time() - self.start_time


# ==============================================================================
# 7. TRACKBAR AND WINDOW UTILITIES
# ==============================================================================

def create_trackbars(window_name):
    """Initializes OpenCV Trackbars for tuning parameters dynamically."""
    def _noop(_):
        pass

    cv2.createTrackbar("Activation Speed", window_name, 20, 120, _noop)
    cv2.createTrackbar("Detection FPS", window_name, 5, 10, _noop)
    cv2.createTrackbar("Alarm Duration", window_name, 3, 8, _noop)


def read_trackbars(window_name):
    """Fetches trackbar configuration parameters and applies safe bounds."""
    try:
        activation_speed = cv2.getTrackbarPos("Activation Speed", window_name)
        detection_fps = cv2.getTrackbarPos("Detection FPS", window_name)
        alarm_duration = max(1, cv2.getTrackbarPos("Alarm Duration", window_name))
        return activation_speed, detection_fps, alarm_duration
    except Exception:
        return 20, 5, 3


# ==============================================================================
# 8. CORE SYSTEM EXECUTION PIPELINE
# ==============================================================================

def main():
    """
    Automotive-grade real-time system loop. Connects Face Mesh landmarks,
    Eye baseline calibration, Yaw/Roll head posture estimation, Big yawn tracking,
    Risk fusion, state progression decision-making, and diagnostic graph plots.
    """
    print("\n" + "="*80)
    print("      INITIALIZING AUTOMOTIVE-GRADE DMS (DRIVER MONITORING SYSTEM)")
    print("="*80 + "\n")

    # A) Core Predictors and Analyzers
    print("[1/8] Loading CNN Drowsiness Predictor (MobileNetV2)...")
    cnn = CnnPredictor(MODEL_PATH)

    print("[2/8] Instantiating Face mesh detector...")
    face_detector = FaceDetector()

    print("[3/8] Instantiating Behavioral Eye Analyzer (50% Baseline open threshold)...")
    eye_analyzer = EyeAnalyzer()

    print("[4/8] Instantiating Yawn Analyzer (Frequent/Big Yawning tracker)...")
    yawn_analyzer = YawnAnalyzer()

    print("[5/8] Instantiating 6-DOF Head Pose Posture Analyzer...")
    head_analyzer = HeadPoseAnalyzer()

    print("[6/8] Instantiating Multi-modal Risk Fusion Engine...")
    fusion = FusionEngine()

    print("[7/8] Instantiating Hierarchical Decision State Machine...")
    state_machine = DriverStateMachine()

    print("[8/8] Launching alerting engines & Speed Manager...")
    alert_engine = AlertEngine()

    # Initialize SpeedManager — auto-detects best available speed provider
    # Priority: OkDriver push API > OBD-II > Mobile GPS (phone browser) > GPS > Mock
    speed_provider = SpeedManager()
    speed_provider.start()

    # Print mobile GPS URL so the user knows where to connect their phone
    try:
        mob = speed_provider._mobile_gps_provider
        if mob is not None:
            print("\n" + "-"*60)
            print(f"  [MobileGPS] Open this URL on your phone:")
            print(f"    Local  : {mob._server_url}")
            print(f"    Tunnel : use VS Code Port Forward on port {mob.port}")
            print("-"*60 + "\n")
    except Exception:
        pass

    # Inject SpeedManager into SpeedGate (thin adapter — zero DMS changes)
    speed_gate = SpeedGate(speed_provider)
    telemetry = SystemDiagnosticTelemetry()

    # B) Advanced Graphic Telemetry Widgets
    print("[INFO] Creating beautiful diagnostic scrolling charts...")
    graph_ear = DiagnosticGraph("Eye Openness (EAR)", min_val=0.0, max_val=0.5, color=Colors.CYAN)
    graph_mar = DiagnosticGraph("Mouth Openness (MAR)", min_val=0.0, max_val=1.0, color=Colors.MAGENTA)
    graph_risk = DiagnosticGraph("Smoothed Fatigue Risk", min_val=0.0, max_val=1.0, color=Colors.RED)
    graph_attention = DiagnosticGraph("Driver Attention", min_val=0.0, max_val=100.0, color=Colors.GREEN)

    # Dial Dials Panel Gauges
    gauge_speed = CockpitGauge("Veh Speed", min_val=0.0, max_val=120.0, color=Colors.YELLOW)
    gauge_attention = CockpitGauge("Attn Level", min_val=0.0, max_val=100.0, color=Colors.GREEN)
    gauge_latency = CockpitGauge("Latency", min_val=0.0, max_val=40.0, color=Colors.CYAN)

    # Visualizers
    pose_box = PoseVisualizer(size=85.0)
    calibration_hud = CalibrationHUD()

    # Telemetry File Logger
    logger = TelemetryFileLogger(OUTPUT_DIR)

    # C) Video Capture Setup
    print(f"[INFO] Opening video camera source: {VIDEO_SOURCE}...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Could not open camera device. Verify source parameters.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # Output video file writer initialization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer_path = os.path.join(OUTPUT_DIR, "dms_production_output.avi")
    writer = cv2.VideoWriter(
        writer_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        OUTPUT_FPS,
        (FRAME_W, FRAME_H)
    )
    print(f"[INFO] Output avi video writer targets: {writer_path}")

    # Set up OpenCV UI Windows
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1420, 880)
    create_trackbars(WINDOW_NAME)

    # Snap snapshot parameters for real-time reporting
    snap = {
        "ear": 0.0, 
        "ear_threshold": 0.0, 
        "mar": 0.0, 
        "perclos": 0.0, 
        "risk": 0.0, 
        "raw_risk": 0.0, 
        "cnn": 0.0, 
        "cnn_raw": 0.0
    }
    snap.update(
        blink_count=0, 
        yawn_count=0, 
        eye_closed_sec=0.0, 
        head_down=False, 
        is_yawning=False, 
        pitch=0.0,
        yaw=0.0,
        roll=0.0
    )
    
    debug = {}
    fps = 20.0
    prev_time = time.time()
    last_log_time = time.time()

    # Timing and frozen cache initialization for Feature 1 & 2
    last_detection_time = 0.0
    waiting_for_detection = False

    # Frozen state cache
    frozen_eye = {
        "ear": 0.0,
        "ear_threshold": 0.0,
        "eyes_closed": False,
        "perclos": 0.0,
        "blink_count": 0,
        "eye_closed_sec": 0.0,
        "raw_ear": 0.0,
        "baseline_ear": 0.0,
        "perclos_window_len": 0,
    }
    frozen_yawn = {
        "mar": 0.0,
        "baseline_mar": 0.20,
        "relative_mar": 0.0,
        "yawn_confidence": 0.0,
        "yawn_count": 0,
        "is_yawning": False,
        "yawn_detected_alert": False,
        "fatigue_yawn": False,
        "frequent_yawning": False,
        "repeated_yawning": False,
        "yawn_bonus": 0.0,
        "is_talking": False,
        "big_yawning": False,
        "last_yawn_sec_ago": None,
    }

    frozen_head = {
        "pitch": 0.0,
        "pitch_baseline": 0.0,
        "relative_pitch": 0.0,
        "head_down": False,
        "yaw": 0.0,
        "roll": 0.0,
        "any_tilt": False,
        "head_up": False,
        "head_left": False,
        "head_right": False,
        "head_tilt_left": False,
        "head_tilt_right": False,
        "calibrated": False,
    }
    frozen_cnn_smooth = 0.0
    frozen_cnn_raw = 0.0
    frozen_smooth_risk = 0.0
    frozen_calibrated = False
    frozen_state = "CALIBRATING"
    frozen_state_color = Colors.CYAN
    frozen_attention = 100.0
    frozen_ui_alerts = []
    frozen_box = (0, 0, 0, 0)
    frozen_landmarks = None
    frozen_face_lost = True
    frozen_debug = {}
    frozen_signals = {
        "eyes_closed": False,
        "eye_closed_sec": 0.0,
        "head_down": False,
        "any_tilt": False,
        "yawn_fatigue": False,
        "big_yawning": False,
        "smooth_risk": 0.0,
        "raw_risk": 0.0,
        "perclos": 0.0,
        "face_lost": True,
    }

    # Safe Defaults Initialization (Control-Flow & Safe Defaults Audit)
    cnn_score = 0.0
    cnn_raw_score = 0.0
    cnn_smooth_score = 0.0
    cnn_smooth = 0.0
    cnn_raw = 0.0

    ear = 0.0
    mar = 0.0
    perclos = 0.0

    risk = 0.0
    raw_risk = 0.0
    smooth_risk = 0.0

    head_down = False
    eyes_closed = False
    yawning = False
    is_yawning = False

    pitch = 0.0
    yaw = 0.0
    roll = 0.0

    attention = 100.0
    attention_score = 100.0

    calibrated = False
    state = "CALIBRATING"
    state_color = Colors.CYAN
    ui_alerts = []
    ui_alerts_list = []

    signals = dict(frozen_signals)
    eye = dict(frozen_eye)
    yawn = dict(frozen_yawn)
    head = dict(frozen_head)

    print("\n" + "="*80)
    print("                    DMS REAL-TIME PIPELINE IS NOW ACTIVE")
    print("                    Instructions: Press ESC inside window to exit")
    print("="*80 + "\n")

    # ==========================================================================
    # 9. PIPELINE LOOP
    # ==========================================================================
    while True:
        loop_start = time.time()
        
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[INFO] Camera source stream ended.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        fps = 0.92 * fps + 0.08 / max(now - prev_time, 1e-6)
        prev_time = now

        # Read sliders
        activation_speed, detection_fps, alarm_duration = read_trackbars(WINDOW_NAME)
        # Update speed gate active status using user-defined activation speed limit
        active = speed_gate.update(activation_speed)

        # Interval Logic Check to decide whether we bypass CNN inference
        current_time = time.time()
        if not active:
            last_detection_time = 0.0
            waiting_for_detection = False
        else:
            if detection_fps == 0:
                waiting_for_detection = False
            else:
                # detection_fps represents Hz (Evaluations per second).
                # Convert FPS to interval in seconds (e.g., 5 FPS -> 0.2s interval)
                detection_interval = 1.0 / float(detection_fps)
                if last_detection_time == 0.0 or (current_time - last_detection_time) >= detection_interval:
                    waiting_for_detection = False
                    last_detection_time = current_time
                else:
                    waiting_for_detection = True

        frame = adjust_brightness(frame)

        # Build Status strings
        status_str = "ACTIVE" if active else "INACTIVE"

        if not active:
            # SYSTEM INACTIVE
            state = "INACTIVE"
            state_color = Colors.RED
            ui_alerts = []
            ui_alerts_list = []
            attention = 100.0
            attention_score = 100.0
            
            # Reset last detection time so it runs immediately when active again
            last_detection_time = 0.0
            
            # Clear debug values when inactive
            debug = {}

            # Safe default variable assignments when inactive
            cnn_score = 0.0
            cnn_raw_score = 0.0
            cnn_smooth_score = 0.0
            cnn_smooth = 0.0
            cnn_raw = 0.0
            ear = 0.0
            mar = 0.0
            perclos = 0.0
            risk = 0.0
            raw_risk = 0.0
            smooth_risk = 0.0
            head_down = False
            eyes_closed = False
            yawning = False
            is_yawning = False
            pitch = 0.0
            yaw = 0.0
            roll = 0.0
            calibrated = False
            
            # Reset snap values when inactive
            snap.update(
                ear=0.0,
                ear_threshold=0.0,
                mar=0.0,
                perclos=0.0,
                risk=0.0,
                raw_risk=0.0,
                cnn=0.0,
                cnn_raw=0.0,
                blink_count=0,
                yawn_count=0,
                eye_closed_sec=0.0,
                head_down=False,
                is_yawning=False,
                pitch=0.0,
                yaw=0.0,
                roll=0.0
            )
            
            # Draw system deactivated warning
            cv2.putText(frame, "SYSTEM DEACTIVATED", (520, 120), cv2.FONT_HERSHEY_DUPLEX, 1.0, Colors.RED, 3, cv2.LINE_AA)
            cv2.putText(frame, "WAITING FOR SPEED", (520, 160), cv2.FONT_HERSHEY_DUPLEX, 1.0, Colors.RED, 3, cv2.LINE_AA)
        else:
            # SYSTEM ACTIVE
            current_time = time.time()
            
            # Evaluate every frame in the background to calculate drowsiness status continuously
            # Face Mesh detection stage
            t_stage = time.time()
            landmarks = face_detector.process(frame)
            telemetry.stage_latency["Face Detection"] = (time.time() - t_stage) * 1000.0

            # Gating: Check if face/eyes are detected
            face_lost = (landmarks is None or len(landmarks) < 468)

            yawn_alert = None
            eye_alert = None
            head_alert = None

            if not face_lost:
                # Draw facial mesh nodes lightly on face
                draw_face_mesh(frame, landmarks, step=5, color=Colors.CYAN)

                # A) Eye openness analysis (EAR)
                t_stage = time.time()
                eye = eye_analyzer.update(
                    [landmarks[i] for i in LEFT_EYE_INDICES],
                    [landmarks[i] for i in RIGHT_EYE_INDICES],
                    fps,
                )
                telemetry.stage_latency["Eye Analysis"] = (time.time() - t_stage) * 1000.0

                # B) 6-DOF geometric head pose estimation (Yaw, Pitch, Roll)
                t_stage = time.time()
                head = head_analyzer.update(
                    landmarks, w, h, eyes_closed=eye["eyes_closed"]
                )
                telemetry.stage_latency["Head Pose"] = (time.time() - t_stage) * 1000.0

                # C) Mouth Openness, Velocity & Multi-Cue Yawn analysis (MAR)
                t_stage = time.time()
                yawn = yawn_analyzer.update(
                    [landmarks[i] for i in MOUTH_INDICES],
                    w,
                    h,
                    fps,
                    ear=eye["ear"],
                    baseline_ear=eye["baseline_ear"],
                    pitch=head["pitch"],
                    relative_pitch=head["relative_pitch"],
                    yaw=head["yaw"],
                    roll=head["roll"],
                    cnn_score=cnn_smooth,
                    eyes_closed=eye["eyes_closed"],
                    current_time=current_time,
                )
                telemetry.stage_latency["Yawn Analysis"] = (time.time() - t_stage) * 1000.0

                # D) Deep CNN prediction
                face_crop, box = crop_face(frame, landmarks)
                if not waiting_for_detection:
                    t_stage = time.time()
                    cnn_smooth, cnn_raw = cnn.predict(face_crop)
                    telemetry.stage_latency["CNN Inference"] = (time.time() - t_stage) * 1000.0
                else:
                    cnn_smooth = frozen_cnn_smooth
                    cnn_raw = frozen_cnn_raw
                    telemetry.stage_latency["CNN Inference"] = 0.0

                # E) Multi-modal raw risk and smoothed risk calculation
                t_stage = time.time()
                smooth_risk = fusion.update(
                    ear=eye["ear"],
                    ear_threshold=eye["ear_threshold"],
                    perclos=eye["perclos"],
                    cnn_score=cnn_smooth,
                    eyes_closed=eye["eyes_closed"],
                    eye_closed_sec=eye["eye_closed_sec"],
                    head_down=head["head_down"],
                    yawn_fatigue=yawn["fatigue_yawn"],
                    yawn_bonus=yawn.get("yawn_bonus", 0.0),
                )
                telemetry.stage_latency["Risk Fusion"] = (time.time() - t_stage) * 1000.0

                calibrated = eye["calibrated"] and head["calibrated"]

                # Rotate and project 3D wireframe box centered on face nose
                nose_pt = landmarks[1]
                pose_box.draw_cube(frame, nose_pt, head["pitch"], head["yaw"], head["roll"], state_color)

                signals = {
                    "eyes_closed": eye["eyes_closed"],
                    "eye_closed_sec": eye["eye_closed_sec"],
                    "head_down": head["head_down"],
                    "any_tilt": head["any_tilt"],
                    "yawn_fatigue": yawn["fatigue_yawn"],
                    "frequent_yawning": yawn.get("frequent_yawning", False),
                    "repeated_yawning": yawn.get("repeated_yawning", False),
                    "big_yawning": yawn["big_yawning"],
                    "smooth_risk": smooth_risk,
                    "raw_risk": fusion.raw_risk,
                    "perclos": eye["perclos"],
                    "face_lost": False,
                }

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2, cv2.LINE_AA)

                head_alert = None
                if head["head_down"]:
                    head_alert = ("HEAD DOWN", Colors.ORANGE)
                elif head["any_tilt"]:
                    if head["head_up"]:
                        head_alert = ("LOOKING HIGH", Colors.ORANGE)
                    elif head["head_left"]:
                        head_alert = ("LOOKING LEFT", Colors.YELLOW)
                    elif head["head_right"]:
                        head_alert = ("LOOKING RIGHT", Colors.YELLOW)
                    elif head["head_tilt_left"] or head["head_tilt_right"]:
                        head_alert = ("HEAD TURN/TILT", Colors.PURPLE)

                eye_alert = ("EYES CLOSED", Colors.RED) if eye["eyes_closed"] else None

                # YAWNING DETECTED is an event notification (short timer), not a sticky state.
                # yawn_active may remain true while the mouth is still open after confirmation,
                # but the UI message must clear after YAWN_DISPLAY_SEC.
                yawn_alert = None
                if yawn.get("yawn_detected_alert", False):
                    if yawn.get("big_yawning", False):
                        yawn_alert = ("BIG YAWNING", Colors.MAGENTA)
                    else:
                        yawn_alert = ("YAWNING DETECTED", Colors.MAGENTA)
                
                ui_alerts_list = []
            else:
                # Face lost state
                calibrated = eye_analyzer.calibrated and head_analyzer.calibrated
                smooth_risk = fusion.update(
                    ear=eye_analyzer.raw_ear,
                    ear_threshold=eye_analyzer.ear_threshold,
                    perclos=eye_analyzer.perclos_window and float(np.mean(eye_analyzer.perclos_window)) or 0.0,
                    cnn_score=cnn.smoothed_score,
                    eyes_closed=False,
                    eye_closed_sec=0.0,
                    head_down=False,
                    yawn_fatigue=False,
                    yawn_bonus=0.0,
                )

                signals = {
                    "eyes_closed": False,
                    "eye_closed_sec": 0.0,
                    "head_down": False,
                    "any_tilt": False,
                    "yawn_fatigue": False,
                    "frequent_yawning": False,
                    "repeated_yawning": False,
                    "big_yawning": False,
                    "smooth_risk": smooth_risk,
                    "raw_risk": fusion.raw_risk,
                    "perclos": eye_analyzer.perclos_window and float(np.mean(eye_analyzer.perclos_window)) or 0.0,
                    "face_lost": True,
                }

                eye = {
                    "ear": eye_analyzer.raw_ear,
                    "ear_threshold": eye_analyzer.ear_threshold,
                    "eyes_closed": False,
                    "perclos": signals["perclos"],
                    "blink_count": eye_analyzer.blink_count,
                    "eye_closed_sec": 0.0,
                    "raw_ear": eye_analyzer.raw_ear,
                    "baseline_ear": eye_analyzer.baseline_ear,
                    "perclos_window_len": len(eye_analyzer.perclos_window),
                }
                yawn = {
                    "mar": yawn_analyzer.smoothed_mar or 0.0,
                    "baseline_mar": yawn_analyzer.baseline_mar,
                    "relative_mar": 0.0,
                    "yawn_confidence": 0.0,
                    "yawn_count": yawn_analyzer.yawn_count,
                    "is_yawning": False,
                    "yawn_detected_alert": False,
                    "fatigue_yawn": False,
                    "frequent_yawning": False,
                    "repeated_yawning": False,
                    "yawn_bonus": 0.0,
                    "is_talking": False,
                    "big_yawning": False,
                    "last_yawn_sec_ago": None,
                }
                head = {

                    "pitch": head_analyzer.pitch,
                    "pitch_baseline": head_analyzer.pitch_baseline,
                    "relative_pitch": 0.0,
                    "head_down": False,
                    "yaw": head_analyzer.yaw,
                    "roll": head_analyzer.roll,
                    "any_tilt": False,
                    "head_up": False,
                    "head_left": False,
                    "head_right": False,
                    "head_tilt_left": False,
                    "head_tilt_right": False,
                    "calibrated": False,
                }
                cnn_smooth, cnn_raw = cnn.smoothed_score, cnn.raw_score
                ui_alerts_list = [("NO FACE DETECTED", Colors.ORANGE)]

            # F) State Machine update
            t_stage = time.time()
            state_bg = state_machine.update(signals, system_calibrated=calibrated)
            state_color_bg = state_machine.state_color
            telemetry.stage_latency["State Machine"] = (time.time() - t_stage) * 1000.0

            # G) Evaluation Timing and Gating Window
            danger_for_eval = (
                calibrated
                and (
                    eye["eyes_closed"]
                    or head["head_down"]
                    or yawn["fatigue_yawn"]
                    or head["any_tilt"]
                )
            )
            eval_ok = alert_engine.update_eval(danger_for_eval, fps, DEFAULT_EVAL_SEC)

            # State-eligible alerting
            alert_active_bg = alert_engine.update_alert(
                state_machine.alert_eligible and eval_ok,
                fps,
                DEFAULT_ALERT_SEC,
                alarm_duration_sec=float(alarm_duration),
            )

            # MESSAGE PRIORITY FILTER:
            ui_alerts_list = []
            if state_bg == "MICROSLEEP":
                ui_alerts_list.append(("MICROSLEEP ALERT", Colors.RED))
            elif alert_active_bg or state_bg == "DROWSY":
                ui_alerts_list.append(("DROWSINESS ALERT", Colors.RED))
            elif state_bg == "FATIGUED":
                ui_alerts_list.append(("FATIGUED", Colors.ORANGE))

            if yawn_alert:
                ui_alerts_list.append(yawn_alert)
            elif eye_alert and state_bg not in ("MICROSLEEP", "DROWSY"):
                ui_alerts_list.append(eye_alert)
            elif head_alert and state_bg != "MICROSLEEP":
                ui_alerts_list.append(head_alert)
                ui_alerts_list.append(head_alert)

            # Continuous real-time UI & State reporting
            snap_eye = dict(eye)
            snap_yawn = dict(yawn)
            snap_head = dict(head)
            snap_cnn_smooth = cnn_smooth
            snap_cnn_raw = cnn_raw
            snap_smooth_risk = smooth_risk
            snap_calibrated = calibrated
            snap_state = state_bg
            snap_state_color = state_color_bg
            snap_attention = state_machine.attention.score
            snap_ui_alerts = list(ui_alerts_list)
            snap_signals = dict(signals)

            # Assign to main display variables
            eye = snap_eye
            yawn = snap_yawn
            head = snap_head
            cnn_smooth = snap_cnn_smooth
            cnn_raw = snap_cnn_raw
            cnn_score = cnn_smooth
            cnn_smooth_score = cnn_smooth
            cnn_raw_score = cnn_raw
            smooth_risk = snap_smooth_risk
            risk = smooth_risk
            calibrated = snap_calibrated
            state = snap_state
            state_color = snap_state_color
            attention = snap_attention
            attention_score = snap_attention
            ui_alerts = snap_ui_alerts
            signals = snap_signals
            ear = eye.get("ear", 0.0)
            mar = yawn.get("mar", 0.0)
            perclos = eye.get("perclos", 0.0)
            head_down = head.get("head_down", False)
            eyes_closed = eye.get("eyes_closed", False)
            is_yawning = yawn.get("is_yawning", False)
            yawning = is_yawning
            pitch = head.get("pitch", 0.0)
            yaw = head.get("yaw", 0.0)
            roll = head.get("roll", 0.0)

            # Draw calibration HUD if not calibrated
            if not calibrated:
                accumulated_frames = len(eye_analyzer._calibration_samples)
                calibration_hud.draw(frame, accumulated_frames, 45)
            else:
                # Update scrolling graphs and cockpit widgets
                graph_ear.add_value(eye["ear"])
                graph_mar.add_value(yawn["mar"])
                graph_risk.add_value(smooth_risk)
                graph_attention.add_value(attention)

                # Draw scrolling charts
                graph_x = w - 340
                graph_ear.draw(frame, graph_x, 100)
                graph_mar.draw(frame, graph_x, 240)
                graph_risk.draw(frame, graph_x, 380)
                graph_attention.draw(frame, graph_x, 520)

                # Draw gauges
                gauge_speed.draw(frame, 550, 100, speed_gate.current_speed)
                gauge_attention.draw(frame, 680, 100, attention)
                gauge_latency.draw(frame, 810, 100, telemetry.avg_latency)

            # Debug Overlay Payload — calibration + temporal yawn/eye diagnostics
            debug = {
                "raw_ear": eye["raw_ear"],
                "ear": eye["ear"],
                "baseline_ear": eye["baseline_ear"] or 0.0,
                "ear_close_thr": eye.get("ear_close_threshold", eye.get("ear_threshold", 0.0)),
                "ear_open_thr": eye.get("ear_open_threshold", 0.0),
                "eye_state": eye.get("eye_state", "CLOSED" if eye.get("eyes_closed") else "OPEN"),
                "blink_count": eye.get("blink_count", 0),
                "raw_mar": yawn.get("raw_mar", 0.0),
                "mar": yawn.get("mar", 0.0),
                "baseline_mar": yawn.get("baseline_mar", 0.20),
                "yawn_start_thr": yawn.get("yawn_start_threshold", 0.50),
                "yawn_end_thr": yawn.get("yawn_end_threshold", 0.38),
                "yawn_peak_mar": yawn.get("yawn_peak_mar", 0.0),
                "yawn_candidate_frames": yawn.get("open_frames", 0),
                "yawn_required_frames": yawn.get("required_frames", 0),
                "yawn_active": yawn.get("yawn_active", yawn.get("is_yawning", False)),
                "yawn_cooldown": yawn.get("yawn_cooldown", 0),
                "yawn_confidence": yawn.get("yawn_confidence", 0.0),
                "yawn_frames": yawn.get("open_frames", 0),
                "yawn_duration": yawn.get("open_duration_sec", 0.0),
                "yawn_state": yawn.get("yawn_state", "NO_YAWN"),
                "raw_risk": signals.get("raw_risk", 0.0),
                "smooth_risk": smooth_risk,
                "perclos": eye["perclos"],
                "perclos_n": eye["perclos_window_len"],
                "cnn_raw": cnn_raw,
                "cnn_smooth": cnn_smooth,
                "pitch": head["pitch"],
                "pitch_baseline": head["pitch_baseline"] or 0.0,
                "rel_pitch": head["relative_pitch"],
                "eyes_closed": eye["eyes_closed"],
                "eye_closed_sec": eye["eye_closed_sec"],
                "eval_frames": alert_engine.eval_counter.count,
                "eval_need": alert_engine.eval_counter.threshold,
                "fatigue_frames": state_machine.debug.get("fatigue_frames", 0),
                "drowsy_frames": state_machine.debug.get("drowsy_frames", 0),
                "micro_frames": state_machine.debug.get("micro_frames", 0),
                "exit_frames": state_machine.debug.get("exit_frames", 0),
                "alert_active": alert_engine.alert_active,
            }

            # Telemetry file logging once per second
            if now - last_log_time >= 1.0:
                last_log_time = now
                log_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "uptime": telemetry.system_uptime,
                    "state": state,
                    "ear": eye["ear"],
                    "mar": yawn["mar"],
                    "yawn_confidence": yawn.get("yawn_confidence", 0.0),
                    "pitch": head["pitch"],
                    "yaw": head.get("yaw", 0.0),
                    "roll": head.get("roll", 0.0),
                    "risk": smooth_risk,
                    "attention": attention,
                    "blink_count": eye["blink_count"],
                    "yawn_count": yawn["yawn_count"],
                    "last_yawn_sec": yawn.get("last_yawn_sec_ago", 0.0) if yawn.get("last_yawn_sec_ago") is not None else 0.0,
                    "alert_active": alert_engine.alert_active,
                    "latency": telemetry.avg_latency
                }
                threading.Thread(target=logger.queue_record, args=(log_data,), daemon=True).start()
                threading.Thread(target=logger.flush, daemon=True).start()

        # Build metrics HUD dashboard metrics array
        if active:
            # Snap records for HUD dashboard drawing
            snap.update(
                ear=eye["ear"],
                ear_threshold=eye["ear_threshold"],
                mar=yawn["mar"],
                yawn_confidence=yawn.get("yawn_confidence", 0.0),
                last_yawn_sec_ago=yawn.get("last_yawn_sec_ago"),
                perclos=eye["perclos"],
                risk=smooth_risk,
                raw_risk=signals.get("raw_risk", 0.0),
                cnn=cnn_smooth,
                cnn_raw=cnn_raw,
                pitch=head["pitch"],
                yaw=head.get("yaw", 0.0),
                roll=head.get("roll", 0.0),
                blink_count=eye["blink_count"],
                yawn_count=yawn["yawn_count"],
                eye_closed_sec=eye["eye_closed_sec"],
                head_down=head["head_down"],
                is_yawning=yawn["is_yawning"],
            )

        # Fetch rich speed status from SpeedManager
        try:
            status_info   = speed_provider.get_status_info()
        except Exception:
            status_info   = {}

        prov_name     = status_info.get("provider", "None")
        prov_status   = status_info.get("status", "SPEED UNAVAILABLE")
        raw_spd       = float(status_info.get("speed", 0.0) or 0.0)
        smooth_spd    = float(status_info.get("smoothed_speed", 0.0) or 0.0)
        threshold_kmh = float(status_info.get("threshold", 0.0) or 0.0)
        tolerance_kmh = float(status_info.get("tolerance", 0.0) or 0.0)
        eff_activate  = float(status_info.get("eff_activate", 0.0) or 0.0)
        gps_acc       = status_info.get("accuracy")

        # Speed display: show smoothed value when a valid source is active
        disconnected_states = {"RECONNECT", "INACTIVE"}
        speed_display = (
            f"{smooth_spd:.1f} km/h"
            if prov_name not in ("None",)
            else "N/A"
        )
        raw_display = f"{raw_spd:.1f} km/h"

        metrics = [
            # ── Speed Subsystem Block ─────────────────────────────────────────
            f"Provider      : {prov_name}",
            f"Current Speed : {speed_display}",
            f"Raw Speed     : {raw_display}",
            f"Threshold     : {threshold_kmh:.0f} km/h",
            f"Tolerance     : {tolerance_kmh:.0f} km/h",
            f"Eff Activate  : {eff_activate:.0f} km/h",
            f"Speed Status  : {prov_status}",
            f"DMS Status    : {status_str}",
            # ── Provider-specific info ─────────────────────────────────────────
        ]
        if prov_name == "GPS" and gps_acc:
            metrics.append(f"GPS Accuracy  : {gps_acc}")
        elif prov_name == "Mobile GPS":
            if gps_acc:
                metrics.append(f"GPS Accuracy  : {gps_acc}")
            # Show the server URL so user knows where to connect their phone
            try:
                mob_url = status_info.get("accuracy") or ""
                srv_url = speed_provider._mobile_gps_provider._server_url if speed_provider._mobile_gps_provider else ""
                if srv_url:
                    metrics.append(f"Phone URL     : {srv_url}")
            except Exception:
                pass
        elif prov_name == "OBD-II":
            metrics.append(f"OBD Port      : {gps_acc or 'Auto'}")
        elif prov_name == "OkDriver":
            metrics.append(f"OkDriver      : {gps_acc or 'Live'}")
        else:
            metrics.append(f"Conn Status   : Connected")

        last_yawn_sec = snap.get("last_yawn_sec_ago")
        last_yawn_str = f"{int(last_yawn_sec)}s ago" if last_yawn_sec is not None else "N/A"
        yawn_conf = snap.get("yawn_confidence", 0.0)

        metrics.extend([
            f"Detection Interval : {int(detection_fps)} sec",
            f"Alarm Duration : {int(alarm_duration)} sec",
            f"STATE      : {state}",
            f"EYES       : {'CLOSED' if debug.get('eyes_closed') else 'OPEN'}  ({snap.get('eye_closed_sec', 0.0):.1f}s)",
            f"RISK sm/raw: {snap.get('risk', 0.0):.2f} / {snap.get('raw_risk', 0.0):.2f}",
            f"EAR / THR  : {snap.get('ear', 0.0):.2f} / {snap.get('ear_threshold', 0.0):.2f}",
            f"PERCLOS    : {snap.get('perclos', 0.0):.2f}",
            f"MAR        : {snap.get('mar', 0.0):.2f}",
            f"YAWN CONF  : {yawn_conf:.2f}",
            f"LAST YAWN  : {last_yawn_str}",
            f"CNN sm/raw : {snap.get('cnn', 0.0):.2f} / {snap.get('cnn_raw', 0.0):.2f}",
            f"ATTENTION  : {int(attention)}%",
            f"PITCH/Y/R  : {snap.get('pitch', 0.0):.1f}/{snap.get('yaw', 0.0):.1f}/{snap.get('roll', 0.0):.1f}",
            f"BLINKS     : {snap.get('blink_count', 0)}",
            f"YAWNS      : {snap.get('yawn_count', 0)}",
            f"FPS        : {int(fps)}",
            f"EVAL/ALERT : {DEFAULT_EVAL_SEC}s / {DEFAULT_ALERT_SEC}s",
            f"CSV LOGS   : {logger.records_written} recs",
        ])


        # Draw sleek UI Status HUD panel
        draw_status_panel(frame, state, state_color, metrics, snap.get("risk", 0.0), attention)
        
        # Draw text diagnostics debug report list
        draw_debug_overlay(frame, debug)

        # Draw warnings HUD cards
        if ui_alerts:
            draw_alerts(frame, ui_alerts)

        # Write frame to video recording
        writer.write(frame)

        # Render display frame
        cv2.imshow(WINDOW_NAME, frame)

        # Log system latency benchmarks
        latency_ms = (time.time() - loop_start) * 1000.0
        telemetry.add_latency(latency_ms)
        telemetry.frame_count += 1

        # Periodic logging of telemetry (every 120 frames)
        if telemetry.frame_count % 120 == 0:
            uptime = telemetry.system_uptime
            avg_lat = telemetry.avg_latency
            print(f"[TELEMETRY] Frames: {telemetry.frame_count:05d} | Uptime: {uptime:.1f}s | Latency: {avg_lat:.2f}ms | Avg FPS: {int(fps)}")
            for stage, l_ms in telemetry.stage_latency.items():
                print(f"   -> {stage:<16} : {l_ms:6.2f} ms")

        # Capture keys and pass them to the speed provider
        key = cv2.waitKey(1)
        if key != -1:
            speed_provider.handle_key(key)
        
        # Capture exit interrupt key ESC
        if (key & 0xFF) == 27:
            print("[INFO] ESC interrupt detected.")
            break

    # ==========================================================================
    # 10. CLEANUP AND TERMINATION
    # ==========================================================================
    print("\n" + "="*80)
    print("                     TERMINATING PIPELINE STREAM")
    print("="*80)
    speed_provider.stop() # Stop the background acquisition thread
    cap.release()
    writer.release()
    face_detector.close()
    cv2.destroyAllWindows()
    print("[INFO] Release handles closed. Success.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt detected. Exiting.")
