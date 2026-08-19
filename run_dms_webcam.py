#!/usr/bin/env python3
# ==============================================================================
#            USB WEBCAM DMS (DRIVER MONITORING SYSTEM) - DASHCAM VIEW
# ==============================================================================
# Purpose: Capture USB webcam feed, run DMS drowsiness detection pipeline,
#          and display dashcam-style view with real-time drowsiness status.
# Usage:   python run_dms_webcam.py
#          Press 'q' or ESC to exit.
# ==============================================================================

import os
import sys
import time
import math
import warnings
import subprocess
import platform
from collections import deque

import cv2
import numpy as np

warnings.filterwarnings("ignore")

DMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1108_dms_f")
if DMS_DIR not in sys.path:
    sys.path.insert(0, DMS_DIR)

from config import (
    DEFAULT_ALERT_SEC,
    DEFAULT_EVAL_SEC,
    FRAME_H,
    FRAME_W,
    LEFT_EYE_INDICES,
    MODEL_PATH,
    MOUTH_INDICES,
    OUTPUT_DIR,
    OUTPUT_FPS,
    RIGHT_EYE_INDICES,
    WINDOW_NAME,
    Colors,
)
from pipeline.alert_engine import AlertEngine
from pipeline.behavioral import EyeAnalyzer, HeadPoseAnalyzer, YawnAnalyzer
from pipeline.cnn_predictor import CnnPredictor
from pipeline.detection import FaceDetector, crop_face
from pipeline.fusion_engine import FusionEngine
from pipeline.state_machine import DriverStateMachine
from utils.brightness import adjust_brightness
from utils.visualization import (
    draw_alerts,
    draw_face_mesh,
    draw_status_panel,
)

OUTPUT_DIR_FULL = os.path.join(DMS_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_DIR_FULL, exist_ok=True)


_OS = platform.system()


def _build_backend_list():
    """Return list of (name, backend_id) to try, OS-specific ordering."""
    if _OS == "Darwin":
        return [
            ("CAP_ANY (auto-detect)", cv2.CAP_ANY),
            ("CAP_AVFOUNDATION (Mac native)", cv2.CAP_AVFOUNDATION),
        ]
    if _OS == "Windows":
        return [
            ("CAP_DSHOW (DirectShow USB)", cv2.CAP_DSHOW),
            ("CAP_ANY (auto-detect)", cv2.CAP_ANY),
            ("CAP_MSMF (MediaFoundation)",
             getattr(cv2, "CAP_MSMF", 1400)),
        ]
    return [
        ("CAP_ANY (auto-detect)", cv2.CAP_ANY),
        ("CAP_V4L2 (V4L2 Linux)", getattr(cv2, "CAP_V4L2", 200)),
    ]


BACKENDS_TO_TRY = _build_backend_list()


def _try_open_with_retries(index: int, backend_name: str, backend_id: int,
                           retries: int = 4, sleep_between: float = 0.7):
    """Try to open camera + probe a frame (Mac TCC dialog may take seconds)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            cap = cv2.VideoCapture(index, backend_id)
        except Exception as e:
            last_err = f"VideoCapture exception: {e}"
            time.sleep(sleep_between)
            continue
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            last_err = "cap.isOpened()=False"
            time.sleep(sleep_between)
            continue
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            last_err = f"opened but cap.read()={getattr(frame, 'shape', None)}"
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(sleep_between)
            continue
        return cap, attempt, None
    return None, retries, last_err


def _run_mac_diagnostics():
    """On Mac, tell the user exactly what's connected + TCC fix."""
    if _OS != "Darwin":
        return
    print("\n" + "=" * 72)
    print("  🍎  MAC CAMERA DIAGNOSTIC (running)")
    print("=" * 72)
    print("\n--- system_profiler SPCameraDataType ---")
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.STDOUT, text=True, timeout=15)
        print(out)
    except Exception as e:
        print(f"  (couldn't run system_profiler: {e})")

    print("--- TCC PERMISSION FIX (run THESE in Terminal if camera is STATUS 0): ---")
    print("  1. Reset Camera permission DB:")
    print("       tccutil reset Camera")
    print("  2. QUIT Terminal/IDE fully (Cmd+Q), reopen project folder.")
    print("  3. Run: python3 test_webcam.py")
    print("       → Click ALLOW when Mac prompts 'Camera access'.")
    print("  4. Verify toggle:")
    print("       System Settings → Privacy & Security → Camera")
    print("       → ON for Terminal / your IDE (VS Code/PyCharm/iTerm)")
    print()


def probe_webcam():
    """Ultra-robust USB webcam finder. Returns (cap, index, backend_name)."""
    print("\n" + "=" * 72)
    print("  📷  ROBUST USB WEBCAM PROBER (Quantron QPC-1020 HD ready)")
    print("=" * 72)
    print(f"  OS           : {_OS}")
    print(f"  OpenCV       : {cv2.__version__}")
    print(f"  Backends     : {[b[0] for b in BACKENDS_TO_TRY]}")

    if _OS == "Darwin":
        _run_mac_diagnostics()

    max_idx = 9
    # Pass 1: OpenCV default backend (no explicit backend flag)
    print(f"\n--- Pass 1/2: OpenCV default backend, indices 0..{max_idx} ---")
    for i in range(max_idx + 1):
        try:
            cap = cv2.VideoCapture(i)
        except Exception:
            cap = None
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"  ✅ PASS1 idx={i} DEFAULT backend shape={frame.shape}")
                return cap, i, "OpenCV_default"
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # Pass 2: Explicit backends in OS-preferred order, with retries per combo
    print(f"\n--- Pass 2/2: Explicit backends with retries, indices 0..{max_idx} ---")
    for bname, bid in BACKENDS_TO_TRY:
        for i in range(max_idx + 1):
            print(f"  Probing idx={i:2d}  {bname} ...", flush=True, end=" ")
            cap, attempts, err = _try_open_with_retries(i, bname, bid)
            if cap is not None:
                ok, frame = cap.read()
                shape = getattr(frame, "shape", None) if ok else None
                print(f"✅ WORKING  attempts={attempts}  shape={shape}")
                return cap, i, bname
            print(f"❌ fail  ({err})")

    # ---- No camera found ---------------------------------------------------
    print("\n" + "=" * 72)
    print("  ❌  NO CAMERA COULD BE OPENED.")
    print("=" * 72)
    if _OS == "Darwin":
        print("""
 🍎 TROUBLESHOOTING (copy-paste into Terminal):

   Step A: Reset TCC so Mac will PROMPT you again:
       tccutil reset Camera

   Step B: QUIT & RE-OPEN Terminal/IDE (Cmd+Q), then try:
       python3 test_webcam.py
       → Click ALLOW on the permission dialog!

   Step C: If still fails, check USB enumeration:
       system_profiler SPCameraDataType
       Expected: "Quantron QPC-1020 HD" or "USB Video Class Video"
       If missing: try different USB port / cable / hub (direct port preferred)
""")
    sys.exit(1)


def draw_simple_hud(frame, state, state_color, ear, mar, cnn_score, risk, attention,
                    eyes_closed, is_yawning, head_down, calibrated, fps, face_lost):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)

    cv2.putText(frame, "MARUTI DMS - DASHCAM VIEW", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, Colors.CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, Colors.WHITE, 1, cv2.LINE_AA)

    status_label = state
    if not calibrated:
        status_label = "CALIBRATING..."
    elif face_lost:
        status_label = "NO FACE"

    cv2.putText(frame, f"STATUS: {status_label}", (20, 68),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, state_color, 2, cv2.LINE_AA)

    panel_x, panel_y = 20, 100
    panel_w, panel_h = 380, 280
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, dst=frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), Colors.CYAN, 2, cv2.LINE_AA)

    y = panel_y + 30
    cv2.putText(frame, "--- DMS METRICS ---", (panel_x + 20, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, Colors.YELLOW, 1, cv2.LINE_AA)

    y += 28
    eye_color = Colors.RED if eyes_closed else Colors.GREEN
    eye_status = "CLOSED" if eyes_closed else "OPEN"
    cv2.putText(frame, f"EYES: {eye_status}", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"EAR: {ear:.3f}", (panel_x + 190, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)

    y += 25
    mouth_color = Colors.MAGENTA if is_yawning else Colors.WHITE
    mouth_status = "YAWNING" if is_yawning else "NORMAL"
    cv2.putText(frame, f"MOUTH: {mouth_status}", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"MAR: {mar:.3f}", (panel_x + 190, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)

    y += 25
    head_color = Colors.ORANGE if head_down else Colors.GREEN
    head_status = "HEAD DOWN" if head_down else "NORMAL"
    cv2.putText(frame, f"HEAD: {head_status}", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_color, 2, cv2.LINE_AA)

    y += 30
    cv2.putText(frame, f"CNN Drowsy: {cnn_score * 100:.1f}%", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)

    y += 22
    cv2.putText(frame, "FATIGUE RISK:", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)
    bar_x = panel_x + 140
    cv2.rectangle(frame, (bar_x, y - 13), (bar_x + 200, y + 2), (50, 50, 50), -1)
    fill = int(200 * min(1.0, risk))
    r_color = Colors.GREEN if risk < 0.3 else (Colors.YELLOW if risk < 0.6 else Colors.RED)
    cv2.rectangle(frame, (bar_x, y - 13), (bar_x + fill, y + 2), r_color, -1)
    cv2.rectangle(frame, (bar_x, y - 13), (bar_x + 200, y + 2), Colors.WHITE, 1)
    cv2.putText(frame, f"{risk * 100:.0f}%", (bar_x + 210, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_color, 1, cv2.LINE_AA)

    y += 25
    cv2.putText(frame, "ATTENTION:", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Colors.WHITE, 1, cv2.LINE_AA)
    att_fill = int(200 * min(1.0, attention / 100.0))
    att_color = Colors.GREEN if attention > 60 else (Colors.YELLOW if attention > 30 else Colors.RED)
    cv2.rectangle(frame, (bar_x, y - 13), (bar_x + 200, y + 2), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, y - 13), (bar_x + att_fill, y + 2), att_color, -1)
    cv2.rectangle(frame, (bar_x, y - 13), (bar_x + 200, y + 2), Colors.WHITE, 1)
    cv2.putText(frame, f"{attention:.0f}%", (bar_x + 210, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, att_color, 1, cv2.LINE_AA)

    y += 30
    cal_color = Colors.ORANGE if not calibrated else Colors.GREEN
    cal_status = "PLEASE STAY STILL" if not calibrated else "READY"
    cv2.putText(frame, f"CALIBRATION: {cal_status}", (panel_x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cal_color, 2, cv2.LINE_AA)


def draw_calibration_screen(frame, cal_frames, target_frames=45):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    pct = min(1.0, cal_frames / target_frames)
    cv2.circle(frame, (cx, cy), 100, (40, 40, 40), 8, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (100, 100), -90, 0, int(360 * pct),
                Colors.CYAN, 8, cv2.LINE_AA)
    cv2.putText(frame, f"{int(pct * 100)}%", (cx - 35, cy + 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, Colors.WHITE, 2, cv2.LINE_AA)

    cv2.putText(frame, "CALIBRATING DMS SYSTEM", (cx - 200, cy - 180),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, Colors.CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, "PLEASE KEEP EYES OPEN & LOOK STRAIGHT", (cx - 260, cy - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.YELLOW, 2, cv2.LINE_AA)


def draw_3d_head_box(frame, landmarks, pitch, yaw, roll, color=Colors.CYAN):
    if landmarks is None:
        return
    cx, cy = landmarks[1]
    size = 70.0

    def rot_mat(p, y, r):
        p, y, r = math.radians(p), math.radians(y), math.radians(r)
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        return Rz @ Ry @ Rx

    R = rot_mat(pitch, yaw, roll)
    cube = np.array([[-size, -size, -size], [size, -size, -size],
                     [size, size, -size], [-size, size, -size],
                     [-size, -size, size], [size, -size, size],
                     [size, size, size], [-size, size, size]], dtype=np.float32)
    proj = []
    for pt in cube:
        rx, ry, rz = R @ pt
        depth = rz + 300.0
        scale = 500.0 / depth
        proj.append((int(cx + rx * scale), int(cy + ry * scale)))
    lines = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    overlay = frame.copy()
    for s, e in lines:
        cv2.line(overlay, proj[s], proj[e], color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)


def main():
    print("\n" + "=" * 70)
    print("  MARUTI USB WEBCAM DMS (DRIVER MONITORING SYSTEM)")
    print("=" * 70)

    print("\n[1/6] Loading CNN Drowsiness Predictor...")
    cnn = CnnPredictor(MODEL_PATH)

    print("[2/6] Loading Face Mesh Detector (MediaPipe)...")
    face_detector = FaceDetector()

    print("[3/6] Loading Behavioral Analyzers (Eye / Yawn / Head)...")
    eye_analyzer = EyeAnalyzer()
    yawn_analyzer = YawnAnalyzer()
    head_analyzer = HeadPoseAnalyzer()

    print("[4/6] Loading Fusion Engine & State Machine...")
    fusion = FusionEngine()
    state_machine = DriverStateMachine()
    alert_engine = AlertEngine()

    camera_idx = probe_webcam()

    print(f"[5/6] Opening USB Webcam (index {camera_idx})...")
    cap, backend_used = _open_camera(camera_idx)
    if cap is None or not cap.isOpened():
        print(f"ERROR: Could not open camera at index {camera_idx}. Check USB cable and camera permissions.")
        if sys.platform == "darwin":
            print("  [MAC TIP] Go to System Settings → Privacy & Security → Camera → Enable Terminal/your IDE")
        return
    print(f"[INFO] Camera backend: {backend_used}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    writer_path = os.path.join(OUTPUT_DIR_FULL, "dms_webcam_output.avi")
    writer = cv2.VideoWriter(
        writer_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        OUTPUT_FPS,
        (FRAME_W, FRAME_H)
    )
    print(f"[INFO] Output recording: {writer_path}")

    WINDOW = "MARUTI DMS - DASHCAM (press q/ESC to exit)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    fps = 15.0
    prev_time = time.time()
    frame_count = 0

    ear = 0.0
    mar = 0.0
    cnn_smooth = 0.0
    cnn_raw = 0.0
    smooth_risk = 0.0
    attention = 100.0
    state = "CALIBRATING"
    state_color = Colors.CYAN
    ui_alerts = []
    calibrated = False
    face_lost = True
    eyes_closed = False
    is_yawning = False
    head_down = False

    print("\n[6/6] System Ready! Starting DMS pipeline...")
    print("=" * 70)
    print("  Instructions:")
    print("  - Keep eyes open & look straight for ~2 seconds (calibration)")
    print("  - Press 'q' or ESC key to exit")
    print("=" * 70 + "\n")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[INFO] Camera feed lost. Exiting...")
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now
        frame_count += 1

        frame = adjust_brightness(frame)

        landmarks = face_detector.process(frame)
        face_lost = (landmarks is None or len(landmarks) < 468)

        if not face_lost:
            draw_face_mesh(frame, landmarks, step=5, color=Colors.CYAN)

            eye = eye_analyzer.update(
                [landmarks[i] for i in LEFT_EYE_INDICES],
                [landmarks[i] for i in RIGHT_EYE_INDICES],
                fps,
            )
            head = head_analyzer.update(
                landmarks, w, h, eyes_closed=eye["eyes_closed"]
            )
            yawn = yawn_analyzer.update(
                [landmarks[i] for i in MOUTH_INDICES],
                w, h, fps,
                ear=eye["ear"],
                baseline_ear=eye["baseline_ear"],
                pitch=head["pitch"],
                relative_pitch=head["relative_pitch"],
                yaw=head["yaw"],
                roll=head["roll"],
                cnn_score=cnn_smooth,
                eyes_closed=eye["eyes_closed"],
                current_time=now,
            )

            face_crop, box = crop_face(frame, landmarks)
            if face_crop is not None and face_crop.size > 0:
                cnn_smooth, cnn_raw = cnn.predict(face_crop)

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

            calibrated = eye["calibrated"] and head["calibrated"]
            draw_3d_head_box(frame, landmarks, head["pitch"], head["yaw"], head["roll"], state_color)

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

            ear = eye["ear"]
            mar = yawn["mar"]
            eyes_closed = eye["eyes_closed"]
            is_yawning = yawn["is_yawning"] or yawn.get("big_yawning", False)
            head_down = head["head_down"]

            danger = (calibrated and (eyes_closed or head_down or
                                      yawn["fatigue_yawn"] or head["any_tilt"]))
            eval_ok = alert_engine.update_eval(danger, fps, DEFAULT_EVAL_SEC)

            state_bg = state_machine.update(signals, system_calibrated=calibrated)
            state_color_bg = state_machine.state_color
            attention = state_machine.attention.score

            alert_active = alert_engine.update_alert(
                state_machine.alert_eligible and eval_ok,
                fps, DEFAULT_ALERT_SEC, alarm_duration_sec=3.0,
            )

            ui_alerts = []
            if state_bg == "MICROSLEEP":
                ui_alerts.append(("MICROSLEEP ALERT", Colors.RED))
            elif alert_active or state_bg == "DROWSY":
                ui_alerts.append(("DROWSINESS DETECTED", Colors.RED))
            elif state_bg == "FATIGUED":
                ui_alerts.append(("DRIVER FATIGUED", Colors.ORANGE))

            if yawn.get("yawn_detected_alert", False):
                if yawn.get("big_yawning", False):
                    ui_alerts.append(("BIG YAWN", Colors.MAGENTA))
                else:
                    ui_alerts.append(("YAWNING", Colors.MAGENTA))
            elif eyes_closed and state_bg not in ("MICROSLEEP", "DROWSY"):
                ui_alerts.append(("EYES CLOSED", Colors.RED))
            elif head["head_down"] and state_bg != "MICROSLEEP":
                ui_alerts.append(("HEAD DOWN", Colors.ORANGE))

            state = state_bg
            state_color = state_color_bg
        else:
            calibrated = eye_analyzer.calibrated and head_analyzer.calibrated
            smooth_risk = fusion.raw_risk
            ui_alerts = [("NO FACE DETECTED", Colors.ORANGE)]
            eyes_closed = False
            is_yawning = False
            head_down = False

        if not calibrated:
            cal_count = len(eye_analyzer._calibration_samples)
            draw_calibration_screen(frame, cal_count, 45)
        else:
            draw_simple_hud(
                frame, state, state_color, ear, mar, cnn_smooth, smooth_risk,
                attention, eyes_closed, is_yawning, head_down, calibrated, fps, face_lost
            )
            draw_alerts(frame, ui_alerts)

        writer.write(frame)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("\n[INFO] User exit requested.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Session recording saved: {writer_path}")
    print(f"[DONE] Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
