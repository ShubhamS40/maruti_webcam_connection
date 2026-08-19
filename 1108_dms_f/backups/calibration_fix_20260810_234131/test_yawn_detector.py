"""
Verification Test Script for Multi-Stage Behavior-Aware Yawning Detector.
Tests all 10 required test cases specified in the requirements.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.behavioral import YawnAnalyzer
from pipeline.state_machine import DriverStateMachine


def create_dummy_mouth_pts(mar_val):
    """
    Creates dummy 6 landmark points that produce approximate MAR = mar_val.
    MAR = (dist(top1, bottom1) + dist(top2, bottom2)) / (2 * dist(left, right))
    Let dist(left, right) = 100. So vertical = 100 * mar_val.
    """
    left = (0.0, 0.0)
    right = (100.0, 0.0)
    vert = 100.0 * mar_val
    half_v = vert / 2.0
    top1 = (30.0, -half_v)
    top2 = (70.0, -half_v)
    bottom1 = (30.0, half_v)
    bottom2 = (70.0, half_v)
    return [left, top1, top2, right, bottom2, bottom1]


def run_test_cases():
    print("\n" + "=" * 80)
    print("      RUNNING YAWNING DETECTOR COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 80 + "\n")

    fps = 20.0
    num_yawn_frames = int(1.5 * fps) + 15 # 45 frames (~2.25s) to account for EMA smoothing ramp-up

    passed_count = 0
    total_count = 0

    # --------------------------------------------------------------------------
    # Test 1: Eyes Closed Only -> NO YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    mouth_closed = create_dummy_mouth_pts(0.20)
    t = time.time()
    for _ in range(num_yawn_frames):
        res = yawn_analyzer.update(
            mouth_closed, 640, 480, fps,
            ear=0.10, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.90, eyes_closed=True, current_time=t
        )
        t += 0.05

    t1_ok = (not res["is_yawning"]) and (res["yawn_state"] != "CONFIRMED_YAWN")
    print(f"[TEST 1] Eyes Closed Only -> NO YAWNING : {'PASSED' if t1_ok else 'FAILED'} (is_yawning={res['is_yawning']}, state={res['yawn_state']})")
    if t1_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 2: Real Yawning -> YES YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    mouth_open = create_dummy_mouth_pts(0.90)
    t = time.time()
    for i in range(num_yawn_frames):
        res = yawn_analyzer.update(
            mouth_open, 640, 480, fps,
            ear=0.25, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.80, eyes_closed=False, current_time=t
        )
        t += 0.05

    t2_ok = res["is_yawning"] and (res["yawn_state"] == "CONFIRMED_YAWN" or res["yawn_detected_alert"])
    print(f"[TEST 2] Real Yawning -> YES YAWNING : {'PASSED' if t2_ok else 'FAILED'} (is_yawning={res['is_yawning']}, state={res['yawn_state']}, conf={res['yawn_confidence']:.2f})")
    if t2_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 3: Talking -> NO YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    t = time.time()
    for i in range(num_yawn_frames):
        mar_val = 0.70 if (i % 2 == 0) else 0.30
        pts = create_dummy_mouth_pts(mar_val)
        res = yawn_analyzer.update(
            pts, 640, 480, fps,
            ear=0.30, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.10, eyes_closed=False, current_time=t
        )
        t += 0.05

    t3_ok = (not res["is_yawning"]) and (res["yawn_state"] != "CONFIRMED_YAWN")
    print(f"[TEST 3] Talking -> NO YAWNING : {'PASSED' if t3_ok else 'FAILED'} (is_yawning={res['is_yawning']}, is_talking={res['is_talking']}, state={res['yawn_state']})")
    if t3_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 4: Smiling -> NO YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    smile_pts = create_dummy_mouth_pts(0.40)
    t = time.time()
    for _ in range(num_yawn_frames):
        res = yawn_analyzer.update(
            smile_pts, 640, 480, fps,
            ear=0.30, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.05, eyes_closed=False, current_time=t
        )
        t += 0.05

    t4_ok = not res["is_yawning"]
    print(f"[TEST 4] Smiling -> NO YAWNING : {'PASSED' if t4_ok else 'FAILED'} (is_yawning={res['is_yawning']})")
    if t4_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 5: Laughing -> NO YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    t = time.time()
    for i in range(num_yawn_frames):
        mar_val = 0.65 if (i % 3 == 0) else 0.35
        pts = create_dummy_mouth_pts(mar_val)
        res = yawn_analyzer.update(
            pts, 640, 480, fps,
            ear=0.25, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.10, eyes_closed=False, current_time=t
        )
        t += 0.05

    t5_ok = not res["is_yawning"]
    print(f"[TEST 5] Laughing -> NO YAWNING : {'PASSED' if t5_ok else 'FAILED'} (is_yawning={res['is_yawning']})")
    if t5_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 6: Chewing -> NO YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    t = time.time()
    for i in range(num_yawn_frames):
        mar_val = 0.55 if (i % 4 == 0) else 0.25
        pts = create_dummy_mouth_pts(mar_val)
        res = yawn_analyzer.update(
            pts, 640, 480, fps,
            ear=0.30, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.10, eyes_closed=False, current_time=t
        )
        t += 0.05

    t6_ok = not res["is_yawning"]
    print(f"[TEST 6] Chewing -> NO YAWNING : {'PASSED' if t6_ok else 'FAILED'} (is_yawning={res['is_yawning']})")
    if t6_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 7: Mouth Slightly Open -> NO YAWNING
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    slight_pts = create_dummy_mouth_pts(0.50)
    t = time.time()
    for _ in range(num_yawn_frames):
        res = yawn_analyzer.update(
            slight_pts, 640, 480, fps,
            ear=0.30, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0,
            yaw=0.0, roll=0.0, cnn_score=0.10, eyes_closed=False, current_time=t
        )
        t += 0.05

    t7_ok = not res["is_yawning"]
    print(f"[TEST 7] Mouth Slightly Open -> NO YAWNING : {'PASSED' if t7_ok else 'FAILED'} (is_yawning={res['is_yawning']})")
    if t7_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 8: Repeated Real Yawns -> YES (Count >= 2, repeated_yawning True)
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    mouth_open = create_dummy_mouth_pts(0.90)
    mouth_closed = create_dummy_mouth_pts(0.20)
    t = time.time()

    # Yawn 1 (45 frames)
    for _ in range(num_yawn_frames):
        yawn_analyzer.update(mouth_open, 640, 480, fps, ear=0.25, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0, yaw=0.0, roll=0.0, cnn_score=0.80, eyes_closed=False, current_time=t)
        t += 0.05

    # Pause 2s (40 frames)
    for _ in range(40):
        yawn_analyzer.update(mouth_closed, 640, 480, fps, ear=0.30, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0, yaw=0.0, roll=0.0, cnn_score=0.10, eyes_closed=False, current_time=t)
        t += 0.05

    # Yawn 2 (45 frames)
    for _ in range(num_yawn_frames):
        res = yawn_analyzer.update(mouth_open, 640, 480, fps, ear=0.25, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0, yaw=0.0, roll=0.0, cnn_score=0.80, eyes_closed=False, current_time=t)
        t += 0.05

    t8_ok = (res["yawn_count"] >= 2) and res["repeated_yawning"]
    print(f"[TEST 8] Repeated Real Yawns -> YES : {'PASSED' if t8_ok else 'FAILED'} (yawn_count={res['yawn_count']}, repeated={res['repeated_yawning']})")
    if t8_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 9: Eyes Closed + Mouth Closed -> Drowsy/Eyes Closed Only (NO Yawning)
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    state_machine = DriverStateMachine()
    mouth_closed = create_dummy_mouth_pts(0.20)
    t = time.time()

    for i in range(num_yawn_frames):
        res_y = yawn_analyzer.update(mouth_closed, 640, 480, fps, ear=0.10, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0, yaw=0.0, roll=0.0, cnn_score=0.90, eyes_closed=True, current_time=t)
        t += 0.05
        closed_sec = (i + 1) * 0.05
        signals = {
            "eyes_closed": True,
            "eye_closed_sec": closed_sec,
            "head_down": False,
            "any_tilt": False,
            "yawn_fatigue": res_y["fatigue_yawn"],
            "big_yawning": False,
            "smooth_risk": 0.80,
            "raw_risk": 0.80,
            "perclos": 0.50,
            "face_lost": False,
        }
        state = state_machine.update(signals, system_calibrated=True)

    t9_ok = (not res_y["is_yawning"]) and (state in ("DROWSY", "MICROSLEEP", "FATIGUED"))
    print(f"[TEST 9] Eyes Closed + Mouth Closed -> Drowsy Only : {'PASSED' if t9_ok else 'FAILED'} (state={state}, is_yawning={res_y['is_yawning']})")
    if t9_ok: passed_count += 1

    # --------------------------------------------------------------------------
    # Test 10: Eyes Closed + Real Yawn -> Fatigued + Yawning (NOT just yawning)
    # --------------------------------------------------------------------------
    total_count += 1
    yawn_analyzer = YawnAnalyzer()
    state_machine = DriverStateMachine()
    mouth_open = create_dummy_mouth_pts(0.90)
    t = time.time()

    for i in range(num_yawn_frames):
        res_y = yawn_analyzer.update(mouth_open, 640, 480, fps, ear=0.10, baseline_ear=0.30, pitch=0.0, relative_pitch=0.0, yaw=0.0, roll=0.0, cnn_score=0.85, eyes_closed=True, current_time=t)
        t += 0.05
        closed_sec = (i + 1) * 0.05
        signals = {
            "eyes_closed": True,
            "eye_closed_sec": closed_sec,
            "head_down": False,
            "any_tilt": False,
            "yawn_fatigue": res_y["fatigue_yawn"],
            "big_yawning": False,
            "smooth_risk": 0.60,
            "raw_risk": 0.60,
            "perclos": 0.30,
            "face_lost": False,
        }
        state = state_machine.update(signals, system_calibrated=True)

    t10_ok = res_y["is_yawning"] and (state in ("FATIGUED", "DROWSY", "MICROSLEEP"))
    print(f"[TEST 10] Eyes Closed + Real Yawn -> Fatigued + Yawning : {'PASSED' if t10_ok else 'FAILED'} (state={state}, is_yawning={res_y['is_yawning']})")
    if t10_ok: passed_count += 1

    print("\n" + "=" * 80)
    print(f"VERIFICATION RESULT: {passed_count} / {total_count} TEST CASES PASSED")
    print("=" * 80 + "\n")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_test_cases()
    if not success:
        sys.exit(1)
