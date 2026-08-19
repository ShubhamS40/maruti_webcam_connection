"""
Verification suite for temporal yawning + personalized eye calibration.
"""

import os
import sys
import time
import math
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import YAWN_MIN_DURATION_SEC
from pipeline.behavioral import YawnAnalyzer, EyeAnalyzer
from pipeline.state_machine import DriverStateMachine


def create_dummy_mouth_pts(mar_val):
    """6 landmarks producing approximate MAR = mar_val."""
    left = (0.0, 0.0)
    right = (100.0, 0.0)
    vert = 100.0 * mar_val
    half_v = vert / 2.0
    top1 = (30.0, -half_v)
    top2 = (70.0, -half_v)
    bottom1 = (30.0, half_v)
    bottom2 = (70.0, half_v)
    return [left, top1, top2, right, bottom2, bottom1]


def create_dummy_eye_pts(ear_val):
    """6 eye landmarks producing approximate EAR = ear_val."""
    # EAR = (v1+v2)/(2*h). Let h=100, each vertical = 100*ear_val
    h = 100.0
    v = h * ear_val
    p1 = (0.0, 0.0)
    p4 = (h, 0.0)
    p2 = (30.0, -v / 2)
    p3 = (70.0, -v / 2)
    p6 = (30.0, v / 2)
    p5 = (70.0, v / 2)
    return [p1, p2, p3, p4, p5, p6]


def calibrate_mar(analyzer, fps, mar=0.20, frames=50):
    pts = create_dummy_mouth_pts(mar)
    t = time.time()
    for _ in range(frames):
        analyzer.update(pts, 640, 480, fps, current_time=t)
        t += 1.0 / fps
    return t


def run_test_cases():
    print("\n" + "=" * 80)
    print("  DMS TEMPORAL YAWN + EYE CALIBRATION VERIFICATION")
    print("=" * 80 + "\n")

    fps = 20.0
    # Extra frames for EMA ramp + 1.5s sustain
    num_yawn_frames = int(math.ceil(YAWN_MIN_DURATION_SEC * fps)) + 20
    dt = 1.0 / fps

    passed = 0
    total = 0

    def check(name, ok, detail=""):
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # ------------------------------------------------------------------
    # TEST 01 / 07: closed mouth â†’ no yawn
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    closed = create_dummy_mouth_pts(0.22)
    for _ in range(num_yawn_frames):
        res = ya.update(closed, 640, 480, fps, eyes_closed=False, current_time=t)
        t += dt
    check("Closed mouth â†’ NO YAWN", (not res["is_yawning"]) and res["yawn_state"] == "NO_YAWN"
          and res["yawn_confidence"] < 0.2,
          f"state={res['yawn_state']} conf={res['yawn_confidence']:.2f}")

    # ------------------------------------------------------------------
    # TEST: eyes closed only â†’ NO YAWN
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    for _ in range(num_yawn_frames):
        res = ya.update(closed, 640, 480, fps, ear=0.08, eyes_closed=True, current_time=t)
        t += dt
    check("Eyes closed + closed mouth â†’ NO YAWN",
          (not res["yawn_active"]) and res["yawn_state"] != "CONFIRMED_YAWN",
          f"state={res['yawn_state']}")

    # ------------------------------------------------------------------
    # TEST 11: genuine yawn ~1.5s+ â†’ YES
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    open_pts = create_dummy_mouth_pts(0.85)
    for _ in range(num_yawn_frames):
        res = ya.update(open_pts, 640, 480, fps, current_time=t)
        t += dt
    check("Sustained open â‰¥1.5s â†’ YAWN",
          res["yawn_active"] or res["yawn_detected_alert"] or res["yawn_count"] >= 1,
          f"state={res['yawn_state']} count={res['yawn_count']} conf={res['yawn_confidence']:.2f}")

    # ------------------------------------------------------------------
    # TEST 10: open <1s â†’ NO YAWN
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    short_frames = max(3, int(0.8 * fps))
    for _ in range(short_frames):
        res = ya.update(open_pts, 640, 480, fps, current_time=t)
        t += dt
    for _ in range(int(fps)):
        res = ya.update(closed, 640, 480, fps, current_time=t)
        t += dt
    check("Open <1s â†’ NO YAWN", res["yawn_count"] == 0 and not res["yawn_active"],
          f"count={res['yawn_count']} state={res['yawn_state']}")

    # ------------------------------------------------------------------
    # TEST 12: one physical yawn â†’ count +1 once
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    for _ in range(num_yawn_frames):
        res = ya.update(open_pts, 640, 480, fps, current_time=t)
        t += dt
    count_after_open = res["yawn_count"]
    for _ in range(num_yawn_frames):
        res = ya.update(open_pts, 640, 480, fps, current_time=t)
        t += dt
    check("One sustained yawn â†’ single count",
          count_after_open == 1 and res["yawn_count"] == 1,
          f"count={res['yawn_count']}")

    # ------------------------------------------------------------------
    # Mouth closes â†’ state resets, confidence decays
    # ------------------------------------------------------------------
    for _ in range(int(2 * fps)):
        res = ya.update(closed, 640, 480, fps, current_time=t)
        t += dt
    check("After mouth close â†’ reset (not sticky CONFIRMED)",
          res["yawn_state"] == "NO_YAWN" and not res["yawn_active"] and res["yawn_confidence"] < 0.35,
          f"state={res['yawn_state']} conf={res['yawn_confidence']:.2f}")

    # ------------------------------------------------------------------
    # TEST 03 / 08: talking oscillations â†’ NO YAWN
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    for i in range(num_yawn_frames):
        mar_val = 0.65 if (i % 2 == 0) else 0.25
        res = ya.update(create_dummy_mouth_pts(mar_val), 640, 480, fps, current_time=t)
        t += dt
    check("Talking oscillation â†’ NO YAWN",
          res["yawn_count"] == 0 and res["yawn_state"] != "CONFIRMED_YAWN",
          f"state={res['yawn_state']} talking={res['is_talking']}")

    # ------------------------------------------------------------------
    # TEST 09: smile â†’ NO YAWN
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    smile = create_dummy_mouth_pts(0.38)
    for _ in range(num_yawn_frames):
        res = ya.update(smile, 640, 480, fps, current_time=t)
        t += dt
    check("Smile â†’ NO YAWN", res["yawn_count"] == 0, f"count={res['yawn_count']}")

    # ------------------------------------------------------------------
    # TEST 13: repeated yawns increment correctly
    # ------------------------------------------------------------------
    ya = YawnAnalyzer()
    t = calibrate_mar(ya, fps)
    for _ in range(num_yawn_frames):
        ya.update(open_pts, 640, 480, fps, current_time=t)
        t += dt
    for _ in range(int(2.5 * fps)):
        ya.update(closed, 640, 480, fps, current_time=t)
        t += dt
    for _ in range(num_yawn_frames):
        res = ya.update(open_pts, 640, 480, fps, current_time=t)
        t += dt
    check("Repeated yawns â†’ count >= 2",
          res["yawn_count"] >= 2 and res["repeated_yawning"],
          f"count={res['yawn_count']} repeated={res['repeated_yawning']}")

    # ------------------------------------------------------------------
    # FPS scaling: required frames â‰ˆ ceil(1.5 * fps)
    # ------------------------------------------------------------------
    for test_fps, approx in ((5.0, 8), (10.0, 15), (15.0, 23)):
        ya = YawnAnalyzer()
        t = calibrate_mar(ya, test_fps, frames=50)
        need = None
        open_n = int(math.ceil(YAWN_MIN_DURATION_SEC * test_fps)) + 10
        for _ in range(open_n):
            res = ya.update(open_pts, 640, 480, test_fps, current_time=t)
            need = res["required_frames"]
            t += 1.0 / test_fps
        check(f"FPS={test_fps:g} required_framesâ‰ˆ{approx}",
              need == int(math.ceil(1.5 * test_fps)) and res["yawn_count"] >= 1,
              f"required={need} count={res['yawn_count']}")

    # ------------------------------------------------------------------
    # Small-eye user: open EAR ~0.18 â†’ NOT closed after calibration
    # ------------------------------------------------------------------
    ea = EyeAnalyzer()
    eye_open = create_dummy_eye_pts(0.18)
    for _ in range(60):
        er = ea.update(eye_open, eye_open, fps)
    slight = create_dummy_eye_pts(0.15)
    for _ in range(5):
        er = ea.update(slight, slight, fps)
    check("Small eyes (EAR~0.18) attentive â†’ not EYES CLOSED",
          ea.calibrated and not er["eyes_closed"] and er["eye_state"] != "CLOSED",
          f"base={er['baseline_ear']:.3f} thr={er['ear_threshold']:.3f} state={er['eye_state']}")

    # ------------------------------------------------------------------
    # Long closure â†’ eyes closed
    # ------------------------------------------------------------------
    closed_eye = create_dummy_eye_pts(0.05)
    for _ in range(int(0.5 * fps)):
        er = ea.update(closed_eye, closed_eye, fps)
    check("Long closure â†’ EYES CLOSED",
          er["eyes_closed"] and er["eye_closed_sec"] > 0.2,
          f"closed={er['eyes_closed']} sec={er['eye_closed_sec']:.2f}")

    # ------------------------------------------------------------------
    # Microsleep path via state machine
    # ------------------------------------------------------------------
    sm = DriverStateMachine()
    state = "ATTENTIVE"
    for i in range(int(6.0 * fps)):
        closed_sec = (i + 1) / fps
        state = sm.update({
            "eyes_closed": True,
            "eye_closed_sec": closed_sec,
            "head_down": False,
            "any_tilt": False,
            "yawn_fatigue": False,
            "frequent_yawning": False,
            "repeated_yawning": False,
            "big_yawning": False,
            "smooth_risk": 0.7,
            "raw_risk": 0.7,
            "perclos": 0.6,
            "face_lost": False,
        }, system_calibrated=True)
    check("Prolonged closure â†’ MICROSLEEP",
          state == "MICROSLEEP", f"state={state}")

    print("\n" + "=" * 80)
    print(f"RESULT: {passed} / {total} PASSED")
    print("=" * 80 + "\n")
    return passed == total


if __name__ == "__main__":
    ok = run_test_cases()
    sys.exit(0 if ok else 1)

