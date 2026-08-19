"""
Short real-camera probe for yawn/eye calibration (non-interactive, ~12s).
Forces analyzers on regardless of speed gate for diagnostic sampling.
"""
import os
import sys
import time
import math

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES, VIDEO_SOURCE
from pipeline.behavioral import EyeAnalyzer, YawnAnalyzer, HeadPoseAnalyzer
from pipeline.detection import FaceDetector


def main(duration_sec=12.0):
    print("=" * 70)
    print("REAL CAMERA PROBE — temporal yawn / personalized EAR")
    print("=" * 70)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("FAIL: camera not opened")
        return 1

    face = FaceDetector()
    eye_a = EyeAnalyzer()
    yawn_a = YawnAnalyzer()
    head_a = HeadPoseAnalyzer()

    fps_ema = 15.0
    prev = time.time()
    t0 = prev
    samples = []
    frames = 0

    while time.time() - t0 < duration_sec:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        fps_ema = 0.9 * fps_ema + 0.1 / max(now - prev, 1e-6)
        prev = now

        landmarks = face.process(frame)
        if landmarks is None or len(landmarks) < 468:
            continue

        frames += 1
        eye = eye_a.update(
            [landmarks[i] for i in LEFT_EYE_INDICES],
            [landmarks[i] for i in RIGHT_EYE_INDICES],
            fps_ema,
        )
        head = head_a.update(landmarks, w, h, eyes_closed=eye["eyes_closed"])
        yawn = yawn_a.update(
            [landmarks[i] for i in MOUTH_INDICES],
            w, h, fps_ema,
            pitch=head["pitch"], relative_pitch=head["relative_pitch"],
            yaw=head["yaw"], roll=head["roll"],
            eyes_closed=eye["eyes_closed"],
            current_time=now,
        )
        samples.append({
            "raw_mar": yawn["raw_mar"],
            "mar": yawn["mar"],
            "base_mar": yawn["baseline_mar"],
            "start": yawn["yawn_start_threshold"],
            "end": yawn["yawn_end_threshold"],
            "peak": yawn["yawn_peak_mar"],
            "cand": yawn["open_frames"],
            "need": yawn["required_frames"],
            "dur": yawn["open_duration_sec"],
            "conf": yawn["yawn_confidence"],
            "ystate": yawn["yawn_state"],
            "raw_ear": eye["raw_ear"],
            "ear": eye["ear"],
            "base_ear": eye["baseline_ear"],
            "close_thr": eye["ear_close_threshold"],
            "open_thr": eye["ear_open_threshold"],
            "estate": eye["eye_state"],
            "closed": eye["eyes_closed"],
        })

    cap.release()
    if not samples:
        print("FAIL: no face samples")
        return 1

    mars = [s["mar"] for s in samples]
    ears = [s["ear"] for s in samples]
    last = samples[-1]
    sticky = sum(1 for s in samples if s["ystate"] == "CONFIRMED_YAWN")
    false_yawn_closed = sum(
        1 for s in samples
        if s["ystate"] == "CONFIRMED_YAWN" and s["mar"] < s["end"]
    )

    print(f"Frames with face : {frames}")
    print(f"Eval FPS (EMA)   : {fps_ema:.1f}")
    print(f"MAR median/p95   : {np.median(mars):.3f} / {np.percentile(mars, 95):.3f}")
    print(f"EAR median/p05   : {np.median(ears):.3f} / {np.percentile(ears, 5):.3f}")
    print(f"MAR baseline     : {last['base_mar']:.3f}")
    print(f"YAWN start/end   : {last['start']:.3f} / {last['end']:.3f}")
    print(f"EAR baseline     : {last['base_ear']}")
    print(f"EAR close/open   : {last['close_thr']:.3f} / {last['open_thr']:.3f}")
    print(f"Last yawn state  : {last['ystate']} conf={last['conf']:.2f}")
    print(f"Last eye state   : {last['estate']} closed={last['closed']}")
    print(f"CONFIRMED frames : {sticky} (false while mar<end: {false_yawn_closed})")
    print(f"Yawn count       : {yawn_a.yawn_count}")
    print(f"Required frames  : {last['need']} (= ceil(1.5*{fps_ema:.1f}))")

    # Acceptance for attentive sitting with closed/normal mouth
    ok = True
    if last["ystate"] == "CONFIRMED_YAWN" and last["mar"] < last["end"]:
        print("FAIL: sticky CONFIRMED_YAWN with mouth below end thr")
        ok = False
    if false_yawn_closed > 0:
        print("FAIL: CONFIRMED while MAR below end threshold")
        ok = False
    if np.median(mars) >= last["start"]:
        print("WARN: median MAR >= start thr (face may be mid-yawn during probe)")
    if eye_a.calibrated and np.median(ears) >= last["close_thr"] and last["closed"] and last["estate"] == "CLOSED":
        # only fail if clearly open relative to baseline
        if last["base_ear"] and np.median(ears) > 0.85 * last["base_ear"]:
            print("FAIL: eyes marked closed while median EAR near baseline")
            ok = False

    print("PROBE RESULT:", "PASS" if ok else "FAIL")
    print("=" * 70)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
