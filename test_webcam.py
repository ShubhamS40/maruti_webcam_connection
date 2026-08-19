#!/usr/bin/env python3
"""
QUICK WEBCAM TEST (Cross-platform)
-----------------------------------
Checks if Quantron QPC-1020 HD (or any USB webcam) is detected by OpenCV.
Prints all found camera indices + OS backend info.

Usage: python3 test_webcam.py
Press 'q' inside preview window to exit.
"""
import os
import sys
import cv2

OS = sys.platform  # darwin = Mac, win32 = Windows, linux = Linux
print(f"OS Detected: {OS}")

if OS == "darwin":
    BACKEND = cv2.CAP_AVFOUNDATION
    BACKEND_NAME = "AVFoundation (Mac native)"
elif OS == "win32":
    BACKEND = cv2.CAP_DSHOW
    BACKEND_NAME = "DirectShow (Windows native)"
else:
    BACKEND = 0
    BACKEND_NAME = "Default (V4L2 etc)"

print(f"Using OpenCV backend: {BACKEND_NAME}")
print("-" * 60)

found = []
for idx in range(0, 6):
    if BACKEND != 0:
        cap = cv2.VideoCapture(idx, BACKEND)
    else:
        cap = cv2.VideoCapture(idx)
    if cap is None:
        continue
    opened = cap.isOpened()
    ok = False
    w = h = 0
    if opened:
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            found.append((idx, w, h))
            print(f"  ✅ Camera index {idx}: OPENED  ({w}x{h})")
        else:
            print(f"  ⚠️  Camera index {idx}: device opened but no frame read")
    else:
        print(f"  ❌ Camera index {idx}: NOT FOUND")
    cap.release()

print("-" * 60)
if not found:
    print("ERROR: No cameras detected! Try these fixes:")
    if OS == "darwin":
        print("  → Mac: System Settings → Privacy & Security → Camera → ENABLE your Terminal/IDE")
        print("  → Reconnect USB cable, try different USB port")
        print("  → Check: system_profiler SPCameraDataType")
    elif OS == "win32":
        print("  → Windows: Settings → Privacy & Security → Camera → Allow apps to access camera")
        print("  → Device Manager → Cameras → Quantron QPC-1020 HD → enable if disabled")
    sys.exit(1)

best_idx = found[0][0]
print(f"\n🎯 Best camera found: index {best_idx}  ({found[0][1]}x{found[0][2]})")
print(f"   Opening preview window... press 'q' to close.")

if BACKEND != 0:
    cap = cv2.VideoCapture(best_idx, BACKEND)
else:
    cap = cv2.VideoCapture(best_idx)

if not cap.isOpened():
    cap = cv2.VideoCapture(best_idx)

if not cap.isOpened():
    print("ERROR: Could not open camera even though it was detected earlier.")
    sys.exit(1)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("ERROR: Lost camera feed.")
        break
    # Draw overlay for Quantron QPC-1020 HD confirmation
    h, w = frame.shape[:2]
    cv2.putText(frame, "QUANTRON QPC-1020 HD WEBCAM TEST", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Resolution: {w}x{h}   |   Press 'q' to exit", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Quantron QPC-1020 HD - Preview (press q to close)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Webcam test complete. Camera is working perfectly!")
