# ==============================================================================
#  test_webcam.py - ROBUST USB Webcam Detector (Mac ARM + Windows + Linux)
#  SPECIAL: Mac AVFoundation TCC permission handling + Multi-backend fallback
# ==============================================================================
#  Run this first to make sure your Quantron QPC-1020 HD USB webcam is detected
#  BEFORE running the full DMS pipeline.
#
#  Mac Troubleshooting (run these in Terminal BEFORE this script if stuck):
#      1. Reset camera permission DB (prompts new "Allow" dialog):
#           tccutil reset Camera
#      2. Check system sees the USB camera:
#           system_profiler SPCameraDataType
#      3. Check TCC auth status for Terminal/IDE:
#           tccutil -d Camera 2>/dev/null ; sqlite3 ~/Library/Application\ Support/com.apple.TCC/TCC.db "SELECT client,auth_value FROM access WHERE service='kTCCServiceCamera';" 2>/dev/null
# ==============================================================================
import os
import sys
import subprocess
import time
import platform

import cv2


# ---------------------------------------------------------------------------
# MAC TCC / UVC UTILITIES
# ---------------------------------------------------------------------------
def run_mac_camera_diagnostics() -> None:
    """Print everything the OS can tell us about cameras + TCC permissions."""
    if platform.system() != "Darwin":
        return
    print("\n" + "=" * 72)
    print("  🍎 MAC CAMERA DIAGNOSTIC")
    print("=" * 72)

    # 1) system_profiler - tells us PHYSICALLY what cameras are attached
    print("\n--- system_profiler SPCameraDataType (physically connected cameras) ---")
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        print(out)
    except Exception as e:
        print(f"  (could not run system_profiler: {e})")

    # 2) USB tree - Quantron QPC-1020 HD USB 2.0 UVC device should appear here
    print("--- ioreg / USB camera devices (VID:PID hints) ---")
    try:
        out = subprocess.check_output(
            ["ioreg", "-p", "IOUSB", "-w0", "-l"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        # Filter lines that look like cameras / video / UVC
        cameraish = [
            l for l in out.splitlines()
            if any(k in l.lower() for k in
                   ("uvc", "camera", "webcam", "video", "quantron",
                    "1080p", "camera control", "videodisplay"))
        ]
        if cameraish:
            for l in cameraish[:40]:  # cap to avoid spam
                print(l)
        else:
            print("  (no UVC/Video lines in ioreg - device may not be USB enumerated)")
    except Exception as e:
        print(f"  (could not run ioreg: {e})")

    # 3) TCC advice
    print("\n--- MAC TCC PERMISSION FIX (RUN THESE IF CAMERA STATUS 0) ---")
    print("  1. Reset Camera permission DB so Mac prompts again:")
    print("       tccutil reset Camera")
    print("  2. After resetting, QUIT & RE-OPEN Terminal/IDE completely,")
    print("     then run test_webcam.py again and click ALLOW when Mac asks.")
    print("  3. Verify manually:")
    print("       System Settings → Privacy & Security → Camera")
    print("       → Turn ON toggle for: Terminal   (or iTerm, VS Code, PyCharm etc.)")
    print("  4. If using a USB-C hub, try direct USB port / different cable /")
    print("     remove other USB devices (some hubs share bandwidth).")
    print()


# ---------------------------------------------------------------------------
# ROBUST CAMERA OPEN: Try every backend + index combination
# ---------------------------------------------------------------------------
BACKENDS_TO_TRY = []
_os = platform.system()
if _os == "Darwin":
    # Mac: AVFoundation is native. CAP_ANY sometimes works when explicit
    # AVF fails due to TCC prompt race condition.
    BACKENDS_TO_TRY = [
        ("CAP_ANY (auto-detect)", cv2.CAP_ANY),
        ("CAP_AVFOUNDATION (Mac native)", cv2.CAP_AVFOUNDATION),
    ]
elif _os == "Windows":
    BACKENDS_TO_TRY = [
        ("CAP_DSHOW (DirectShow, USB webcams)", cv2.CAP_DSHOW),
        ("CAP_ANY (auto-detect)", cv2.CAP_ANY),
        ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else 1400),
    ]
else:  # Linux / other
    BACKENDS_TO_TRY = [
        ("CAP_ANY (auto-detect)", cv2.CAP_ANY),
        ("CAP_V4L2 (Video4Linux)", cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 200),
    ]


def try_open_camera(index: int, backend_name: str, backend_id: int,
                    retries: int = 3, sleep_between: float = 0.6):
    """Try to open camera with retries. Mac TCC dialog can take time."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            cap = cv2.VideoCapture(index, backend_id)
        except Exception as e:
            last_err = f"constructor exception: {e}"
            time.sleep(sleep_between)
            continue

        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            last_err = f"cap.isOpened()=False"
            time.sleep(sleep_between)
            continue

        # Probe a frame to confirm the stream is alive (sometimes AVF opens
        # but returns black frames until TCC is granted).
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            last_err = f"opened but cap.read() failed (frame={getattr(frame,'shape',None)})"
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(sleep_between)
            continue

        return cap, attempt, None
    return None, retries, last_err


def find_working_camera(max_index: int = 10):
    """Probe all indices against every backend. Returns (cap, index, backend_name)."""
    print(f"\nOS Detected: {_os}")
    print(f"Probing backends in order: {[b[0] for b in BACKENDS_TO_TRY]}")
    print("-" * 72)

    # Try DEFAULT first without backend hint (OpenCV runtime pick)
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"  ✅ Camera index {i}: WORKING (OpenCV default backend, "
                      f"shape={frame.shape})")
                return cap, i, "OpenCV_default"
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # If default failed, try every backend explicitly
    for backend_name, backend_id in BACKENDS_TO_TRY:
        for i in range(max_index + 1):
            print(f"  Probing idx={i:2d}  backend={backend_name} ...", flush=True, end=" ")
            cap, attempts, err = try_open_camera(i, backend_name, backend_id)
            if cap is not None:
                ok, frame = cap.read()
                shape = getattr(frame, "shape", None) if ok else None
                print(f"✅ WORKING (attempts={attempts}, shape={shape})")
                return cap, i, backend_name
            else:
                print(f"❌ fail ({err})")

    return None, -1, None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  MARUTI DMS - USB WEBCAM TESTER (Quantron QPC-1020 HD friendly)")
    print("=" * 72)
    print(f"  OpenCV version : {cv2.__version__}")
    print(f"  Python         : {sys.version.split()[0]}")
    print(f"  Platform       : {platform.platform()}")

    if _os == "Darwin":
        run_mac_camera_diagnostics()

    cap, idx, backend = find_working_camera(max_index=8)

    if cap is None:
        print("\n" + "=" * 72)
        print("  ❌ ERROR: No working camera could be opened.")
        print("=" * 72)
        if _os == "Darwin":
            print("\n  🍎 NEXT STEPS (copy-paste into Terminal):")
            print("    ┌─────────────────────────────────────────────────────────────┐")
            print("    │  1. Reset TCC Camera DB (new permission prompt):           │")
            print("    │     tccutil reset Camera                                   │")
            print("    │                                                             │")
            print("    │  2. QUIT Terminal/IDE COMPLETELY (Cmd+Q), reopen it.       │")
            print("    │                                                             │")
            print("    │  3. Run again:                                             │")
            print("    │     python3 test_webcam.py                                 │")
            print("    │     → Click ALLOW on Mac's 'Camera access' dialog.         │")
            print("    │                                                             │")
            print("    │  4. If still fails, confirm USB enumeration:               │")
            print("    │     system_profiler SPCameraDataType                       │")
            print("    │     (Should show 'Quantron QPC-1020 HD' or 'USB Camera')   │")
            print("    └─────────────────────────────────────────────────────────────┘")
        print()
        sys.exit(1)

    # ---- Live preview loop -------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  ✅ SUCCESS → Camera index={idx}  Backend={backend}")
    print("  Press 'q' or ESC to close preview.")
    print("=" * 72)

    # Request HD if supported (Quantron QPC-1020 HD does 1920x1080 @ 30fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_smooth = 0.0
    t_prev = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("  ⚠️  cap.read() failed mid-stream; retrying...")
                time.sleep(0.1)
                continue
            frames += 1
            now = time.time()
            dt = now - t_prev
            if dt > 0:
                instant = 1.0 / dt
                fps_smooth = instant if fps_smooth == 0 else (fps_smooth * 0.9 + instant * 0.1)
            t_prev = now

            h, w = frame.shape[:2]
            # HUD overlay
            label = f"CAM idx={idx} | {w}x{h} | {fps_smooth:.1f} fps | Backend: {backend}"
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Maruti Webcam Test - press q/ESC to quit",
                        (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Maruti Webcam Test (press q/ESC)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  Cleanup done. Total frames shown: {frames}")

    print("  ✅ Preview test PASSED. Now you can run: python3 run_dms_webcam.py")


if __name__ == "__main__":
    main()
