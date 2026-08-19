"""DMS dashboard + debug overlay."""

import cv2

from config import Colors, DEBUG_OVERLAY


def draw_bar(frame, x, y, w, h, value, color):
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * value)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), Colors.WHITE, 2)


def draw_face_mesh(frame, landmarks, step=4, color=Colors.CYAN):
    for idx in range(0, len(landmarks), step):
        cv2.circle(frame, landmarks[idx], 1, color, -1)


def draw_status_panel(frame, state, state_color, metrics, risk, attention):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 120), (500, 720), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.rectangle(frame, (10, 120), (500, 720), Colors.MAGENTA, 3)

    cv2.putText(
        frame, state, (30, 190),
        cv2.FONT_HERSHEY_DUPLEX, 1.8, state_color, 4,
    )

    y = 230
    num_lines = len(metrics)
    font_scale = 0.78 if num_lines <= 10 else (0.52 if num_lines > 14 else 0.62)
    line_spacing = 34 if num_lines <= 10 else (21 if num_lines > 14 else 26)
    thickness = 2 if font_scale > 0.6 else 1

    for line in metrics:
        cv2.putText(
            frame, line, (35, y),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, Colors.WHITE, thickness,
        )
        y += line_spacing

    cv2.putText(frame, "DROWSINESS (smoothed risk)", (30, 600),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, Colors.WHITE, 2)
    draw_bar(frame, 30, 620, 400, 28, risk, state_color)

    cv2.putText(frame, "ATTENTION", (30, 680),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, Colors.WHITE, 2)
    draw_bar(frame, 30, 700, 400, 28, attention / 100.0, Colors.GREEN)


def draw_debug_overlay(frame, dbg):
    if not DEBUG_OVERLAY or not dbg:
        return
    x0, y0 = 520, 210

    lines = [
        "--- DEBUG ---",
        f"raw EAR      {dbg.get('raw_ear', 0):.3f}",
        f"smooth EAR   {dbg.get('ear', 0):.3f}",
        f"baseline EAR {dbg.get('baseline_ear', 0):.3f}",
        f"raw MAR      {dbg.get('raw_mar', 0):.3f}",
        f"smooth MAR   {dbg.get('mar', 0):.3f}",
        f"yawn conf    {dbg.get('yawn_confidence', 0):.2f}",
        f"yawn frames  {dbg.get('yawn_frames', 0)}",
        f"yawn duration{dbg.get('yawn_duration', 0.0):.1f}s",
        f"yawn state   {dbg.get('yawn_state', 'NO_YAWN')}",
        f"raw risk     {dbg.get('raw_risk', 0):.3f}",
        f"smooth risk  {dbg.get('smooth_risk', 0):.3f}",
        f"PERCLOS n={dbg.get('perclos_n', 0)}  {dbg.get('perclos', 0):.3f}",
        f"CNN raw/sm   {dbg.get('cnn_raw', 0):.3f} / {dbg.get('cnn_smooth', 0):.3f}",
        f"pitch/base   {dbg.get('pitch', 0):.1f} / {dbg.get('pitch_baseline', 0):.1f}",
        f"rel pitch    {dbg.get('rel_pitch', 0):.1f}",
        f"eyes closed  {dbg.get('eyes_closed', False)}",
        f"closed sec   {dbg.get('eye_closed_sec', 0):.2f}",
        f"eval frames  {dbg.get('eval_frames', 0)}/{dbg.get('eval_need', 0)}",
        f"state cnt    F{dbg.get('fatigue_frames', 0)} D{dbg.get('drowsy_frames', 0)} M{dbg.get('micro_frames', 0)} E{dbg.get('exit_frames', 0)}",
        f"alert on     {dbg.get('alert_active', False)}",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x0, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            Colors.CYAN if i == 0 else Colors.WHITE,
            1 if i > 0 else 2,
        )


def draw_alerts(frame, alerts):
    y = 520
    for text, color in alerts:
        cv2.putText(
            frame, text, (650, y),
            cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3,
        )
        y += 70


def draw_standby_banner(frame, message="WAITING FOR SPEED LIMIT"):
    cv2.putText(
        frame, message, (520, 120),
        cv2.FONT_HERSHEY_DUPLEX, 1, Colors.YELLOW, 3,
    )
