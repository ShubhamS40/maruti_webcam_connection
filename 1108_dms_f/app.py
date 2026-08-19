from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.requests import Request

import cv2

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="./static"),
    name="static"
)

# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="./templates"
)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

# camera width
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# camera height
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================================================
# FRAME GENERATOR
# =========================================================

def generate_frames():

    while True:

        success, frame = cap.read()

        if not success:
            break

        # =============================================
        # SIMPLE UI OVERLAY
        # =============================================

        cv2.rectangle(
            frame,
            (20, 20),
            (420, 250),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "AI DRIVER MONITORING SYSTEM",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "STATUS : ACTIVE",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "EAR MONITORING",
            (40, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "HEAD POSE TRACKING",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "PERCLOS ANALYSIS",
            (40, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "REAL-TIME AI INFERENCE",
            (40, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # =============================================
        # ENCODE FRAME
        # =============================================

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        # =============================================
        # STREAM FRAME
        # =============================================

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)

async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )

# =========================================================
# VIDEO FEED
# =========================================================

@app.get("/video_feed")

def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type=
        "multipart/x-mixed-replace; boundary=frame"
    )

# =========================================================
# CLEANUP
# =========================================================

@app.on_event("shutdown")

def shutdown_event():

    cap.release()

    cv2.destroyAllWindows()