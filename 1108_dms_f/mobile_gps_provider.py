import math
import time
import threading
import logging
import socket
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MOBILE_GPS_PORT = 5000
GPS_STALE_SEC = 12.0
GPS_MAX_SPEED_KMH = 200.0
GPS_MIN_DT_SEC = 0.3
GPS_EMA_ALPHA = 0.20

# ── HTML / JS FRONTEND ────────────────────────────────────────────────────────
_PHONE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>DMS Phone GPS</title>
    <style>
        :root {
            --bg-color: #0b1120;
            --card-bg: #111827;
            --text-main: #f3f4f6;
            --text-dim: #9ca3af;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.5);
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
            --border: #1f2937;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-main); display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 1.5rem 1rem; }
        header { text-align: center; margin-bottom: 2rem; }
        h1 { font-size: 1.75rem; font-weight: 700; background: linear-gradient(to right, #60a5fa, #3b82f6); -webkit-background-clip: text; color: transparent; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
        h1 svg { width: 1.5rem; height: 1.5rem; fill: #3b82f6; }
        .subtitle { color: var(--text-dim); font-size: 0.875rem; margin-top: 0.25rem; font-weight: 500; letter-spacing: 0.025em; }
        
        .card { background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 1rem; padding: 1.5rem; width: 100%; max-width: 400px; margin-bottom: 1rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
        
        .status-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; font-weight: 600; font-size: 0.95rem; }
        .indicator { width: 10px; height: 10px; border-radius: 50%; background-color: var(--text-dim); box-shadow: 0 0 0 0 rgba(0,0,0,0); transition: all 0.3s ease; }
        .indicator.active { background-color: var(--success); box-shadow: 0 0 8px 1px rgba(16, 185, 129, 0.5); }
        .indicator.error { background-color: var(--danger); box-shadow: 0 0 8px 1px rgba(239, 68, 68, 0.5); }
        
        .speed-display { margin: 1rem 0; }
        .speed-val { font-size: 4rem; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; color: #38bdf8; text-shadow: 0 0 20px rgba(56, 189, 248, 0.2); transition: color 0.3s; }
        .speed-label { color: var(--text-dim); font-size: 0.875rem; margin-top: 0.5rem; font-weight: 500; }
        
        .progress-bar-bg { width: 100%; height: 6px; background-color: var(--border); border-radius: 3px; margin-top: 1rem; overflow: hidden; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, var(--success), #34d399); width: 0%; transition: width 0.3s ease; }
        
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem; }
        .stat-box { display: flex; flex-direction: column; }
        .stat-label { color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; font-weight: 600; }
        .stat-val { font-weight: 600; font-size: 1rem; font-variant-numeric: tabular-nums; }
        
        .badge { display: inline-flex; align-items: center; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid var(--border); background: var(--bg-color); color: var(--text-dim); transition: all 0.3s; }
        .badge.active { border-color: var(--success); color: var(--success); box-shadow: inset 0 0 10px rgba(16, 185, 129, 0.1); }
        .badge.waiting { border-color: var(--warning); color: var(--warning); box-shadow: inset 0 0 10px rgba(245, 158, 11, 0.1); }
        
        .btn { width: 100%; padding: 1rem; border-radius: 0.75rem; border: none; background: #1e293b; color: var(--text-dim); font-weight: 600; font-size: 1rem; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.5rem; transition: all 0.2s; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .btn:active { transform: scale(0.98); }
        .btn.active { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3); }
        
        .log-box { width: 100%; max-width: 400px; background: #000; border: 1px solid var(--border); border-radius: 0.75rem; padding: 1rem; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 0.7rem; color: #a3a3a3; height: 120px; overflow-y: auto; margin-top: 0.5rem; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); }
        .log-line { margin-bottom: 0.25rem; word-break: break-all; }
        .log-time { color: #4b5563; margin-right: 0.5rem; }
    </style>
</head>
<body>
    <header>
        <h1>
            <svg viewBox="0 0 24 24"><path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.85 7h10.28l1.08 3.11H5.77L6.85 7zM19 17H5v-5h14v5z"/><circle cx="7.5" cy="14.5" r="1.5"/><circle cx="16.5" cy="14.5" r="1.5"/></svg>
            DMS Mobile GPS
        </h1>
        <div class="subtitle">Driver Monitoring System — Phone GPS Provider</div>
    </header>

    <div class="card">
        <div class="status-header">
            <div id="status-dot" class="indicator"></div>
            <span id="status-text">GPS: Inactive</span>
        </div>
        
        <div class="speed-display">
            <div id="speed" class="speed-val">--</div>
            <div class="speed-label">km/h • smoothed speed</div>
        </div>
        
        <div class="progress-bar-bg">
            <div id="speed-bar" class="progress-bar"></div>
        </div>
    </div>

    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <div class="stat-box">
                <span class="stat-label">DMS Backend Status</span>
                <div id="dms-badge" class="badge">DISCONNECTED</div>
            </div>
            <div class="stat-box" style="align-items: flex-end;">
                <span class="stat-label">Fixes Sent</span>
                <span id="fixes-count" class="stat-val" style="color: #38bdf8;">0</span>
            </div>
        </div>
        
        <hr style="border: none; border-top: 1px solid var(--border); margin: 1rem 0;">
        
        <div class="grid-2">
            <div class="stat-box">
                <span class="stat-label">Latitude</span>
                <span id="lat" class="stat-val">--</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">Longitude</span>
                <span id="lon" class="stat-val">--</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">Accuracy</span>
                <span id="acc" class="stat-val">--</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">Last Update</span>
                <span id="last-time" class="stat-val">--</span>
            </div>
        </div>
    </div>

    <button id="toggle-btn" class="btn">
        <span>📡</span> Start GPS Tracking
    </button>

    <div class="log-box" id="log-box"></div>

    <script>
        const DOM = {
            toggleBtn: document.getElementById('toggle-btn'),
            statusDot: document.getElementById('status-dot'),
            statusText: document.getElementById('status-text'),
            speed: document.getElementById('speed'),
            speedBar: document.getElementById('speed-bar'),
            dmsBadge: document.getElementById('dms-badge'),
            fixesCount: document.getElementById('fixes-count'),
            lat: document.getElementById('lat'),
            lon: document.getElementById('lon'),
            acc: document.getElementById('acc'),
            lastTime: document.getElementById('last-time'),
            logBox: document.getElementById('log-box')
        };

        let watchId = null;
        let fixesSent = 0;
        let isTracking = false;
        let statusPollInterval = null;

        function log(msg) {
            const time = new Date().toLocaleTimeString('en-US', { hour12: false });
            const line = document.createElement('div');
            line.className = 'log-line';
            line.innerHTML = `<span class="log-time">${time}</span>${msg}`;
            DOM.logBox.prepend(line);
        }

        function updateBadge(state) {
            DOM.dmsBadge.className = 'badge';
            if (state === 'ACTIVE') {
                DOM.dmsBadge.textContent = 'DMS ACTIVE';
                DOM.dmsBadge.classList.add('active');
            } else if (state === 'WAITING') {
                DOM.dmsBadge.textContent = 'WAITING FOR SPEED';
                DOM.dmsBadge.classList.add('waiting');
            } else {
                DOM.dmsBadge.textContent = 'DISCONNECTED';
            }
        }

        async function pollStatus() {
            if (!isTracking) return;
            try {
                // Fetch using relative URL to handle any tunnel/proxy base paths
                const res = await fetch('status');
                if (res.ok) {
                    const data = await res.json();
                    if (data.speed !== undefined) {
                        DOM.speed.textContent = data.speed.toFixed(1);
                        DOM.speedBar.style.width = Math.min(100, (data.speed / 120) * 100) + '%';
                    }
                    if (data.dms_active !== undefined) {
                        updateBadge(data.dms_active ? 'ACTIVE' : 'WAITING');
                    }
                }
            } catch (err) {
                // Silent catch, don't spam logs for polling
                updateBadge('DISCONNECTED');
            }
        }

        async function sendPosition(pos) {
            const data = {
                lat: pos.coords.latitude,
                lon: pos.coords.longitude,
                accuracy: pos.coords.accuracy,
                ts: pos.timestamp
            };
            
            DOM.lat.textContent = data.lat.toFixed(6);
            DOM.lon.textContent = data.lon.toFixed(6);
            DOM.acc.textContent = Math.round(data.accuracy) + ' m';
            DOM.lastTime.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });

            try {
                // Post using relative URL
                const res = await fetch('gps', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                if (res.ok) {
                    fixesSent++;
                    DOM.fixesCount.textContent = fixesSent;
                    DOM.statusDot.className = 'indicator active';
                    DOM.statusText.textContent = `GPS: Active — acc ${Math.round(data.accuracy)}m`;
                } else {
                    throw new Error('Server returned ' + res.status);
                }
            } catch (err) {
                log('POST Error: ' + err.message);
                DOM.statusDot.className = 'indicator error';
                DOM.statusText.textContent = 'GPS: Connection Error';
                updateBadge('DISCONNECTED');
            }
        }

        function handleError(err) {
            log(`GPS Error: ${err.message} (${err.code})`);
            DOM.statusDot.className = 'indicator error';
            let msg = 'GPS: Error';
            if (err.code === 1) msg = 'GPS: Permission Denied';
            if (err.code === 2) msg = 'GPS: Position Unavailable';
            if (err.code === 3) msg = 'GPS: Timeout';
            DOM.statusText.textContent = msg;
        }

        function toggleTracking() {
            if (isTracking) {
                if (watchId !== null) navigator.geolocation.clearWatch(watchId);
                clearInterval(statusPollInterval);
                isTracking = false;
                watchId = null;
                DOM.toggleBtn.className = 'btn';
                DOM.toggleBtn.innerHTML = '<span>📡</span> Start GPS Tracking';
                DOM.statusDot.className = 'indicator';
                DOM.statusText.textContent = 'GPS: Inactive';
                DOM.speed.textContent = '--';
                DOM.speedBar.style.width = '0%';
                updateBadge('DISCONNECTED');
                log('Tracking stopped.');
            } else {
                if (!('geolocation' in navigator)) {
                    log('Error: Geolocation not supported by browser.');
                    alert('Geolocation is not supported by your browser.');
                    return;
                }
                log('Requesting location permission...');
                isTracking = true;
                DOM.toggleBtn.className = 'btn active';
                DOM.toggleBtn.innerHTML = '<span>📡</span> Stop Tracking';
                DOM.statusDot.className = 'indicator';
                DOM.statusText.textContent = 'GPS: Acquiring fix...';
                
                watchId = navigator.geolocation.watchPosition(sendPosition, handleError, {
                    enableHighAccuracy: true,
                    maximumAge: 0,
                    timeout: 10000
                });
                
                statusPollInterval = setInterval(pollStatus, 1000);
                log('watchPosition started.');
            }
        }

        DOM.toggleBtn.addEventListener('click', toggleTracking);
        log('Page ready - requesting GPS...');
        
        // Auto-start
        setTimeout(toggleTracking, 500);
    </script>
</body>
</html>"""

# ── HAVERSINE DISTANCE ────────────────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in meters between two points on earth."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ── API SCHEMAS ───────────────────────────────────────────────────────────────
class GPSPayload(BaseModel):
    lat: float
    lon: float
    ts: float       # milliseconds from JS Date.now()
    accuracy: float

# ── PROVIDER CLASS ────────────────────────────────────────────────────────────
class MobileGPSProvider:
    """
    Acts as a SpeedManager hardware provider, but internally runs an embedded
    FastAPI HTTP server to receive GPS updates from a phone browser.
    Uses Haversine formula and EMA smoothing (same as hardware GPS).
    """
    def __init__(self, port: int = MOBILE_GPS_PORT):
        self.port = port
        self.connected = False
        self.current_speed = 0.0
        
        # EMA / Haversine state
        self._prev_lat: Optional[float] = None
        self._prev_lon: Optional[float] = None
        self._prev_ts: Optional[float] = None
        self._last_fix_ts = 0.0
        self._last_accuracy = 0.0
        
        # Threading and server state
        self._fix_lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        
        # Propagated back to phone UI
        self._dms_active = False

        # Get local IP for display
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "127.0.0.1"
        self._server_url = f"http://{ip}:{self.port}"
        
        self._app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Mobile GPS Provider", docs_url=None, redoc_url=None)
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/", response_class=HTMLResponse)
        async def serve_ui():
            return _PHONE_PAGE

        @app.post("/gps")
        async def receive_gps(payload: GPSPayload):
            ts_sec = payload.ts / 1000.0
            self._on_gps_update(payload.lat, payload.lon, ts_sec, payload.accuracy)
            return {"ok": True}

        @app.get("/status")
        async def get_status():
            return {
                "connected": self.connected,
                "speed": self.current_speed,
                "dms_active": self._dms_active
            }

        return app

    def _apply_ema(self, new_speed: float) -> float:
        """Apply Exponential Moving Average smoothing (same as legacy gpsd)."""
        if self.current_speed == 0.0:
            return new_speed
        return (GPS_EMA_ALPHA * new_speed) + ((1.0 - GPS_EMA_ALPHA) * self.current_speed)

    def _compute_speed(self, lat: float, lon: float, ts_sec: float) -> Optional[float]:
        with self._fix_lock:
            if self._prev_lat is None or self._prev_lon is None or self._prev_ts is None:
                self._prev_lat = lat
                self._prev_lon = lon
                self._prev_ts = ts_sec
                return None
            
            dt = ts_sec - self._prev_ts
            if dt < GPS_MIN_DT_SEC:
                return None
            if dt > 60.0:
                self._prev_lat = lat
                self._prev_lon = lon
                self._prev_ts = ts_sec
                return None
                
            dist_m = _haversine(self._prev_lat, self._prev_lon, lat, lon)
            speed_ms = dist_m / dt
            speed_kmh = speed_ms * 3.6
            
            if speed_kmh > GPS_MAX_SPEED_KMH or dist_m > 500.0:
                print(f"[MobileGPS] Outlier: {dist_m:.0f} m jump in {dt:.1f}s — skipped")
                return None
                
            self._prev_lat = lat
            self._prev_lon = lon
            self._prev_ts = ts_sec
            return speed_kmh

    def _on_gps_update(self, lat: float, lon: float, ts_sec: float, accuracy: float):
        raw_speed = self._compute_speed(lat, lon, ts_sec)
        
        with self._fix_lock:
            self._last_fix_ts = time.time()
            self._last_accuracy = accuracy
            self.connected = True
            
            if raw_speed is not None:
                self.current_speed = self._apply_ema(raw_speed)

    def _run_server(self):
        config = uvicorn.Config(
            app=self._app, 
            host="0.0.0.0", 
            port=self.port,
            log_level="error",
            access_log=False
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def start(self):
        if self._server_thread is None:
            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._server_thread.start()
            # Suppress uvicorn/fastapi logging noise
            logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
            logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
            
    def get_speed(self) -> float:
        with self._fix_lock:
            # Stale detection
            if time.time() - self._last_fix_ts > GPS_STALE_SEC:
                self.connected = False
                self.current_speed = 0.0
        return self.current_speed

    def get_status_info(self) -> Dict[str, Any]:
        with self._fix_lock:
            is_stale = (time.time() - self._last_fix_ts) > GPS_STALE_SEC
            if not self.connected and not is_stale:
                status = "Disconnected"
            elif is_stale:
                status = "Stale"
            else:
                status = "Connected"
                
            return {
                "provider": "Mobile GPS",
                "status": status,
                "accuracy": f"{self._last_accuracy:.0f}m" if self._last_accuracy else ""
            }
