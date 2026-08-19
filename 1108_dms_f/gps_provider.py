# ==============================================================================
#                       GPS SPEED PROVIDER — Production Redesign
# ==============================================================================
# File   : gps_provider.py
# Purpose: Estimate vehicle speed from continuous GPS position updates.
#          Instead of reading speed directly from NMEA fields (which are often
#          empty or unreliable), this provider:
#            1. Continuously reads latitude / longitude / timestamp.
#            2. Computes travelled distance between consecutive fixes using the
#               Haversine formula.
#            3. Derives speed = distance / time and converts to km/h.
#            4. Applies an Exponential Moving Average (EMA) over the last
#               GPS_EMA_WINDOW samples to suppress GPS coordinate noise.
#            5. Filters out impossible/unrealistic values before smoothing.
#            6. Exposes GPS fix quality and satellite count for the HUD.
#
# Speed Sources (priority order)
#   1. gpsd daemon  — preferred; reads lat/lon from TPV reports
#   2. Serial NMEA  — direct USB/COM GPS; parses $GPRMC / $GPGGA / $GPGLL
#
# Thread Safety
#   All shared state is protected by self.lock (inherited from SpeedProvider).
#   The background thread writes; all public getters read under the same lock.
#
# Zero-Speed Mode
#   When activation_kmh == 0, SpeedGate / SpeedManager bypass GPS entirely.
#   This class makes no special-case for it — the manager layer handles it.
# ==============================================================================

import math
import time
import threading
import collections

import serial
import serial.tools.list_ports

try:
    import gpsd as _gpsd_module
    _GPSD_AVAILABLE = True
except ImportError:
    _GPSD_AVAILABLE = False

from speed_provider import SpeedProvider


# ── Tunable constants ──────────────────────────────────────────────────────────
# EMA window: number of raw speed samples kept in rolling deque
GPS_EMA_WINDOW   = 15
# EMA alpha: higher = more responsive, lower = smoother (0.2 matches config default)
GPS_EMA_ALPHA    = 0.20
# Maximum believable instantaneous speed in km/h (filter teleport / bad fix)
GPS_MAX_SPEED    = 200.0
# Maximum believable distance jump between two consecutive updates (metres)
GPS_MAX_JUMP_M   = 500.0
# Minimum elapsed seconds between two fixes before we compute speed
# (avoids division-by-very-small-dt causing noise)
GPS_MIN_DT_SEC   = 0.05
# Scan cooldown — seconds between COM port scan retries on failure
GPS_SCAN_COOLDOWN = 30.0
# Earth mean radius in metres (WGS-84 approximation)
_EARTH_R_M = 6_371_000.0


# ==============================================================================
# HELPERS
# ==============================================================================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance (metres) between two WGS-84 coordinates.

    Formula:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2·atan2(√a, √(1−a))
        d = R · c
    """
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    a = (math.sin(d_lat / 2.0) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return _EARTH_R_M * c


def _dms_to_decimal(dms_str: str, hemisphere: str) -> float | None:
    """
    Convert an NMEA coordinate string (DDDMM.MMMM) to decimal degrees.
    Returns None if parsing fails.
    """
    try:
        dms_str = dms_str.strip()
        if not dms_str:
            return None
        # NMEA format: DDDMM.MMMM  (degrees + minutes)
        dot = dms_str.index('.')
        # degrees occupy everything up to 2 digits before the dot
        deg_digits = dot - 2
        degrees = float(dms_str[:deg_digits])
        minutes = float(dms_str[deg_digits:])
        decimal = degrees + minutes / 60.0
        if hemisphere in ('S', 'W'):
            decimal = -decimal
        return decimal
    except (ValueError, IndexError):
        return None


# ==============================================================================
# GPS PROVIDER
# ==============================================================================

class GPSProvider(SpeedProvider):
    """
    Estimates vehicle speed from continuous GPS position updates.

    Supports:
      1. Local gpsd daemon (gpsd-py3) — preferred path
      2. Direct serial / USB GPS receivers parsing NMEA sentences

    Speed Estimation Algorithm
    ──────────────────────────
    On every new (lat, lon, timestamp) fix:
      1. Compute Haversine distance from the previous fix.
      2. Divide by elapsed time → raw speed (m/s → km/h).
      3. Validate: discard negative, >200 km/h, >500 m jump, or dt < 50 ms.
      4. Feed into EMA smoother (window=15, α=0.20).
      5. Store result in self.current_speed (thread-safe via self.lock).

    Status / Quality
    ────────────────
    self.accuracy_str   → "3D Fix (Sats: 9)", "No Fix", "GPS NOT AVAILABLE" …
    self.connected      → True when a valid fix exists and speed is being computed
    self.method_str     → "gpsd" | "COM (COMx)" | "None"
    """

    def __init__(self):
        super().__init__()

        # ── Connection state ───────────────────────────────────────────────────
        self.connected       = False
        self.gpsd_connected  = False
        self.serial_conn     = None
        self.method_str      = "None"
        self.accuracy_str    = "Initializing..."

        # ── Serial port scan management ───────────────────────────────────────
        self.last_scan_time  = 0.0
        self.scan_cooldown   = GPS_SCAN_COOLDOWN

        # ── Previous GPS fix (for delta computation) ──────────────────────────
        self._prev_lat: float | None  = None
        self._prev_lon: float | None  = None
        self._prev_ts:  float | None  = None

        # ── EMA smoothing buffer ──────────────────────────────────────────────
        self._ema_buffer: collections.deque = collections.deque(maxlen=GPS_EMA_WINDOW)
        self._ema_value:  float             = 0.0

        # ── Satellite / quality metadata (updated from NMEA) ──────────────────
        self._sats: int   = 0
        self._hdop: float = 99.9

        # ── Internal lock for EMA / prev-fix state (separate from base lock) ──
        # The base class self.lock protects self.current_speed.
        # self._state_lock protects the EMA buffer and prev-fix variables.
        self._state_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # BACKGROUND ACQUISITION LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def _run(self):
        """
        Main background thread.

        Attempt 1: Connect to gpsd daemon.
        Attempt 2: Scan serial / COM ports for NMEA GPS receivers.
        Once connected, continuously read position and compute speed.
        """
        # 1. Try gpsd first
        if _GPSD_AVAILABLE:
            try:
                _gpsd_module.connect()
                self.gpsd_connected = True
                self.connected      = True
                self.method_str     = "gpsd"
                self.accuracy_str   = "gpsd connected"
                print("[GPS] Connected to gpsd daemon successfully.")
            except Exception as exc:
                self.gpsd_connected = False
                print(f"[GPS] gpsd unavailable ({exc}). Falling back to COM/Serial port scan.")
        else:
            print("[GPS] gpsd-py3 not installed. Using serial NMEA fallback.")

        while self.running:
            if self.gpsd_connected:
                self._gpsd_loop_tick()
            else:
                self._serial_loop_tick()

            # Yield CPU; 10 Hz is sufficient for speed estimation
            time.sleep(0.1)

    # ──────────────────────────────────────────────────────────────────────────
    # GPSD PATH
    # ──────────────────────────────────────────────────────────────────────────

    def _gpsd_loop_tick(self):
        """Single gpsd read cycle — extracts lat/lon/timestamp from TPV packet."""
        try:
            packet = _gpsd_module.get_current()
            mode   = packet.mode  # 0=no val, 1=no fix, 2=2D, 3=3D

            if mode >= 2:
                # Valid fix — extract position
                lat = packet.lat
                lon = packet.lon
                ts  = time.time()

                # Build quality string
                sats_str = getattr(packet, 'sats', '?')
                mode_lbl = {2: "2D Fix", 3: "3D Fix"}.get(mode, "Fix")
                self.accuracy_str = f"{mode_lbl} (Sats: {sats_str})"
                self.connected    = True

                # Compute Haversine speed
                speed_kmh = self._compute_speed(lat, lon, ts)
                if speed_kmh is not None:
                    smoothed = self._apply_ema(speed_kmh)
                    with self.lock:
                        self.current_speed = smoothed
            else:
                # No fix — reset but don't crash
                mode_lbl = {0: "No Value", 1: "No Fix"}.get(mode, "No Fix")
                self.accuracy_str = mode_lbl
                self.connected    = False
                with self._state_lock:
                    self._prev_lat = None
                    self._prev_lon = None
                    self._prev_ts  = None
                with self.lock:
                    self.current_speed = 0.0

        except Exception as exc:
            print(f"[GPS ERROR] gpsd query failed: {exc}")
            self.gpsd_connected = False
            self.connected      = False
            self.method_str     = "None"
            self.accuracy_str   = "gpsd lost"
            with self.lock:
                self.current_speed = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # SERIAL / COM PATH
    # ──────────────────────────────────────────────────────────────────────────

    def _serial_loop_tick(self):
        """Single serial read cycle — scans ports if needed, reads NMEA lines."""
        now = time.time()

        # Scan for a GPS COM port if not yet connected
        if self.serial_conn is None:
            if now - self.last_scan_time >= self.scan_cooldown:
                self.last_scan_time = now
                self.accuracy_str   = "Scanning ports..."
                self._scan_com_ports()

        if self.serial_conn is not None:
            try:
                raw_line = self.serial_conn.readline()
                if raw_line:
                    line = raw_line.decode('ascii', errors='ignore').strip()
                    self._parse_nmea(line)
            except Exception as exc:
                print(f"[GPS ERROR] Serial read error on {self.method_str}: {exc}")
                self.connected    = False
                self.accuracy_str = "Serial error"
                self.method_str   = "None"
                self._close_serial()
        else:
            # No device found yet
            self.connected    = False
            self.accuracy_str = "No GPS device"
            with self.lock:
                self.current_speed = 0.0

    def _scan_com_ports(self):
        """
        Scan all available COM / serial ports for NMEA GPS receivers.
        Tests standard GPS baud rates: 9600, 4800, 115200.
        """
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("[GPS] No serial ports found.")
            self.accuracy_str = "No serial ports"
            return

        for p in ports:
            for baud in [9600, 4800, 115200]:
                print(f"[GPS] Scanning {p.device} @ {baud} baud...")
                try:
                    s = serial.Serial(p.device, baud, timeout=1.0)
                    # Read a few lines and check for NMEA identifier
                    for _ in range(10):
                        raw = s.readline()
                        if not raw:
                            continue
                        line = raw.decode('ascii', errors='ignore')
                        if (line.startswith('$GP')
                                or line.startswith('$GN')
                                or line.startswith('$GL')):
                            self.serial_conn = s
                            self.method_str  = f"COM ({p.device})"
                            self.connected   = True
                            self.accuracy_str = f"COM {p.device} @ {baud}"
                            print(f"[GPS] Active GPS receiver found on {p.device}!")
                            return
                    s.close()
                except Exception:
                    continue

        print("[GPS] No GPS receiver detected on any serial port.")
        self.accuracy_str = "GPS NOT AVAILABLE"

    # ──────────────────────────────────────────────────────────────────────────
    # NMEA PARSING — position extraction (NOT direct speed reading)
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_nmea(self, line: str):
        """
        Parse standard NMEA sentences to extract latitude, longitude, and
        fix quality metadata.  Speed is NOT read from NMEA fields — it is
        derived from successive position deltas via _compute_speed().

        Sentences handled:
          $GPRMC / $GNRMC — position + status (primary source)
          $GPGGA / $GNGGA — position + fix quality + satellites + HDOP
          $GPGLL / $GNGLL — position + status (secondary source)
        """
        parts = line.strip().split(',')
        if not parts or len(parts) < 2:
            return

        sentence = parts[0]

        # ── RMC: Recommended Minimum Navigation Information ──────────────────
        # Fields: [0]=sentence, [1]=UTC, [2]=status(A/V), [3]=lat, [4]=N/S,
        #         [5]=lon, [6]=E/W, [7]=speed(kn), [8]=course, [9]=date, …
        if sentence.endswith('RMC') and len(parts) > 6:
            status = parts[2].strip().upper()
            if status == 'A':
                lat = _dms_to_decimal(parts[3], parts[4].strip().upper())
                lon = _dms_to_decimal(parts[5], parts[6].strip().upper())
                if lat is not None and lon is not None:
                    self.connected = True
                    speed_kmh = self._compute_speed(lat, lon, time.time())
                    if speed_kmh is not None:
                        smoothed = self._apply_ema(speed_kmh)
                        with self.lock:
                            self.current_speed = smoothed
            else:
                # Void fix
                self.connected = False
                self.accuracy_str = "No Fix (RMC Void)"
                with self._state_lock:
                    self._prev_lat = None
                    self._prev_lon = None
                    self._prev_ts  = None
                with self.lock:
                    self.current_speed = 0.0

        # ── GGA: Global Positioning System Fix Data ─────────────────────────
        # Fields: [0]=sentence, [1]=UTC, [2]=lat, [3]=N/S, [4]=lon, [5]=E/W,
        #         [6]=fix(0=invalid,1=GPS,2=DGPS), [7]=sats, [8]=HDOP, [9]=alt
        elif sentence.endswith('GGA') and len(parts) > 8:
            try:
                fix_ind = int(parts[6]) if parts[6].strip().isdigit() else 0
                if fix_ind > 0:
                    lat = _dms_to_decimal(parts[2], parts[3].strip().upper())
                    lon = _dms_to_decimal(parts[4], parts[5].strip().upper())
                    sats_str  = parts[7].strip() or '?'
                    hdop_str  = parts[8].strip() or '?'

                    try:
                        self._sats = int(sats_str)
                        self._hdop = float(hdop_str)
                    except ValueError:
                        pass

                    self.accuracy_str = f"Fix (Sats:{sats_str} HDOP:{hdop_str})"

                    if lat is not None and lon is not None:
                        self.connected = True
                        speed_kmh = self._compute_speed(lat, lon, time.time())
                        if speed_kmh is not None:
                            smoothed = self._apply_ema(speed_kmh)
                            with self.lock:
                                self.current_speed = smoothed
                else:
                    self.connected    = False
                    self.accuracy_str = "No Fix (GGA)"
                    with self._state_lock:
                        self._prev_lat = None
                        self._prev_lon = None
                        self._prev_ts  = None
                    with self.lock:
                        self.current_speed = 0.0
            except (ValueError, IndexError):
                pass

        # ── GLL: Geographic Position — Latitude / Longitude ─────────────────
        # Fields: [0]=sentence, [1]=lat, [2]=N/S, [3]=lon, [4]=E/W,
        #         [5]=UTC, [6]=status(A/V)
        elif sentence.endswith('GLL') and len(parts) > 6:
            status = parts[6].strip().upper() if len(parts) > 6 else ''
            if status == 'A':
                lat = _dms_to_decimal(parts[1], parts[2].strip().upper())
                lon = _dms_to_decimal(parts[3], parts[4].strip().upper())
                if lat is not None and lon is not None:
                    self.connected = True
                    speed_kmh = self._compute_speed(lat, lon, time.time())
                    if speed_kmh is not None:
                        smoothed = self._apply_ema(speed_kmh)
                        with self.lock:
                            self.current_speed = smoothed

    # ──────────────────────────────────────────────────────────────────────────
    # SPEED COMPUTATION ENGINE
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_speed(self, lat: float, lon: float, ts: float) -> float | None:
        """
        Compute instantaneous speed in km/h from the current and previous
        GPS fix using the Haversine formula.

        Validation rules (returns None to skip this sample if violated):
          • dt < GPS_MIN_DT_SEC        → too fast, skip (noise / duplicate)
          • distance > GPS_MAX_JUMP_M  → GPS teleport / bad fix, skip
          • raw_kmh > GPS_MAX_SPEED    → physically impossible, skip
          • raw_kmh < 0               → impossible (sanity guard)

        Returns:
            float  — raw speed in km/h (un-smoothed)
            None   — this sample should be discarded
        """
        with self._state_lock:
            prev_lat = self._prev_lat
            prev_lon = self._prev_lon
            prev_ts  = self._prev_ts

            # Update previous fix for next call
            self._prev_lat = lat
            self._prev_lon = lon
            self._prev_ts  = ts

        # Need at least two fixes to compute speed
        if prev_lat is None or prev_lon is None or prev_ts is None:
            return None

        dt = ts - prev_ts
        if dt < GPS_MIN_DT_SEC:
            # Duplicate or too-fast update — skip
            return None

        # Haversine distance in metres
        dist_m = _haversine(prev_lat, prev_lon, lat, lon)

        # Reject GPS coordinate teleports (bad fix, AGPS glitch, etc.)
        if dist_m > GPS_MAX_JUMP_M:
            print(f"[GPS] Outlier rejected: {dist_m:.0f} m jump in {dt:.2f}s — GPS fix unreliable")
            return None

        # Speed in km/h
        speed_mps = dist_m / dt
        speed_kmh = speed_mps * 3.6

        # Sanity checks
        if speed_kmh < 0.0:
            return None
        if speed_kmh > GPS_MAX_SPEED:
            print(f"[GPS] Outlier rejected: {speed_kmh:.1f} km/h > max {GPS_MAX_SPEED}")
            return None

        return speed_kmh

    # ──────────────────────────────────────────────────────────────────────────
    # EMA SMOOTHER
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_ema(self, raw_kmh: float) -> float:
        """
        Feed a validated raw speed sample into the EMA smoother.

        Formula:  ema_t = α × raw_t + (1 − α) × ema_{t−1}

        The deque stores raw samples for diagnostics; the EMA is computed
        incrementally.  On the first sample the EMA is seeded with raw_kmh
        to avoid a slow ramp-up from zero.

        Returns:
            float — EMA-smoothed speed in km/h
        """
        with self._state_lock:
            self._ema_buffer.append(raw_kmh)
            if len(self._ema_buffer) == 1:
                # Bootstrap: seed EMA with the very first valid sample
                self._ema_value = raw_kmh
            else:
                self._ema_value = (GPS_EMA_ALPHA * raw_kmh
                                   + (1.0 - GPS_EMA_ALPHA) * self._ema_value)
            return self._ema_value

    def _reset_ema(self):
        """Reset the EMA smoother (called on fix loss)."""
        with self._state_lock:
            self._ema_buffer.clear()
            self._ema_value = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # SERIAL HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _close_serial(self):
        """Safely close the serial connection."""
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    # ──────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def stop(self):
        """Stop background thread and release serial port."""
        super().stop()
        self._close_serial()
        self.connected = False
        self.accuracy_str = "Stopped"

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC STATUS INTERFACE (consumed by SpeedManager.get_status_info)
    # ──────────────────────────────────────────────────────────────────────────

    def get_status_info(self) -> dict:
        """
        Returns status metadata for the HUD display panel.

        Keys
        ────
        provider   : "GPS"
        status     : "Connected" | "No Fix" | "GPS NOT AVAILABLE" | "No Speed Source"
        accuracy   : Human-readable fix quality string (sats, HDOP, method)
        """
        if not self.connected:
            # Distinguish between "device found but no fix" vs "device absent"
            if self.serial_conn is not None or self.gpsd_connected:
                status_lbl    = "No Fix"
                accuracy_info = self.accuracy_str
            elif self.accuracy_str == "GPS NOT AVAILABLE":
                status_lbl    = "GPS NOT AVAILABLE"
                accuracy_info = "No GPS device detected"
            else:
                status_lbl    = "No Speed Source"
                accuracy_info = self.accuracy_str
        else:
            status_lbl    = "Connected"
            accuracy_info = f"[{self.method_str}] {self.accuracy_str}"

        return {
            "provider": "GPS",
            "status":   status_lbl,
            "accuracy": accuracy_info,
        }
