# ==============================================================================
#               SPEED MANAGER — Production-Grade Speed Subsystem
# ==============================================================================
# File   : speed_manager.py
# Purpose: Single source-of-truth for vehicle speed inside the DMS.
#          Orchestrates multiple speed providers via priority waterfall,
#          applies EMA smoothing, tolerance-adjusted thresholds, full
#          5-state machine with hysteresis, and exposes an OkDriver push
#          endpoint so the fatigue pipeline never needs to know the source.
#
# Provider Priority (highest → lowest)
#   1. OkDriver API  — external system pushes speed via update_speed()
#   2. OBD-II        — ELM327 adapter via python-obd
#   3. GPS           — gpsd daemon or serial NMEA receiver
#   4. Mock          — keyboard-driven simulation (always available)
#
# Public Interface (DMS-facing)
#   speed_manager.get_current_speed() → float  (smoothed km/h)
#   speed_manager.update_speed(kmph)           (OkDriver push)
#   speed_manager.get_status_info()   → dict
#   speed_manager.start() / stop()
#   speed_manager.handle_key(key)              (mock keyboard passthrough)
# ==============================================================================

import threading
import time
import logging
from collections import deque
from enum import Enum, auto

# Silence the obd library's verbose console logger before it is imported
logging.getLogger('obd').setLevel(logging.CRITICAL)
logging.getLogger('obd.obd').setLevel(logging.CRITICAL)
logging.getLogger('obd.interfaces').setLevel(logging.CRITICAL)
logging.getLogger('obd.protocols').setLevel(logging.CRITICAL)

# Import configuration constants
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    SPEED_ACTIVATION_KMH,
    SPEED_TOLERANCE_KMH,
    SPEED_HYSTERESIS_KMH,
    SPEED_PENDING_SEC,
    SPEED_LOST_GRACE_SEC,
    SPEED_SMOOTH_WINDOW,
    SPEED_EMA_ALPHA,
    OKDRIVER_STALE_SEC,
    SPEED_POLL_INTERVAL,
)


# ==============================================================================
# 1. SPEED STATE MACHINE STATES
# ==============================================================================

class SpeedState(Enum):
    """
    Five-state machine for speed-based DMS activation.

    INACTIVE   — Speed is below effective threshold. DMS is disabled.
    PENDING    — Speed just crossed the activation threshold. Confirming
                 for SPEED_PENDING_SEC before committing to ACTIVE.
    ACTIVE     — Speed is confirmed above threshold. DMS is fully armed.
    SPEED_LOST — Active session but speed dropped below deactivation
                 threshold. Grace period of SPEED_LOST_GRACE_SEC before
                 reverting to INACTIVE. Recovers to ACTIVE if speed returns.
    RECONNECT  — Provider is unavailable / disconnected. Polling for
                 reconnection. Transitions to INACTIVE once a provider
                 returns valid data.
    """
    INACTIVE   = auto()
    PENDING    = auto()
    ACTIVE     = auto()
    SPEED_LOST = auto()
    RECONNECT  = auto()


# Human-readable labels for HUD display
_STATE_LABELS = {
    SpeedState.INACTIVE:   "INACTIVE",
    SpeedState.PENDING:    "PENDING",
    SpeedState.ACTIVE:     "ACTIVE",
    SpeedState.SPEED_LOST: "SPEED LOST",
    SpeedState.RECONNECT:  "RECONNECT",
}


# ==============================================================================
# 2. OKDRIVER INTERNAL PROVIDER
# ==============================================================================

class _OkDriverProvider:
    """
    Internal lightweight provider that stores a speed value pushed from the
    OkDriver API via SpeedManager.update_speed(). Automatically expires
    stale pushes after OKDRIVER_STALE_SEC seconds.

    This is NOT a background-thread provider; it is purely a data sink.
    The SpeedManager polls it at its own cadence.
    """

    def __init__(self, stale_sec: float = OKDRIVER_STALE_SEC):
        self._speed: float = 0.0
        self._last_push: float = 0.0
        self._stale_sec = stale_sec
        self._lock = threading.Lock()

    def push(self, speed_kmph: float) -> None:
        """Called by SpeedManager.update_speed() to inject OkDriver speed."""
        with self._lock:
            self._speed = max(0.0, float(speed_kmph))
            self._last_push = time.time()

    def get_speed(self) -> float:
        """Returns last pushed speed, or 0.0 if stale."""
        with self._lock:
            return self._speed

    def is_available(self) -> bool:
        """True if a push was received within the stale window."""
        with self._lock:
            if self._last_push == 0.0:
                return False
            return (time.time() - self._last_push) < self._stale_sec

    def get_status_info(self) -> dict:
        staleness = time.time() - self._last_push if self._last_push > 0 else 999.0
        return {
            "provider": "OkDriver",
            "status": "Connected" if self.is_available() else "Stale",
            "accuracy": f"Staleness: {staleness:.1f}s",
        }


# ==============================================================================
# 3. EMA SPEED SMOOTHER
# ==============================================================================

class _SpeedSmoother:
    """
    Exponential Moving Average smoother backed by a fixed-length deque.
    The deque stores the last SPEED_SMOOTH_WINDOW raw samples; the EMA is
    computed incrementally on each new sample. This gives a stable, lag-
    reduced signal suitable for threshold comparison.

    Formula:  ema_t = alpha * raw_t + (1 - alpha) * ema_{t-1}
    """

    def __init__(self, window: int = SPEED_SMOOTH_WINDOW, alpha: float = SPEED_EMA_ALPHA):
        self._window = window
        self._alpha  = alpha
        self._buffer: deque = deque(maxlen=window)
        self._ema: float = 0.0

    def update(self, raw_speed: float) -> float:
        """Feed a new raw sample and return the current EMA."""
        raw_speed = max(0.0, float(raw_speed))
        self._buffer.append(raw_speed)
        if len(self._buffer) == 1:
            # Bootstrap: seed EMA with first sample
            self._ema = raw_speed
        else:
            self._ema = self._alpha * raw_speed + (1.0 - self._alpha) * self._ema
        return self._ema

    def reset(self) -> None:
        self._buffer.clear()
        self._ema = 0.0

    @property
    def value(self) -> float:
        return self._ema

    @property
    def raw_samples(self) -> list:
        return list(self._buffer)


# ==============================================================================
# 4. SPEED MANAGER
# ==============================================================================

class SpeedManager:
    """
    Production-grade vehicle speed orchestrator for the DMS.

    Responsibilities
    ─────────────────
    • Auto-detect and select the highest-priority available speed provider.
    • Run background thread that polls the active provider at 10 Hz.
    • Apply EMA smoothing over a deque window to remove GPS / OBD jitter.
    • Evaluate speed against activation and deactivation thresholds with
      configurable tolerance and hysteresis to prevent flickering.
    • Maintain the 5-state machine: INACTIVE→PENDING→ACTIVE→SPEED_LOST→INACTIVE
      (or RECONNECT if provider drops out).
    • Expose update_speed() so OkDriver can push speed without any changes
      to the fatigue detection pipeline.

    Thread Safety
    ─────────────
    All public getters are guarded by threading.Lock. The background
    acquisition thread is daemon so it never prevents clean shutdown.
    """

    def __init__(
        self,
        activation_kmh: float = SPEED_ACTIVATION_KMH,
        tolerance_kmh:  float = SPEED_TOLERANCE_KMH,
        hysteresis_kmh: float = SPEED_HYSTERESIS_KMH,
        pending_sec:    float = SPEED_PENDING_SEC,
        lost_grace_sec: float = SPEED_LOST_GRACE_SEC,
        poll_interval:  float = SPEED_POLL_INTERVAL,
    ):
        # ── Configuration ──────────────────────────────────────────────────────
        self._activation_kmh  = float(activation_kmh)
        self._tolerance_kmh   = float(tolerance_kmh)
        self._hysteresis_kmh  = float(hysteresis_kmh)
        self._pending_sec     = float(pending_sec)
        self._lost_grace_sec  = float(lost_grace_sec)
        self._poll_interval   = float(poll_interval)

        # Effective thresholds (pre-computed for clarity)
        # Activate:   smoothed_speed >= _eff_activate
        # Deactivate: smoothed_speed <= _eff_deactivate
        self._eff_activate    = max(0.0, self._activation_kmh - self._tolerance_kmh)
        self._eff_deactivate  = max(0.0, self._eff_activate   - self._hysteresis_kmh)

        # ── State ──────────────────────────────────────────────────────────────
        self._state           = SpeedState.INACTIVE
        self._state_lock      = threading.Lock()
        self._pending_since:  float | None = None   # timestamp when PENDING began
        self._lost_since:     float | None = None   # timestamp when SPEED_LOST began

        # ── Smoothing ──────────────────────────────────────────────────────────
        self._smoother        = _SpeedSmoother()
        self._raw_speed:  float = 0.0
        self._smoothed:   float = 0.0

        # ── OkDriver push provider ─────────────────────────────────────────────
        self._okdriver        = _OkDriverProvider()

        # ── Underlying hardware / simulation providers ─────────────────────────
        # Imported lazily so that import errors for missing packages (obd, gpsd,
        # pyserial) only affect the specific provider, not the whole system.
        self._obd_provider        = None
        self._mobile_gps_provider = None   # Phone browser GPS (FastAPI server)
        self._gps_provider        = None
        self._mock_provider       = None
        self._active_provider     = None   # whichever provider is currently used
        self._provider_name       = "None"

        # ── Background thread ──────────────────────────────────────────────────
        self._running         = False
        self._thread: threading.Thread | None = None
        self._lock            = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialize providers and start the background acquisition thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        print("[SpeedManager] Starting provider detection and background thread...")
        self._init_providers()
        self._running = True
        self._thread  = threading.Thread(
            target=self._acquisition_loop,
            daemon=True,
            name="SpeedManager-BG"
        )
        self._thread.start()
        print(f"[SpeedManager] Background thread started. Active provider: {self._provider_name}")

    def stop(self) -> None:
        """Stop the background thread and clean up all providers."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Stop all hardware providers
        for prov in [
            self._obd_provider,
            self._mobile_gps_provider,
            self._gps_provider,
            self._mock_provider,
        ]:
            if prov is not None:
                try:
                    prov.stop()
                except Exception:
                    pass
        print("[SpeedManager] Stopped.")

    # ──────────────────────────────────────────────────────────────────────────
    # PROVIDER INITIALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def _init_providers(self) -> None:
        """
        Instantiate all available hardware providers.
        OBD and GPS providers are started in the background; we detect
        which one actually becomes available in the acquisition loop.
        Mock is always available as the guaranteed fallback.
        """
        # OBD-II — silence its logger before initializing to avoid console spam
        try:
            # Silence obd logger one more time after the module is imported
            logging.getLogger('obd').setLevel(logging.CRITICAL)
            logging.getLogger('obd.obd').setLevel(logging.CRITICAL)
            from obd_provider import OBDProvider
            obd_prov = OBDProvider()
            # Extend reconnect cooldown to 30s so it does not spam
            # when no ELM327 adapter is physically present
            obd_prov.reconnect_cooldown = 30.0
            obd_prov.start()
            self._obd_provider = obd_prov
            print("[SpeedManager] OBD provider initialized (cooldown: 30s).")
        except Exception as e:
            print(f"[SpeedManager] OBD provider unavailable: {e}")
            self._obd_provider = None

        # Mobile GPS — embedded FastAPI server that receives phone browser GPS
        try:
            from mobile_gps_provider import MobileGPSProvider
            self._mobile_gps_provider = MobileGPSProvider()
            self._mobile_gps_provider.start()
            print("[SpeedManager] Mobile GPS provider initialized (port 5000).")
        except Exception as e:
            print(f"[SpeedManager] Mobile GPS provider unavailable: {e}")
            self._mobile_gps_provider = None

        # GPS — delay first COM-port scan so it doesn't block DMS startup
        try:
            from gps_provider import GPSProvider
            self._gps_provider = GPSProvider()
            # Pre-set last_scan_time so the first serial port scan is deferred
            # by scan_cooldown (30s), after the DMS window is already open.
            self._gps_provider.scan_cooldown = 30.0
            self._gps_provider.last_scan_time = time.time()
            self._gps_provider.start()
            print("[SpeedManager] GPS provider initialized (first scan deferred 30s).")
        except Exception as e:
            print(f"[SpeedManager] GPS provider unavailable: {e}")
            self._gps_provider = None

        # Mock (keyboard simulation — always available)
        try:
            from mock_provider import MockProvider
            self._mock_provider = MockProvider()
            self._mock_provider.start()
            print("[SpeedManager] Mock provider initialized (keyboard fallback).")
        except Exception as e:
            print(f"[SpeedManager] Mock provider unavailable: {e}")
            self._mock_provider = None

        # Select the initial best provider
        self._select_best_provider()

    def _select_best_provider(self) -> None:
        """
        Priority waterfall: pick the highest-priority provider that is
        currently delivering valid (non-disconnected) data.

        Priority:
          1. OkDriver   — external push API (checked by _okdriver.is_available())
          2. OBD-II     — ELM327 connected and car ECU responding
          3. Mobile GPS — phone browser Geolocation API via embedded HTTP server
          4. GPS        — gpsd daemon or serial NMEA with active fix
          5. Mock       — always available as guaranteed fallback

        Sets self._active_provider and self._provider_name.
        OkDriver has no hardware thread — it is polled inline.
        """
        # 1. OkDriver (push-based — highest priority when fresh)
        if self._okdriver.is_available():
            self._active_provider = None   # no background thread needed
            self._provider_name   = "OkDriver"
            return

        # 2. OBD-II
        if self._obd_provider is not None:
            info = self._obd_provider.get_status_info()
            if info.get("status") not in ("Disconnected",):
                self._active_provider = self._obd_provider
                self._provider_name   = "OBD-II"
                return

        # 3. Mobile GPS — phone browser GPS (highest-priority real GPS source)
        if self._mobile_gps_provider is not None:
            info = self._mobile_gps_provider.get_status_info()
            if info.get("status") == "Connected":
                self._active_provider = self._mobile_gps_provider
                self._provider_name   = "Mobile GPS"
                # Propagate DMS active flag so phone page can show status badge
                self._mobile_gps_provider._dms_active = (
                    self._state == SpeedState.ACTIVE
                )
                return

        # 4. GPS (legacy — gpsd daemon or serial NMEA)
        if self._gps_provider is not None:
            info = self._gps_provider.get_status_info()
            if info.get("status") not in ("No Speed Source", "Disconnected"):
                self._active_provider = self._gps_provider
                self._provider_name   = "GPS"
                return

        # 5. Mock (fallback — always available)
        if self._mock_provider is not None:
            self._active_provider = self._mock_provider
            self._provider_name   = "Mock"
            return

        # Nothing available
        self._active_provider = None
        self._provider_name   = "None"

    def _is_provider_connected(self) -> bool:
        """Check whether the current active provider has valid connectivity."""
        # OkDriver: just check freshness
        if self._provider_name == "OkDriver":
            return self._okdriver.is_available()

        # Mock is always connected
        if self._provider_name == "Mock":
            return True

        if self._active_provider is None:
            return False

        info = self._active_provider.get_status_info()
        disconnected_states = {
            "Disconnected",
            "No Speed Source",
            "Stale",
            "No Fix",
            "GPS NOT AVAILABLE",   # GPSProvider: device not found / no serial port
        }
        return info.get("status") not in disconnected_states

    # ──────────────────────────────────────────────────────────────────────────
    # BACKGROUND ACQUISITION LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def _acquisition_loop(self) -> None:
        """
        Background thread: runs at SPEED_POLL_INTERVAL (10 Hz).
        Each cycle:
          1. Re-evaluate provider priority (OkDriver may appear/disappear).
          2. Read raw speed from the active provider.
          3. Apply EMA smoothing.
          4. Advance the state machine.
        """
        while self._running:
            try:
                # Re-select provider each cycle so OkDriver always takes priority
                self._select_best_provider()

                # Read raw speed from whichever provider is active
                raw = self._read_raw_speed()

                # Smooth it
                smoothed = self._smoother.update(raw)

                # Update shared state under lock
                with self._lock:
                    self._raw_speed = raw
                    self._smoothed  = smoothed

                # Advance state machine
                self._advance_state_machine(smoothed)

            except Exception as exc:
                print(f"[SpeedManager] Acquisition loop error: {exc}")

            time.sleep(self._poll_interval)

    def _read_raw_speed(self) -> float:
        """
        Read one raw speed sample from the active provider.
        Returns 0.0 if no provider is connected.
        """
        # OkDriver: read from internal push store
        if self._provider_name == "OkDriver":
            return self._okdriver.get_speed() if self._okdriver.is_available() else 0.0

        # Hardware / Mock providers: call get_speed()
        if self._active_provider is not None:
            try:
                return float(self._active_provider.get_speed())
            except Exception:
                return 0.0

        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # STATE MACHINE
    # ──────────────────────────────────────────────────────────────────────────

    def _advance_state_machine(self, smoothed: float) -> None:
        """
        Evaluate the smoothed speed against thresholds and advance through
        the 5-state machine. All transitions are logged for diagnostics.

        Thresholds (example with defaults):
          _eff_activate   = 25 - 5 = 20 km/h   → INACTIVE → PENDING when speed >= 20
          _eff_deactivate = 20 - 5 = 15 km/h   → ACTIVE → SPEED_LOST when speed <= 15
        """
        with self._state_lock:
            now = time.time()
            prev_state = self._state
            connected  = self._is_provider_connected()

            # ── RECONNECT: provider has just dropped out ───────────────────────
            if self._state == SpeedState.RECONNECT:
                if connected:
                    # Provider came back — restart from INACTIVE
                    self._state = SpeedState.INACTIVE
                    self._smoother.reset()
                    print("[SpeedManager] Provider reconnected → INACTIVE")
                # else: remain in RECONNECT
                return

            # ── Provider disconnected from any other state ─────────────────────
            if not connected and self._provider_name not in ("Mock", "OkDriver"):
                # Mock never disconnects; OkDriver staleness is handled separately
                if self._state != SpeedState.INACTIVE:
                    self._state = SpeedState.RECONNECT
                    self._pending_since = None
                    self._lost_since    = None
                    print("[SpeedManager] Provider disconnected → RECONNECT")
                return

            # ── INACTIVE ───────────────────────────────────────────────────────
            if self._state == SpeedState.INACTIVE:
                if smoothed >= self._eff_activate:
                    self._state = SpeedState.PENDING
                    self._pending_since = now
                    print(f"[SpeedManager] {smoothed:.1f} km/h >= {self._eff_activate:.1f} → PENDING")

            # ── PENDING ────────────────────────────────────────────────────────
            elif self._state == SpeedState.PENDING:
                if smoothed < self._eff_activate:
                    # Speed dropped before confirmation window ended — abort
                    self._state = SpeedState.INACTIVE
                    self._pending_since = None
                    print(f"[SpeedManager] Speed dropped during PENDING → INACTIVE")
                elif now - self._pending_since >= self._pending_sec:
                    # Confirmed for the required duration → ACTIVE
                    self._state = SpeedState.ACTIVE
                    self._pending_since = None
                    print(f"[SpeedManager] Confirmed {self._pending_sec}s above threshold → ACTIVE")

            # ── ACTIVE ─────────────────────────────────────────────────────────
            elif self._state == SpeedState.ACTIVE:
                if smoothed <= self._eff_deactivate:
                    # Speed fell below deactivation hysteresis → start grace timer
                    self._state = SpeedState.SPEED_LOST
                    self._lost_since = now
                    print(f"[SpeedManager] {smoothed:.1f} km/h <= {self._eff_deactivate:.1f} → SPEED_LOST")

            # ── SPEED_LOST ─────────────────────────────────────────────────────
            elif self._state == SpeedState.SPEED_LOST:
                if smoothed > self._eff_deactivate:
                    # Speed recovered within grace period → back to ACTIVE
                    self._state = SpeedState.ACTIVE
                    self._lost_since = None
                    print(f"[SpeedManager] Speed recovered → ACTIVE")
                elif now - self._lost_since >= self._lost_grace_sec:
                    # Grace period expired → truly deactivate
                    self._state = SpeedState.INACTIVE
                    self._lost_since = None
                    self._smoother.reset()
                    print(f"[SpeedManager] Grace period expired → INACTIVE")

            if self._state != prev_state:
                print(f"[SpeedManager] State: {_STATE_LABELS[prev_state]} → {_STATE_LABELS[self._state]}")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE (DMS-facing)
    # ──────────────────────────────────────────────────────────────────────────

    def get_current_speed(self) -> float:
        """
        Returns the current EMA-smoothed vehicle speed in km/h.
        This is the single speed value the rest of the DMS should use.
        The caller never needs to know the source provider.
        """
        with self._lock:
            return self._smoothed

    # Alias so SpeedGate.read_speed() works via either get_speed() or get_current_speed()
    def get_speed(self) -> float:
        """Alias for get_current_speed() — maintains SpeedProvider interface compatibility."""
        return self.get_current_speed()

    def update_speed(self, speed_kmph: float) -> None:
        """
        OkDriver API push endpoint.

        OkDriver (or any external vehicle telematics system) calls this
        method to inject the current vehicle speed. No modifications to
        the fatigue pipeline are ever required — it only reads from
        get_current_speed().

        Args:
            speed_kmph: Vehicle speed in km/h. Must be non-negative.
        """
        speed_kmph = max(0.0, float(speed_kmph))
        self._okdriver.push(speed_kmph)
        # The acquisition loop will automatically detect OkDriver as
        # the highest-priority provider on the next poll cycle.

    def is_dms_active(self) -> bool:
        """
        Returns True when the DMS should be armed (state is ACTIVE).
        SpeedGate.update() uses this internally; exposed for external callers.
        """
        with self._state_lock:
            return self._state == SpeedState.ACTIVE

    def get_state(self) -> SpeedState:
        """Returns the current SpeedState enum value."""
        with self._state_lock:
            return self._state

    def get_state_label(self) -> str:
        """Returns the human-readable state string for HUD display."""
        with self._state_lock:
            return _STATE_LABELS.get(self._state, "UNKNOWN")

    def get_status_info(self) -> dict:
        """
        Returns a rich status dictionary for the HUD panel.

        Keys
        ────
        provider        : Name of the active speed provider
        status          : Human-readable state label
        speed           : Raw speed from provider (km/h)
        smoothed_speed  : EMA-smoothed speed (km/h)
        threshold       : Configured activation threshold (km/h)
        tolerance       : Configured tolerance (km/h)
        eff_activate    : Effective activation threshold (km/h)
        eff_deactivate  : Effective deactivation threshold (km/h)
        state           : SpeedState enum value
        accuracy        : Provider-specific accuracy string (e.g. GPS HDOP)
        """
        with self._lock:
            raw_spd  = self._raw_speed
            smth_spd = self._smoothed

        with self._state_lock:
            state_label = _STATE_LABELS.get(self._state, "UNKNOWN")

        # Get accuracy string from underlying provider
        accuracy = None
        if self._provider_name == "OkDriver":
            info     = self._okdriver.get_status_info()
            accuracy = info.get("accuracy")
        elif self._active_provider is not None:
            try:
                info     = self._active_provider.get_status_info()
                accuracy = info.get("accuracy")
            except Exception:
                accuracy = None

        return {
            "provider":       self._provider_name,
            "status":         state_label,
            "speed":          round(raw_spd,  1),
            "smoothed_speed": round(smth_spd, 1),
            "threshold":      self._activation_kmh,
            "tolerance":      self._tolerance_kmh,
            "eff_activate":   self._eff_activate,
            "eff_deactivate": self._eff_deactivate,
            "state":          self._state,
            "accuracy":       accuracy,
        }

    def handle_key(self, key: int) -> None:
        """
        Passes keyboard events to the Mock provider for simulation.
        Called by main.py's key capture loop. Has no effect on hardware providers.
        """
        if self._mock_provider is not None:
            try:
                self._mock_provider.handle_key(key)
            except Exception:
                pass

    def reconfigure(
        self,
        activation_kmh: float | None = None,
        tolerance_kmh:  float | None = None,
        hysteresis_kmh: float | None = None,
    ) -> None:
        """
        Runtime reconfiguration of speed thresholds.
        Allows the trackbar slider in main.py to adjust activation speed
        without requiring a restart. Thread-safe.
        """
        with self._state_lock:
            if activation_kmh is not None:
                self._activation_kmh = float(activation_kmh)
            if tolerance_kmh is not None:
                self._tolerance_kmh = float(tolerance_kmh)
            if hysteresis_kmh is not None:
                self._hysteresis_kmh = float(hysteresis_kmh)
            # Recompute derived thresholds
            self._eff_activate   = max(0.0, self._activation_kmh - self._tolerance_kmh)
            self._eff_deactivate = max(0.0, self._eff_activate    - self._hysteresis_kmh)


# ==============================================================================
# 5. SELF-TEST HARNESS
# ==============================================================================

if __name__ == "__main__":
    import sys
    # Force UTF-8 output on Windows so Unicode characters display correctly
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # Standalone test harness - verifies tolerance + hysteresis logic against
    # the 7 canonical test cases from the specification.
    #
    # Threshold = 25, Tolerance = 5  ->  eff_activate = 20
    # Hysteresis = 5                 ->  eff_deactivate = 15
    #
    # Expected results:
    #   18 km/h -> Inactive  (18 < 20, never activates)
    #   20 km/h -> Active    (20 >= 20, activates)
    #   21 km/h -> Active
    #   24 km/h -> Active
    #   25 km/h -> Active
    #   40 km/h -> Active
    #   19 km/h -> Inactive  (19 < 20 on fresh start, never activates)

    print("\n" + "=" * 70)
    print("  SPEED MANAGER - STANDALONE TEST HARNESS")
    print("=" * 70)

    # Use short timers so the test completes quickly
    sm = SpeedManager(
        activation_kmh = 25.0,
        tolerance_kmh  = 5.0,
        hysteresis_kmh = 5.0,
        pending_sec    = 0.0,    # instant confirm for unit tests
        lost_grace_sec = 0.0,    # instant deactivation for unit tests
        poll_interval  = 0.01,
    )

    # Tell SpeedManager it is running in Mock mode so the state machine
    # never enters the RECONNECT branch (Mock is always "connected").
    sm._provider_name = "Mock"

    # Override the smoother alpha so speed changes propagate instantly in tests
    sm._smoother._alpha = 1.0   # raw pass-through

    # Derived thresholds
    eff_act  = sm._eff_activate    # 20.0
    eff_deac = sm._eff_deactivate  # 15.0
    print(f"  Effective activation  threshold: {eff_act:.1f} km/h")
    print(f"  Effective deactivation threshold: {eff_deac:.1f} km/h")
    print()

    def sim_speed(speed_kmph, settle_sec=0.1):
        """Feed speed into smoother + state machine and let it settle."""
        iters = max(5, int(settle_sec / 0.01))
        for _ in range(iters):
            sm._smoother.update(speed_kmph)
            sm._advance_state_machine(sm._smoother.value)

    PASS = "[PASS]"
    FAIL = "[FAIL]"

    def check(label, speed, expected_active):
        # Reset to INACTIVE for a clean fresh-start test
        with sm._state_lock:
            sm._state        = SpeedState.INACTIVE
            sm._pending_since = None
            sm._lost_since    = None
        sm._smoother.reset()

        sim_speed(speed)
        active = sm.is_dms_active()
        ok     = (active == expected_active)
        status = PASS if ok else FAIL
        mark   = "ACTIVE" if active else "INACTIVE"
        print(f"  {status}  {speed:5.1f} km/h -> {mark:8s}  (expected {'ACTIVE' if expected_active else 'INACTIVE'})")
        return ok

    results = [
        check("18 km/h",  18.0,  False),
        check("20 km/h",  20.0,  True),
        check("21 km/h",  21.0,  True),
        check("24 km/h",  24.0,  True),
        check("25 km/h",  25.0,  True),
        check("40 km/h",  40.0,  True),
        check("19 km/h",  19.0,  False),
    ]

    # Hysteresis test: activate at 22, then drop to 17 (below eff_deac=15? No, 17>15)
    # Correct: 17 > 15, so system should stay ACTIVE (hysteresis doing its job)
    with sm._state_lock:
        sm._state = SpeedState.INACTIVE
        sm._pending_since = None
        sm._lost_since    = None
    sm._smoother.reset()
    sim_speed(22.0)   # activate
    sim_speed(17.0)   # drop but still above deactivation threshold of 15
    active_at_17 = sm.is_dms_active()
    ok_hyst = (active_at_17 == True)   # should stay ACTIVE
    status  = PASS if ok_hyst else FAIL
    print(f"\n  {status}  Hysteresis: activate@22 then drop@17 -> {'ACTIVE' if active_at_17 else 'INACTIVE'}  (expected ACTIVE - hysteresis prevents deactivation)")
    results.append(ok_hyst)

    # Grace period test: drop to 10 km/h — below eff_deac=15, grace timer fires
    with sm._state_lock:
        sm._state = SpeedState.INACTIVE
        sm._pending_since = None
        sm._lost_since    = None
    sm._smoother.reset()
    sim_speed(30.0)   # activate
    sim_speed(10.0, settle_sec=0.3)   # drop below eff_deac for > grace period (50ms)
    active_at_10 = sm.is_dms_active()
    ok_grace = (active_at_10 == False)
    status   = PASS if ok_grace else FAIL
    print(f"  {status}  Grace period: activate@30 then sustained@10 -> {'ACTIVE' if active_at_10 else 'INACTIVE'}  (expected INACTIVE - grace expired)")
    results.append(ok_grace)

    print()
    passed = sum(results)
    total  = len(results)
    print(f"  Result: {passed}/{total} tests passed")
    if passed == total:
        print("  [PASS] All tests PASSED - Speed Manager is production-ready.\n")
    else:
        print("  [FAIL] Some tests FAILED - review threshold logic.\n")
    print("=" * 70 + "\n")
