"""
Vehicle speed activation gate.

Controls DMS activation state based on vehicle speed delivered by a
SpeedManager (or any legacy SpeedProvider).

This module is a THIN ADAPTER. All hysteresis, smoothing, tolerance and
state-machine logic now lives inside SpeedManager. SpeedGate only:
  1. Reads the smoothed speed from the injected speed source.
  2. Checks whether the SpeedManager state machine considers DMS active.
  3. Exposes the same update(speed_limit) signature so main.py is unchanged.

Backward Compatibility
──────────────────────
SpeedGate accepts either:
  • SpeedManager  (new)  — calls .get_current_speed() and .is_dms_active()
  • SpeedProvider (legacy) — falls back to the old hysteresis logic
"""

import time

# Global variable retained for any legacy code that imports get_vehicle_speed()
vehicle_speed: float = 0.0


def get_vehicle_speed() -> float:
    """Returns the current vehicle speed (backward-compatibility stub)."""
    global vehicle_speed
    return float(vehicle_speed)


class SpeedGate:
    """
    Activation gate — decides whether the DMS should run this frame.

    When backed by a SpeedManager, SpeedGate is a pure read-through:
      • Active state is determined entirely by SpeedManager's state machine.
      • The speed_limit parameter from the trackbar reconfigures SpeedManager
        thresholds at runtime so the UI slider still works.

    When backed by a legacy SpeedProvider (no state machine), SpeedGate
    applies its own simple hysteresis to preserve backward compatibility.
    """

    def __init__(self, speed_source):
        """
        Args:
            speed_source: Either a SpeedManager or a legacy SpeedProvider.
        """
        self.speed_source   = speed_source
        self.current_speed  = 0.0
        self.limit          = 20.0
        self.deactivate_limit = 15.0
        self.active         = False
        self._below_since: float | None = None  # used only for legacy fallback

        # Detect whether we have the new SpeedManager interface
        self._is_speed_manager = hasattr(speed_source, "is_dms_active") and callable(
            getattr(speed_source, "is_dms_active", None)
        )

    def read_speed(self) -> float:
        """
        Read the current smoothed speed from the injected source.
        SpeedManager exposes get_current_speed(); legacy providers expose get_speed().
        """
        if self._is_speed_manager:
            return self.speed_source.get_current_speed()
        return self.speed_source.get_speed()

    def update(self, speed_limit: float) -> bool:
        """
        Evaluate whether the DMS should be active this frame.

        Args:
            speed_limit: Activation speed from the UI trackbar (km/h).

        Returns:
            True  → DMS should process this frame.
            False → DMS is inactive; show standby screen.
        """
        global vehicle_speed

        self.limit         = float(max(speed_limit, 0))
        self.current_speed = self.read_speed()
        vehicle_speed      = self.current_speed  # keep global in sync

        # ── SpeedManager path (new production architecture) ──────────────────
        if self._is_speed_manager:
            # Bypass threshold entirely when limit is 0 — DMS always active.
            # This mirrors the legacy path behaviour and prevents the state
            # machine getting stuck in INACTIVE/SPEED_LOST at base speed.
            if self.limit == 0:
                self.active = True
                self._below_since = None
                return True

            # Forward the trackbar value to SpeedManager so it can reconfigure
            # its activation threshold at runtime (tolerance stays fixed from config)
            try:
                self.speed_source.reconfigure(activation_kmh=self.limit)
            except Exception:
                pass   # reconfigure is optional — never crash the DMS

            # SpeedManager's state machine owns the activation decision
            self.active = self.speed_source.is_dms_active()
            return self.active

        # ── Legacy SpeedProvider path (fallback for backward compatibility) ───
        # This block is only reached when SpeedGate wraps an old-style provider.

        # Provider connectivity check
        try:
            status_info  = self.speed_source.get_status_info()
            prov_status  = status_info.get("status", "")
            is_disconnected = prov_status in ("Disconnected", "No Speed Source", "Searching...")
        except Exception:
            is_disconnected = False

        if is_disconnected:
            self.active          = False
            self._below_since    = None
            return False

        # Bypass threshold if limit is 0
        if self.limit == 0:
            self.active       = True
            self._below_since = None
            return True

        # Hysteresis band: deactivate 5 km/h below activation threshold
        self.deactivate_limit = float(max(0.0, self.limit - 5.0))

        if not self.active:
            if self.current_speed >= self.limit:
                self.active          = True
                self._below_since    = None
        else:
            if self.current_speed < self.deactivate_limit:
                if self._below_since is None:
                    self._below_since = time.time()
                elif time.time() - self._below_since > 5.0:
                    self.active       = False
                    self._below_since = None
            else:
                self._below_since = None

        return self.active
