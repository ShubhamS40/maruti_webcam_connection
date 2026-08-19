import time
import obd
from speed_provider import SpeedProvider

class OBDProvider(SpeedProvider):
    """
    Acquires vehicle speed from an OBD-II ELM327 adapter using the python-obd library.
    Runs asynchronously in a background thread to prevent latency in the video stream.
    """
    def __init__(self):
        super().__init__()
        self.connection = None
        self.connected = False
        self.last_reconnect_attempt = 0.0
        self.reconnect_cooldown = 5.0 # seconds before attempting reconnect on failure

    def _run(self):
        while self.running:
            now = time.time()
            if not self.connected or self.connection is None or not self.connection.is_connected():
                if now - self.last_reconnect_attempt >= self.reconnect_cooldown:
                    self.last_reconnect_attempt = now
                    try:
                        print("[OBD] Attempting to connect to ELM327 adapter...")
                        # obd.OBD() automatically scans serial/COM ports for ELM327
                        # Use debug=False to avoid cluttering standard output
                        self.connection = obd.OBD()
                        if self.connection.is_connected():
                            self.connected = True
                            print("[OBD] Connected successfully!")
                        else:
                            self.connected = False
                            with self.lock:
                                self.current_speed = 0.0
                            print("[OBD] Connection failed (No ELM327 detected or ignition off).")
                    except Exception as e:
                        self.connected = False
                        with self.lock:
                            self.current_speed = 0.0
                        print(f"[OBD ERROR] Connection attempt raised exception: {e}")
            
            if self.connected and self.connection and self.connection.is_connected():
                try:
                    response = self.connection.query(obd.commands.SPEED)
                    if response and not response.is_null():
                        # Extract value and convert safely to km/h magnitude
                        raw_val = response.value
                        try:
                            # pint quantity check
                            speed_kmh = float(raw_val.to('kph').magnitude)
                        except Exception:
                            speed_kmh = float(raw_val.magnitude) if hasattr(raw_val, 'magnitude') else float(raw_val)
                            
                        with self.lock:
                            self.current_speed = speed_kmh
                    else:
                        # Null response or invalid command
                        pass
                except Exception as e:
                    print(f"[OBD ERROR] Failed to query vehicle speed: {e}")
                    # Mark connection as lost so we reconnect on next cycle
                    self.connected = False
                    with self.lock:
                        self.current_speed = 0.0
            
            # Polling rate of 5Hz (every 0.2s) is standard for OBD speed checks
            time.sleep(0.2)

    def stop(self):
        """Clean up OBD connection when stopping."""
        super().stop()
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
            self.connected = False

    def get_status_info(self):
        if not self.connected or self.connection is None or not self.connection.is_connected():
            status_str = "Disconnected"
        else:
            obd_status = self.connection.status()
            if obd_status == obd.OBDStatus.CAR_CONNECTED:
                status_str = "Car Connected"
            elif obd_status == obd.OBDStatus.ELM_CONNECTED:
                status_str = "ELM Connected"
            else:
                status_str = "Connected"
                
        return {
            "provider": "OBD-II",
            "status": status_str,
            "accuracy": None
        }
