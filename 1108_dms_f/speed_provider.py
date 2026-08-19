import threading
import time

class SpeedProvider:
    """
    Base class for vehicle speed providers.
    Acquires speed asynchronously in a background thread to prevent blocking camera inference.
    """
    def __init__(self):
        self.current_speed = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        """Starts the background speed acquisition loop."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the background speed acquisition loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _run(self):
        """Core acquisition loop to be overridden by subclasses."""
        while self.running:
            time.sleep(1.0)

    def get_speed(self):
        """Returns the current estimated vehicle speed in km/h."""
        with self.lock:
            return self.current_speed

    def get_status_info(self):
        """
        Returns status metadata dictionary.
        Keys:
          'provider': Name of the provider (str)
          'status': Status representation e.g. Connected/Disconnected (str)
          'accuracy': Accuracy metrics like GPS satellites/HDOP or None
        """
        return {
            "provider": "Base",
            "status": "Inactive",
            "accuracy": None
        }

    def handle_key(self, key):
        """Keyboard event handler for Mock provider simulation."""
        pass
