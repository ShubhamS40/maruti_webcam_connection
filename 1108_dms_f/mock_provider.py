from speed_provider import SpeedProvider

class MockProvider(SpeedProvider):
    """
    Simulates speed changes using keyboard inputs.
    Useful for testing the speed activation logic without actual vehicle connection.
    """
    def __init__(self):
        super().__init__()
        # Initialize speed to 0.0
        self.current_speed = 0.0

    def _run(self):
        # The mock provider does not require a background thread polling loop,
        # but we respect the interface and keep the thread alive.
        while self.running:
            import time
            time.sleep(0.5)

    def handle_key(self, key):
        """
        Processes key events.
        Key codes supported:
          - Up Arrow: 2490368 (OpenCV Win), 82 (OpenCV Linux/OSX), 38 (VK_UP), ord('w'), ord('u')
          - Down Arrow: 2621440 (OpenCV Win), 84 (OpenCV Linux/OSX), 40 (VK_DOWN), ord('s'), ord('d')
        """
        # Determine if key is up or down arrow (or fallbacks)
        is_up = (key in [2490368, 82, 38, 0x260000] or 
                 key == ord('w') or key == ord('W') or 
                 key == ord('u') or key == ord('U'))
                 
        is_down = (key in [2621440, 84, 40, 0x280000] or 
                   key == ord('s') or key == ord('S') or 
                   key == ord('d') or key == ord('D'))

        if is_up:
            with self.lock:
                self.current_speed = min(200.0, self.current_speed + 5.0)
                print(f"[MOCK SPEED] Speed increased to: {self.current_speed} km/h")
        elif is_down:
            with self.lock:
                self.current_speed = max(0.0, self.current_speed - 5.0)
                print(f"[MOCK SPEED] Speed decreased to: {self.current_speed} km/h")

    def get_status_info(self):
        return {
            "provider": "Mock",
            "status": "Connected",
            "accuracy": None
        }
