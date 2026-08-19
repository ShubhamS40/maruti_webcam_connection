"""
Temporal primitives for automotive-style behavioral DMS.

ConsecutiveCounter resets on inactive frames (required for eval windows).
PersistenceCounter is for slow-decay signals (e.g. head pose hold).
"""

from collections import deque


class MovingAverage:
    def __init__(self, maxlen=30):
        self._values = deque(maxlen=maxlen)

    def update(self, value):
        self._values.append(float(value))
        return self.value

    @property
    def value(self):
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reset(self):
        self._values.clear()


class ExponentialSmoother:
    """EMA for risk and attention-linked signals."""

    def __init__(self, alpha=0.1, initial=0.0):
        self.alpha = float(alpha)
        self.value = float(initial)
        self._initialized = False

    def update(self, raw):
        raw = float(raw)
        if not self._initialized:
            self.value = raw
            self._initialized = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * raw
        return self.value

    def reset(self, value=0.0):
        self.value = float(value)
        self._initialized = False


class ConsecutiveCounter:
    """
    Strict consecutive-frame counter.
    Resets to 0 when condition is false — fixes sticky eval/alert bugs.
    """

    def __init__(self, threshold=1):
        self.threshold = max(1, int(threshold))
        self.count = 0

    def set_threshold(self, threshold):
        self.threshold = max(1, int(threshold))

    def update(self, active):
        if active:
            self.count += 1
        else:
            self.count = 0
        return self.satisfied

    @property
    def satisfied(self):
        return self.count >= self.threshold

    def reset(self):
        self.count = 0


class PersistenceCounter:
    """Hold signal with decay (head-down display hold)."""

    def __init__(self, threshold, decay=1):
        self.threshold = max(1, int(threshold))
        self.decay = max(1, int(decay))
        self.count = 0

    def update(self, active):
        if active:
            self.count += 1
        else:
            self.count = max(0, self.count - self.decay)
        return self.triggered

    @property
    def triggered(self):
        return self.count >= self.threshold

    def reset(self):
        self.count = 0
