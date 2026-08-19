"""State labels — driven ONLY by state machine tier, not raw metrics."""

from config import Colors


STATE_COLORS = {
    "MONITOR OFF": Colors.GRAY,
    "CALIBRATING": Colors.CYAN,
    "ATTENTIVE": Colors.GREEN,
    "FATIGUED": Colors.YELLOW,
    "DROWSY": Colors.ORANGE,
    "MICROSLEEP": Colors.RED,
}


def color_for_state(state):
    return STATE_COLORS.get(state, Colors.WHITE)
