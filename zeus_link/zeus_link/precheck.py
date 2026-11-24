"""
Operator rules shared by everything that can arm the robot.

There used to be an ankle-zero pre-check here too: both ankles within 2 deg of
zero before the first arm. Removed 2026-11-23 - the robot is not always
upright with its feet flat, so it was a condition the builder often could not
meet, and it blocked runs for no safety gain. The ankles are zeroed when the
link node starts and whenever ZERO ANKLES is pressed while disarmed.
"""

from __future__ import annotations

from .nexus_proto import SAFETY_FAULT, STREAM_RETURN


def return_refusal(enabled: bool, safety, flags: int):
    """(ok, message) for a BACK TO STANDING request. Since 2026-11-24 it is
    accepted from ANYWHERE the board can be driven: idle (it arms straight into
    the way back), the ramp, the hold, or the running gait. Refused only with
    no state from the board, or while the board is in a fault - which a STOP
    has to clear first. `enabled` is kept for the callers' signature."""
    if safety is None:
        return False, "no state from the board - nothing to drive back with"
    if safety == SAFETY_FAULT:
        return False, "the board is in a fault - press STOP to clear it, then BACK TO STANDING"
    if flags & STREAM_RETURN:
        return True, "already going back to / at the standing pose"
    return True, "every joint to zero over 3 s, from where the legs are now; STOP to idle the drives"