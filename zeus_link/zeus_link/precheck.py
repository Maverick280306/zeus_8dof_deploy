"""
Operator rules shared by everything that can arm the robot.

There used to be an ankle-zero pre-check here too: both ankles within 2 deg of
zero before the first arm. Removed 2026-11-23 - the robot is not always
upright with its feet flat, so it was a condition the builder often could not
meet, and it blocked runs for no safety gain. The ankles are zeroed when the
link node starts and whenever ZERO ANKLES is pressed while disarmed.
"""

from __future__ import annotations

from .nexus_proto import (SAFETY_ARMED, STREAM_HOLD, STREAM_RAMP,
                          STREAM_RETURN)


def return_refusal(enabled: bool, safety, flags: int):
    """(ok, message) for a BACK TO STANDING request. Only a held arm can go
    back: during the ramp to the start pose or the hold there. Shared by both
    commanders so they refuse the same things with the same words."""
    if not enabled or safety != SAFETY_ARMED:
        return False, "not armed - nothing to go back from"
    if flags & STREAM_RETURN:
        return True, "already going back to / at the standing pose"
    if not flags & (STREAM_RAMP | STREAM_HOLD):
        return False, ("the gait is running - BACK TO STANDING only works from the start "
                       "pose. Press STOP.")
    return True, "every joint to zero over 3 s, then STOP to idle the drives"
