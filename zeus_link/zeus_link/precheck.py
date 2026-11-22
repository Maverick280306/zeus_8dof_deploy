"""
Pre-flight rules shared by everything that can arm the robot.

ANKLE ZERO. The ankle drives do not keep a usable zero across a reboot - the
left one reads its motor's commutation encoder through 15:1, and after a saved
offset both have come back reporting NaN. The builder zeroes them by hand at
the start of every session, with the feet flat. This is the check that it was
done, and done right: with the robot standing on flat feet, both ankles must
read within ANKLE_ZERO_TOL_DEG of zero.

Why 2 degrees: the policy's own closed-loop check (zeus_control_interface/
sim_check/RESULTS.md) still walks with every joint zero off by 2 deg, degrades
at 3, and falls within 2 s at 5. A lost zero - the right ankle has been seen
6.5 and 10 deg out - is caught; the reading noise (~0.1 deg) and a careful
hand zero (~0.5 deg) are not.

It is only meaningful with the feet flat, which is how the robot stands before
it first arms. So callers check it ONCE, before their first arm, not on every
arm: an arm into the gait's start pose puts the ankles at -14..-19 deg on
purpose.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .nexus_proto import (ACT_TELEM_FRESH, JOINT_INDEX, SAFETY_ARMED, STREAM_HOLD, STREAM_RAMP,
                          STREAM_RETURN)

ANKLE_ZERO_TOL_DEG = 2.0
ANKLES = ("left_ankle_pitch", "right_ankle_pitch")


def ankle_zero_problems(joint_pos: Sequence[float],
                        act_flags: Optional[Sequence[int]] = None,
                        tol_deg: float = ANKLE_ZERO_TOL_DEG) -> List[str]:
    """Empty when both ankles are reporting and read within tol_deg of zero.
    Otherwise one plain sentence per ankle saying what is wrong."""
    why = []
    for name in ANKLES:
        j = JOINT_INDEX[name]
        a = float(joint_pos[j])
        if act_flags is not None and not (int(act_flags[j]) & ACT_TELEM_FRESH):
            why.append(f"{name} is not reporting a position (drive powered?)")
        elif not math.isfinite(a):
            why.append(f"{name} has no position (NaN) - zero it, feet flat")
        elif abs(math.degrees(a)) > tol_deg:
            why.append(f"{name} reads {math.degrees(a):+.1f} deg, more than {tol_deg:g} deg "
                       "from zero - are the feet flat? if they are, re-zero it")
    return why


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
    return True, "going back to the standing pose over 3 s, then STOP to idle the drives"
