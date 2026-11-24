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

# ---- jammed joint -------------------------------------------------------------
#
# 2026-11-24, 18:5x: 0.4 s into a walk the right knee stopped dead at 16 deg
# while commanded toward -7 under 75 Nm, slipped 9 deg, and stopped dead again
# at 7 deg - and every run after that pushed 36-55 Nm into it, because nothing
# looked for a joint that is far from its command and NOT MOVING. That is the
# signature: friction lags a moving command; a jam ignores it.
#
# Thresholds against the first ground walk (2026-11-23): worst normal error
# 12 deg on the hips, 18 on the knees, and only while the joint was moving.
# Ankles are left out - the robot's weight folds them 25-30 deg off target,
# which is real and not a jam.
JAM_ERR_DEG = 20.0
JAM_VEL_DEG_S = 5.0
JAM_SEC = 0.3


class JamWatch:
    """Feed it every armed state; it names a joint that has jammed."""

    def __init__(self, joint_names, skip=("left_ankle_pitch", "right_ankle_pitch")):
        self.names = list(joint_names)
        self.watch = [i for i, n in enumerate(self.names) if n not in skip]
        self.since = {}

    def reset(self) -> None:
        self.since = {}

    def update(self, target_rad, pos_rad, vel_rad_s, now: float):
        """None, or a sentence saying which joint jammed and how."""
        import math
        for i in self.watch:
            t, p, v = target_rad[i], pos_rad[i], vel_rad_s[i]
            if not (math.isfinite(t) and math.isfinite(p) and math.isfinite(v)):
                self.since.pop(i, None)
                continue
            err = math.degrees(t - p)
            if abs(err) > JAM_ERR_DEG and abs(math.degrees(v)) < JAM_VEL_DEG_S:
                t0 = self.since.setdefault(i, now)
                if now - t0 >= JAM_SEC:
                    return (f"{self.names[i]} jammed: commanded {math.degrees(t):.1f} deg, stuck at "
                            f"{math.degrees(p):.1f} deg for {now - t0:.1f} s - check it before "
                            "the next START")
            else:
                self.since.pop(i, None)
        return None
