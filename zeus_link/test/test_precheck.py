"""
The ankle-zero pre-check says yes only when it should.

What a failure would mean on the robot: an ankle whose zero was lost (seen 6.5
and 10 deg out on the right one) arms anyway, and every gait target for that
foot is off by the same amount - the policy's own simulation falls at 5 deg of
joint-zero error. Or the opposite: a correctly zeroed robot refused for noise.
"""

import math

from zeus_link import nexus_proto as P
from zeus_link.precheck import ANKLE_ZERO_TOL_DEG, ankle_zero_problems

L = P.JOINT_INDEX["left_ankle_pitch"]
R = P.JOINT_INDEX["right_ankle_pitch"]
FRESH = [P.ACT_TELEM_FRESH] * P.NUM_JOINTS


def pose(left_deg=0.0, right_deg=0.0):
    q = [0.0] * P.NUM_JOINTS
    q[L], q[R] = math.radians(left_deg), math.radians(right_deg)
    return q


def test_a_good_zero_passes():
    # Hand-zeroed feet sit within half a degree; readings wander 0.1 deg.
    assert ankle_zero_problems(pose(0.5, -0.4), FRESH) == []
    assert ankle_zero_problems(pose(ANKLE_ZERO_TOL_DEG - 0.01, 0.0), FRESH) == []


def test_a_lost_zero_is_named():
    # The right ankle read -6.5 and later -10 deg with the foot flat.
    why = ankle_zero_problems(pose(0.0, -6.5), FRESH)
    assert len(why) == 1 and "right_ankle_pitch" in why[0] and "-6.5" in why[0]
    why = ankle_zero_problems(pose(10.0, -10.0), FRESH)
    assert len(why) == 2


def test_nan_is_refused_not_treated_as_zero():
    # After a reboot both ankle drives reported NaN. NaN compares false with
    # everything, so a naive "abs(a) > tol" would let it through.
    q = pose()
    q[L] = float("nan")
    why = ankle_zero_problems(q, FRESH)
    assert len(why) == 1 and "left_ankle_pitch" in why[0] and "NaN" in why[0]


def test_a_silent_drive_is_refused_even_at_zero():
    # Board just booted, drives unpowered: positions read 0.00 because nothing
    # ever arrived, not because the ankles are at zero.
    flags = list(FRESH)
    flags[R] = 0
    why = ankle_zero_problems(pose(), flags)
    assert len(why) == 1 and "right_ankle_pitch" in why[0] and "not reporting" in why[0]


def test_other_joints_do_not_matter():
    # Only the ankles lose their zero; the others have absolute encoders.
    q = pose()
    q[P.JOINT_INDEX["right_hip_roll"]] = math.radians(4.0)
    assert ankle_zero_problems(q, FRESH) == []


def test_back_to_standing_only_from_a_held_arm():
    # What a failure would mean: RETURN accepted mid-stride (the board would
    # ignore it, but the operator would believe the robot was going back), or
    # refused at the start pose, leaving STOP - every joint dropped from a
    # 35 deg knee bend - as the only way down.
    from zeus_link.precheck import return_refusal
    A = P.SAFETY_ARMED
    assert return_refusal(True, A, P.STREAM_GAIT_LIVE | P.STREAM_HOLD)[0]
    assert return_refusal(True, A, P.STREAM_GAIT_LIVE | P.STREAM_RAMP)[0]
    assert return_refusal(True, A, P.STREAM_RETURN)[0]                 # already going
    ok, why = return_refusal(True, A, P.STREAM_GAIT_LIVE)               # walking
    assert not ok and "STOP" in why
    assert not return_refusal(False, A, P.STREAM_HOLD)[0]               # not enabled
    assert not return_refusal(True, P.SAFETY_IDLE, P.STREAM_HOLD)[0]    # not armed
