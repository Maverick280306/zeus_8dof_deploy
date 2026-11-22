"""
BACK TO STANDING is accepted only where it can do what it says.
"""

from zeus_link import nexus_proto as P


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
