"""
BACK TO STANDING is accepted from anywhere the board can drive the legs.
"""

from zeus_link import nexus_proto as P


def test_back_to_standing_from_anywhere_the_board_can_drive():
    # What a failure would mean: the robot stuck mid-stride, holding a bad
    # start pose, or crumpled after a fall, with STOP - every joint dropped -
    # as the only way out (the operator asked for this from anywhere,
    # 2026-11-24).
    from zeus_link.precheck import return_refusal
    A, I = P.SAFETY_ARMED, P.SAFETY_IDLE
    assert return_refusal(True, A, P.STREAM_GAIT_LIVE | P.STREAM_HOLD)[0]   # hold
    assert return_refusal(True, A, P.STREAM_GAIT_LIVE | P.STREAM_RAMP)[0]   # ramp
    assert return_refusal(True, A, P.STREAM_GAIT_LIVE)[0]                   # walking
    assert return_refusal(False, I, P.STREAM_GAIT_LIVE)[0]                  # stopped: arms into it
    assert return_refusal(True, A, P.STREAM_RETURN)[0]                      # already going


def test_back_to_standing_refused_only_without_a_board_to_drive():
    from zeus_link.precheck import return_refusal
    ok, why = return_refusal(True, None, 0)
    assert not ok and "no state" in why
    ok, why = return_refusal(True, P.SAFETY_FAULT, 0)
    assert not ok and "STOP" in why
