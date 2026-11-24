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


def _jam_case(err_deg, vel_deg_s, seconds, joint="right_knee_pitch"):
    import math
    from zeus_link.precheck import JamWatch
    w = JamWatch(P.JOINT_NAMES)
    i = P.JOINT_INDEX[joint]
    tgt = [0.0] * P.NUM_JOINTS
    pos = [0.0] * P.NUM_JOINTS
    vel = [0.0] * P.NUM_JOINTS
    tgt[i] = math.radians(err_deg)
    vel[i] = math.radians(vel_deg_s)
    out = None
    t = 0.0
    while t <= seconds + 1e-9:
        out = w.update(tgt, pos, vel, t) or out
        t += 0.004
    return out


def test_jam_the_right_knee_on_2026_11_24_is_caught():
    # Commanded -23, stuck at 6.8 and not moving: 30 deg off, 0 deg/s.
    why = _jam_case(30.0, 0.5, 0.35)
    assert why and "right_knee_pitch jammed" in why


def test_jam_not_tripped_by_normal_walking_or_brief_stalls():
    assert _jam_case(18.0, 0.0, 2.0) is None        # worst normal knee error, even static
    assert _jam_case(25.0, 120.0, 2.0) is None      # far off but moving: catching up
    assert _jam_case(30.0, 0.0, 0.2) is None        # stalled for less than 0.3 s
    assert _jam_case(30.0, 0.0, 2.0, joint="left_ankle_pitch") is None   # ankles: body weight


def test_jam_ignores_an_idle_drive():
    import math
    from zeus_link.precheck import JamWatch
    w = JamWatch(P.JOINT_NAMES)
    tgt = [math.nan] * P.NUM_JOINTS          # nothing commanded
    pos = [0.5] * P.NUM_JOINTS
    assert all(w.update(tgt, pos, [0.0] * P.NUM_JOINTS, t * 0.01) is None for t in range(100))
