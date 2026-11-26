"""
Board state -> training observation, hand-built with the answer written first.

Each of these is a way to feed the network a wrong robot without anything
crashing, so each states what it would mean on the real one.
"""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from zeus_control_interface.obs_adapter import (Guards, MAX_HELD_FOOT_FRAMES, ObsAdapter,
                                                ResidualFilter, quat_yaw, rotate, without_yaw)
from zeus_control_interface.policy_runtime import Policy
from zeus_link import nexus_proto as P

BUNDLE = os.path.join(os.path.dirname(__file__), "..", "policy", "z1_softland_tol05")
IX = P.POLICY_INDEX

# Every joint count below comes from the protocol, never a literal. These
# fixtures were written with a hardcoded 10 for the ankle-roll robot, which
# passed only while the protocol was also ten joints - so the day the workspace
# went back to the real 8-joint hardware they broke, in the adapter rather than
# in an assertion, with a numpy broadcast error. The modules under test size
# everything from the policy bundle's own joint_names; the tests should too.


@pytest.fixture(scope="module")
def policy():
    return Policy(BUNDLE)


def state(policy, **over):
    phase = over.pop("phase", 0.1)
    v = dict(seq=0, pelvis_z=0.93, quat=[1.0, 0.0, 0.0, 0.0], gyro=[0.0, 0.0, 0.0],
             vel_hdg=[0.0, 0.3, 0.0],
             joint_pos=list(policy.gait.ref(phase)),
             joint_vel=[0.02 * i for i in range(P.NUM_JOINTS)],
             spring_angle=[1e-3, 2e-3, 3e-3, 4e-3], ref_angle=list(policy.gait.ref(phase)),
             contact=[1.0, 0.0], foot_z=[0.05, 0.0], phase=phase, stream_flags=1, fused_valid=2,
             fk_valid=3, health=0)
    v.update(over)
    return SimpleNamespace(**v)


def test_frame_is_the_packet_block_plus_prev_action_and_speed(policy):
    a = ObsAdapter(policy)
    a.a_prev = np.arange(P.NUM_JOINTS) * 0.1
    f = a.frame(state(policy))
    assert f.size == policy.frame_size
    o, n = IX["joint_vel"]
    assert f[o:o + n] == pytest.approx([0.02 * i for i in range(P.NUM_JOINTS)])
    o, n = IX["contact"]
    assert list(f[o:o + n]) == [1.0, 0.0]            # left, then right
    o, n = IX["foot_z"]
    assert list(f[o:o + n]) == [0.05, 0.0]           # right, then left
    assert f[IX["phase"][0]] == pytest.approx(0.1)
    assert f[P.POLICY_COUNT:P.POLICY_COUNT + P.NUM_JOINTS] == pytest.approx(a.a_prev)
    assert f[-1] == pytest.approx(policy.gait.speed)


def test_height_is_the_hips_not_the_imu(policy):
    # IMU 0.93 m up, upright, the IMU imu_in_base[2] above the hips, lower foot
    # at 0: the
    # network must read 0.6897. Left alone it sees a robot 0.24 m too tall.
    f = ObsAdapter(policy).frame(state(policy))
    dz = float(policy.imu_in_base[2])
    assert f[IX["pelvis_z"][0]] == pytest.approx(0.93 - dz, abs=1e-4)


def test_height_is_measured_from_the_lower_foot(policy):
    f = ObsAdapter(policy).frame(state(policy, foot_z=[0.05, 0.02]))
    dz = float(policy.imu_in_base[2])
    assert f[IX["pelvis_z"][0]] == pytest.approx(0.93 - dz - 0.02, abs=1e-4)


def test_torso_pitch_rate_is_not_forward_speed(policy):
    # Pitching forward at 1 rad/s with the hips still: the IMU, 0.24 m above
    # them, moves 0.24 m/s forward. The base speed is the only thing trained on.
    r = policy.imu_in_base
    f = ObsAdapter(policy).frame(state(policy, gyro=[0.0, 1.0, 0.0], vel_hdg=[0.0, r[2], 0.0]))
    o, _ = IX["vel_hdg"]
    assert f[o + 1] == pytest.approx(0.0, abs=1e-3)


def test_yaw_at_engage_is_removed_and_later_yaw_is_kept(policy):
    y = 1.1
    q0 = [math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)]
    a = ObsAdapter(policy)
    o, n = IX["quat"]
    assert a.frame(state(policy, quat=q0))[o:o + n] == pytest.approx([1, 0, 0, 0], abs=1e-9)

    dy = 0.2
    q1 = [math.cos((y + dy) / 2), 0.0, 0.0, math.sin((y + dy) / 2)]
    got = a.frame(state(policy, quat=q1))[o:o + n]
    assert quat_yaw(got) == pytest.approx(dy, abs=1e-9)


def test_quat_sign_is_the_one_training_saw():
    assert without_yaw(np.array([-1.0, 0.0, 0.0, 0.0]), 0.0)[0] == pytest.approx(1.0)


def test_rotate_is_body_to_world():
    h = math.pi / 4
    q = np.array([math.cos(h), 0.0, math.sin(h), 0.0])          # 90 deg about +y
    assert rotate(q, np.array([0.0, 0.0, 1.0])) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_short_foot_dropout_holds_then_gives_up(policy):
    a = ObsAdapter(policy)
    a.frame(state(policy))
    nan = [float("nan"), float("nan")]
    for _ in range(MAX_HELD_FOOT_FRAMES):
        assert a.frame(state(policy, foot_z=nan)) is not None
    assert a.frame(state(policy, foot_z=nan)) is None


def test_garbage_state_is_refused_not_passed_on(policy):
    assert ObsAdapter(policy).frame(state(policy, quat=[0.0, 0.0, 0.0, 0.0])) is None
    bad = [float("nan")] * P.NUM_JOINTS
    assert ObsAdapter(policy).frame(state(policy, joint_pos=bad)) is None


def test_history_is_oldest_first_and_starts_full(policy):
    a = ObsAdapter(policy)
    f = np.arange(float(policy.frame_size))
    s = a.stack(f)
    n = policy.frame_size
    assert s.size == n * policy.hist and (s[:n] == f).all() and (s[2 * n:] == f).all()
    s = a.stack(f + 1)
    assert (s[:n] == f).all() and (s[2 * n:] == f + 1).all()


def test_filter_has_the_training_time_constant(policy):
    # A step reaches 1 - 1/e of its value in 1 / (2 pi fc) seconds, whatever the
    # rate. Run at a different rate than training and the constant must not move.
    f = ResidualFilter(policy, 250.0)
    f.command(np.ones(P.NUM_JOINTS))
    ticks = int(round(250.0 / (2 * math.pi * policy.act_fc)))
    for _ in range(ticks):
        v = f.tick()
    assert v / policy.act_scale == pytest.approx(1.0 - math.exp(-1.0), abs=0.03)


def test_guards_pass_a_board_playing_the_training_gait(policy):
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    # Both feet down: released on one, the gait lifts it (2026-11-25).
    assert g.engage_blockers(state(policy, contact=[1.0, 1.0]), P.HEALTH_NAMES.get) == []


@pytest.mark.parametrize("over,word", [
    (dict(fused_valid=1), "estimator"),
    (dict(stream_flags=0), "gait"),
    (dict(fk_valid=1), "foot"),
    (dict(quat=[0.9, 0.43, 0.0, 0.0]), "tilt"),
])
def test_each_guard_blocks(policy, over, word):
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    why = g.engage_blockers(state(policy, **over), lambda m: [])
    assert any(word in w for w in why)


def test_the_old_human_gait_table_is_refused(policy):
    # Today's firmware plays final_humangait.xlsx. It would arm happily, and the
    # network would be correcting a stride it has never seen.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    ref = list(policy.gait.ref(0.1))
    ref[3] += math.radians(2.0)
    why = g.engage_blockers(state(policy, ref_angle=ref), lambda m: [])
    assert any("left_ankle_pitch" in w for w in why)


@pytest.mark.parametrize("phase,ok", [(0.0, True), (0.97, True), (0.29, True), (0.5, False),
                                      (0.31, False), (0.94, False),
                                      # the board's start phase 0.95f as it arrives,
                                      # float32: 0.94999999 - must be inside
                                      (float(np.float32(0.95)), True)])
def test_engage_only_from_a_phase_it_can_start_from(policy, phase, ok):
    # Outside the window the policy fell within 1.5 s of starting from rest.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    why = g.engage_blockers(state(policy, phase=phase), lambda m: [])
    assert (not any("start window" in w for w in why)) == ok


def test_a_leg_away_from_the_reference_is_refused(policy):
    # Arming would send it straight to the reference, at whatever speed the
    # gains allow. It is the leg-through-the-stand failure.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    # By NAME, not by index: 7 was right_knee_pitch on the ten-joint robot and
    # is right_ankle_pitch on this one, so the literal perturbed a joint the
    # assertion did not name and the test still passed for the wrong reason.
    q = list(policy.gait.ref(0.1))
    q[P.JOINT_INDEX["right_knee_pitch"]] += math.radians(8.0)
    why = g.engage_blockers(state(policy, joint_pos=q), lambda m: [])
    assert any("right_knee_pitch" in w and "reference pose" in w for w in why)


def test_a_clock_at_the_wrong_rate_is_refused(policy):
    # The old GAIT_CYCLE_S was 1.26 s, this stride 1.858 s. The table can be right
    # and still play a quarter too fast.
    g = Guards(policy)
    dp = 0.5 / 1.26
    g.watch_clock(SimpleNamespace(seq=0, phase=0.0))
    g.watch_clock(SimpleNamespace(seq=500, phase=dp))
    why = g.engage_blockers(state(policy), lambda m: [])
    assert any("stride clock" in w for w in why)


def test_stride_clock_is_measured_across_the_wrap(policy):
    g = Guards(policy)
    g.watch_clock(SimpleNamespace(seq=0, phase=0.9))
    g.watch_clock(SimpleNamespace(seq=500, phase=(0.9 + 0.5 / policy.gait.period) % 1.0))
    assert g.phase_rate == pytest.approx(1.0 / policy.gait.period, rel=1e-6)


def test_a_standing_robot_may_arm_but_not_release(policy):
    # The whole point of the soft start: a robot standing at zero is 20-35 deg
    # from the start pose. That must NOT stop it arming into the hold - the
    # board's ramp is what moves it - and MUST stop the policy taking over
    # until it has arrived.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    standing = state(policy, joint_pos=[0.0] * P.NUM_JOINTS, phase=0.0,
                     ref_angle=list(policy.gait.ref(0.0)))
    assert g.arm_blockers(standing, lambda m: []) == []
    why = g.release_blockers(standing)
    assert any("reference pose" in w for w in why)

    arrived = state(policy, joint_pos=list(policy.gait.ref(0.0)), phase=0.0,
                    ref_angle=list(policy.gait.ref(0.0)), contact=[1.0, 1.0])
    assert g.release_blockers(arrived) == []


def test_the_stride_clock_is_not_measured_across_a_hold(policy):
    # Held at the start phase the board's clock stands still on purpose. A rate
    # measured across that would read as a stride clock at 0 Hz and block
    # release forever - or, measured into a hold, a rate that is neither.
    g = Guards(policy)
    rate = 1.0 / policy.gait.period
    g.watch_clock(SimpleNamespace(seq=0, phase=0.2, stream_flags=P.STREAM_GAIT_LIVE))
    g.watch_clock(SimpleNamespace(seq=500, phase=(0.2 + 0.5 * rate) % 1.0,
                                  stream_flags=P.STREAM_GAIT_LIVE))
    assert g.phase_rate == pytest.approx(rate, rel=1e-6)

    held = P.STREAM_GAIT_LIVE | P.STREAM_HOLD
    for k in range(5):
        g.watch_clock(SimpleNamespace(seq=600 + 500 * k, phase=0.0, stream_flags=held))
    assert g.phase_rate == pytest.approx(rate, rel=1e-6)


def test_fall_conditions_are_the_training_ones(policy):
    g, a = Guards(policy), ObsAdapter(policy)
    assert g.fallen(state(policy), a) is None
    th = policy.tilt_stop + 0.02
    q = [math.cos(th / 2), math.sin(th / 2), 0.0, 0.0]
    assert "tilt" in g.fallen(state(policy, quat=q), a)
    low = 0.39 + policy.imu_in_base[2]
    assert "height" in g.fallen(state(policy, pelvis_z=low), a)


def test_an_ankle_pushed_by_the_floor_still_releases(policy):
    # 2026-11-24: holding the start pose, the robot's weight pushed the right
    # ankle 8.4 deg off its reference. The trained ankle gains are soft and the
    # floor sets the ankle, so 5 deg could never pass; ankles get 10.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    q = list(policy.gait.ref(0.1))
    q[P.JOINT_INDEX["right_ankle_pitch"]] += math.radians(8.4)
    why = g.release_blockers(state(policy, joint_pos=q))
    assert not any("reference pose" in w for w in why)
    q[P.JOINT_INDEX["right_ankle_pitch"]] += math.radians(3.0)        # 11.4: too far
    why = g.release_blockers(state(policy, joint_pos=q))
    assert any("right_ankle_pitch" in w and "reference pose" in w for w in why)


@pytest.mark.parametrize("contact,ok", [([1.0, 1.0], True), ([1.0, 0.0], False),
                                        ([0.0, 1.0], False), ([0.0, 0.0], False)])
def test_walk_is_released_only_on_both_feet(policy, contact, ok):
    # 2026-11-25: released on the left foot alone, and the gait's first move
    # lifts the left foot - the robot rolled 8 -> 16 deg in 0.9 s.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    why = g.release_blockers(state(policy, contact=contact))
    assert (not any("both feet" in w for w in why)) == ok


@pytest.mark.parametrize("lean_deg,ok", [(0.0, True), (2.5, True), (3.5, False), (7.6, False)])
def test_walk_is_released_only_upright(policy, lean_deg, ok):
    # 2026-11-25: released leaning 7.6 deg onto one foot, both hips stalled
    # against the robot's own weight and it rolled over in 1 s.
    g = Guards(policy)
    g.phase_rate = 1.0 / policy.gait.period
    h = math.radians(lean_deg) / 2
    why = g.release_blockers(state(policy, quat=[math.cos(h), math.sin(h), 0.0, 0.0],
                                   contact=[1.0, 1.0]))
    assert (not any("leaning" in w for w in why)) == ok
