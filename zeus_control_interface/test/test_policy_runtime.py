"""
The numpy network answers what SB3 answered.

golden.npz was recorded from the trained policy on real rollouts at the deployed
speed. A failure here means the robot is not running the policy that was
evaluated, whatever the plots look like.
"""

import os

import numpy as np
import pytest

from zeus_control_interface.policy_runtime import Policy, bundle_dir
from zeus_link import nexus_proto as P

BUNDLE = os.path.join(os.path.dirname(__file__), "..", "policy", "m1_mukesh_3M")


@pytest.fixture(scope="module")
def policy():
    return Policy(BUNDLE)


def test_matches_sb3_on_recorded_rollouts(policy):
    g = np.load(os.path.join(BUNDLE, "golden.npz"))
    got = np.array([policy.act(o) for o in g["obs"]])
    assert np.abs(got - g["action"]).max() < 1e-5


def test_the_policy_is_about_the_protocols_robot(policy):
    assert policy.joint_names == tuple(P.JOINT_NAMES)
    assert policy.spring_order == tuple(P.SPRING_NAMES)
    assert policy.frame_size == P.POLICY_COUNT + P.NUM_JOINTS + 1
    assert policy.obs_mean.size == policy.frame_size * policy.hist


def test_residual_stays_inside_what_the_board_accepts(policy):
    # Full action on every joint must not reach the 0.1 turn the board rejects.
    assert policy.act_scale.max() < 0.1 * 2.0 * np.pi


def test_bad_input_cannot_exceed_the_action_range(policy):
    a = policy.act(np.full(policy.obs_mean.size, 1e9))
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0


def test_gait_interpolates_and_wraps_like_the_firmware(policy):
    g = policy.gait
    n = g.q.shape[0]
    assert g.ref(0.0) == pytest.approx(g.q[0])
    assert g.ref(0.5 / n) == pytest.approx(0.5 * (g.q[0] + g.q[1]))
    assert g.ref((n - 0.5) / n) == pytest.approx(0.5 * (g.q[n - 1] + g.q[0]))   # the seam
    assert g.ref(1.25) == pytest.approx(g.ref(0.25))


def test_source_tree_is_found_without_ros():
    assert os.path.exists(os.path.join(bundle_dir(), "policy.npz"))
