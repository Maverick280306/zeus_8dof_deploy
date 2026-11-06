"""
The trained network and everything frozen with it, in plain numpy.

The policy was trained with Stable-Baselines3, which pulls in torch. The Pi does
not need either: inference is four matrix products, and ROS Jazzy's numpy 1.26
cannot load an SB3 pickle made under numpy 2 anyway. scripts/export_policy.py
writes policy.npz; golden.npz holds what SB3 itself answered, and
test/test_policy_runtime.py fails if this file ever answers differently.

Nothing here imports rclpy, so all of it can be tested on any machine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# The 8-joint robot's policy. It was "m1_mukesh_3M" - the TEN-joint Mukesh
# bundle - which on this robot is a policy whose action vector is two joints
# longer than the board has drives.
DEFAULT_BUNDLE = "z1_softland_tol05"


def bundle_dir(name: str = DEFAULT_BUNDLE) -> str:
    """Installed share directory if there is one, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("zeus_control_interface"),
                            "policy", name)
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "policy", name)


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(x) - 1.0)


@dataclass(frozen=True)
class Gait:
    """The reference the policy was trained on, at the one speed it is deployed at."""

    speed: float        # m/s, the vcmd input
    period: float       # s per stride
    q: np.ndarray       # (N, joints) rad, output side, HW joint order
    dq: np.ndarray      # (N, joints) rad/s
    tau: np.ndarray     # (N, joints) Nm, output side

    def ref(self, phase: float) -> np.ndarray:
        """Same interpolation as training, and as the firmware's gait_sample()."""
        n = self.q.shape[0]
        x = (float(phase) % 1.0) * n
        i = int(x)
        f = x - i
        return (1.0 - f) * self.q[i % n] + f * self.q[(i + 1) % n]


class Policy:
    """policy.npz: weights, the frozen observation normaliser, deployment constants."""

    def __init__(self, directory: str | None = None) -> None:
        directory = directory or bundle_dir()
        z = np.load(os.path.join(directory, "policy.npz"), allow_pickle=False)
        self.weights = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        self.obs_mean = z["obs_mean"]
        self.obs_std = np.sqrt(z["obs_var"] + float(z["obs_eps"]))
        self.clip_obs = float(z["clip_obs"])

        self.joint_names = tuple(str(n) for n in z["joint_names"])
        self.spring_order = tuple(str(n) for n in z["spring_order"])
        self.num_joints = len(self.joint_names)
        self.act_scale = z["act_scale"]             # rad per unit of action
        self.act_fc = float(z["act_fc"])            # Hz, first-order residual filter
        self.policy_hz = float(z["policy_hz"])
        self.hist = int(z["hist"])                  # frames stacked, 50 Hz apart
        self.frame_size = int(z["frame_size"])
        self.imu_in_base = z["imu_in_base"]         # m, IMU origin in base_link
        self.tilt_stop = float(z["tilt_stop"])      # rad, training's fall condition
        self.z_stop = float(z["z_stop"])            # m, base height, likewise
        self.gait = Gait(float(z["gait_speed"]), float(z["gait_period"]),
                         z["gait_q"], z["gait_dq"], z["gait_tau"])
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1] from one RAW stacked observation."""
        x = np.clip((np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std,
                    -self.clip_obs, self.clip_obs).astype(np.float32)
        for w, b in self.weights[:-1]:
            x = _elu(w @ x + b)
        w, b = self.weights[-1]
        return np.clip(w @ x + b, -1.0, 1.0)
