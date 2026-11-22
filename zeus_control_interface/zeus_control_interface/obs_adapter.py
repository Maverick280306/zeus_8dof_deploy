"""
What the board reports -> what the network was trained on.

The policy block is close to the training observation, and "close" is the
dangerous part: the two differences below do not crash, they just hand the
network a robot it has never seen. Both were found by reading fusion.c against
zeus_env.py, and sim_check/closed_loop_check.py proves the correction against
MuJoCo.

  1. THE REFERENCE POINT. The estimator's body frame is imu_link
     (zeus_kinematics.h), so pelvis_z and vel_hdg are the IMU's. Training used
     base_link, at the hip centre, 240 mm lower on this robot. Uncorrected, the
     height reads 0.24 m high and every 1 rad/s of torso pitch rate adds 0.24 m/s
     of fake forward speed.

  2. HEADING. Yaw is unobservable to the filter and starts wherever it starts;
     training always began at yaw 0, and quat goes into the network whole. The
     yaw at engage is removed here, so the same robot gives the same observation
     whichever way it was facing when it powered up.

Frame layout, training order (zeus_env.py _obs1), for NJ = 10:

    pelvis_z 1 | quat 4 | gyro 3 | vel_hdg 3 | joint_pos NJ | joint_vel NJ |
    spring_angle 4 | ref_angle NJ | contact 2 | foot_z 2 | phase 1 |
    prev_action NJ | vcmd 1

The first eleven fields are the packet's policy block in the packet's own order,
so nothing is permuted. Stacked three frames, oldest first, 50 Hz apart.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from zeus_link.nexus_proto import POLICY_INDEX, STREAM_HOLD, STREAM_RAMP

from .policy_runtime import Policy

# Foot height is NaN when the leg's joints were unreadable that tick. Training
# never saw NaN, and a held value is close for a few ticks, not for long.
MAX_HELD_FOOT_FRAMES = 5


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v by the rotation of wxyz quaternion q (body -> world)."""
    w, u = q[0], q[1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(q: np.ndarray) -> float:
    """Angle between the torso's z axis and vertical, rad. Blind to yaw."""
    zz = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return math.acos(max(-1.0, min(1.0, float(zz))))


def without_yaw(q: np.ndarray, yaw0: float) -> np.ndarray:
    """q with a rotation of yaw0 about world z taken off the front."""
    c, s = math.cos(yaw0 / 2.0), math.sin(yaw0 / 2.0)       # conjugate of q_z(yaw0)
    w, x, y, z = q
    out = np.array([c * w + s * z, c * x + s * y, c * y - s * x, c * z - s * w])
    return -out if out[0] < 0.0 else out      # q and -q are one rotation; training saw w > 0


class ObsAdapter:

    def __init__(self, policy: Policy) -> None:
        self.p = policy
        self._yaw0: Optional[float] = None
        self._hist = None
        self._foot = None
        self._held = 0
        self.a_prev = np.zeros(policy.num_joints)

    def reset(self) -> None:
        """Call on every engage: history from before it is not this episode."""
        self._yaw0 = None
        self._hist = None
        self._foot = None
        self._held = 0
        self.a_prev = np.zeros(self.p.num_joints)

    def _field(self, state, name: str) -> np.ndarray:
        return np.atleast_1d(np.asarray(getattr(state, name), dtype=np.float64))

    def base_height(self, state, quat: np.ndarray) -> float:
        """base_link above the world datum, from the IMU's own height."""
        return float(state.pelvis_z) - float(rotate(quat, self.p.imu_in_base)[2])

    def frame(self, state) -> Optional[np.ndarray]:
        """One raw 61-value frame, or None if the state cannot be trusted."""
        quat_raw = self._field(state, "quat")
        n = float(np.linalg.norm(quat_raw))
        if not math.isfinite(n) or n < 0.5:
            return None
        quat_raw = quat_raw / n
        if self._yaw0 is None:
            self._yaw0 = quat_yaw(quat_raw)
        quat = without_yaw(quat_raw, self._yaw0)

        foot = self._field(state, "foot_z")
        if np.isfinite(foot).all():
            self._foot, self._held = foot, 0
        else:
            self._held += 1
            if self._foot is None or self._held > MAX_HELD_FOOT_FRAMES:
                return None
            foot = self._foot

        pz = self.base_height(state, quat_raw) - float(foot.min())

        # v_base = v_imu - w x r, in the heading frame the board already used
        gyro = self._field(state, "gyro")
        lat, fwd, up = self._field(state, "vel_hdg")
        w_x_r = rotate(quat_raw, np.cross(gyro, self.p.imu_in_base))
        yaw = quat_yaw(quat_raw)
        c, s = math.cos(yaw), math.sin(yaw)
        fwd -= c * w_x_r[0] + s * w_x_r[1]
        lat -= -s * w_x_r[0] + c * w_x_r[1]
        up -= w_x_r[2]

        out = np.empty(self.p.frame_size, dtype=np.float64)
        out[POLICY_INDEX["pelvis_z"][0]] = pz
        o, k = POLICY_INDEX["quat"]
        out[o:o + k] = quat
        o, k = POLICY_INDEX["gyro"]
        out[o:o + k] = gyro
        o, k = POLICY_INDEX["vel_hdg"]
        out[o:o + k] = lat, fwd, up
        for name in ("joint_pos", "joint_vel", "spring_angle", "ref_angle", "contact"):
            o, k = POLICY_INDEX[name]
            out[o:o + k] = self._field(state, name)
        o, k = POLICY_INDEX["foot_z"]
        out[o:o + k] = foot
        out[POLICY_INDEX["phase"][0]] = float(state.phase)
        o = POLICY_INDEX["phase"][0] + 1
        out[o:o + self.p.num_joints] = self.a_prev
        out[-1] = self.p.gait.speed
        return out if np.isfinite(out).all() else None

    def stack(self, frame: np.ndarray) -> np.ndarray:
        """Training repeats the first frame to fill the history; so does this."""
        self._hist = [frame] * self.p.hist if self._hist is None else self._hist[1:] + [frame]
        return np.concatenate(self._hist)


class ResidualFilter:
    """
    The first-order low-pass that sits INSIDE the training environment, so the
    policy learned on the plant it will drive. Its cut-off is a continuous-time
    number; only alpha depends on the rate this runs at.
    """

    def __init__(self, policy: Policy, rate_hz: float) -> None:
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * policy.act_fc / rate_hz)
        self.scale = policy.act_scale
        self.value = np.zeros(policy.num_joints)
        self.target = np.zeros(policy.num_joints)

    def reset(self) -> None:
        self.value[:] = 0.0
        self.target[:] = 0.0

    def command(self, action: np.ndarray) -> None:
        self.target = np.asarray(action, dtype=np.float64) * self.scale

    def tick(self) -> np.ndarray:
        self.value += self.alpha * (self.target - self.value)
        return self.value


class Guards:
    """
    Whether the board's picture matches the one the policy was trained for. The
    node sends enable true only when engage_blockers() is empty.

    The board's own safety.c decides whether a command is well formed. Only the
    Pi can know whether the STORED GAIT is the one this network was trained on,
    so that check lives here: a robot on the old table would arm happily.
    """

    REF_TOL_RAD = math.radians(1.0)
    PHASE_RATE_TOL = 0.02        # fractional error of the stride clock
    # Started from rest, the policy walked from these phases and fell from every
    # other one tested (sim_check/RESULTS.md). The stride is not symmetric to the
    # network even though the gait is: it was only ever evaluated from phase 0.
    START_WINDOW = (0.95, 0.30)
    # The board sends its phase as float32: its start phase 0.95f arrives as
    # 0.94999999, which a strict comparison put OUTSIDE the window - WALK was
    # refused at exactly the phase the board holds.
    PHASE_EPS = 1e-4
    # Arming sends ref + residual straight to the drives. A leg far from the
    # reference gets there as fast as the gains can take it.
    POSE_TOL_RAD = math.radians(5.0)
    PHASE_MEASURE_TICKS = 500    # 1 kHz ticks to measure it over

    def __init__(self, policy: Policy) -> None:
        self.p = policy
        self._seq0 = None
        self._phase0 = 0.0
        self.phase_rate = None       # strides per second, once measured

    def watch_clock(self, state) -> None:
        # Measure only a clock that is free-running. Through a soft-start ramp
        # or hold the board freezes it at the start phase on purpose, and a
        # rate measured across that would read as a broken stride clock.
        flags = int(getattr(state, "stream_flags", 1))
        if (flags & 1) == 0 or flags & (STREAM_RAMP | STREAM_HOLD):
            self._seq0 = None
            return
        if self._seq0 is None:
            self._seq0, self._phase0 = state.seq, float(state.phase)
            return
        ticks = (state.seq - self._seq0) & 0xFFFFFFFF
        if ticks >= self.PHASE_MEASURE_TICKS:
            self.phase_rate = ((float(state.phase) - self._phase0) % 1.0) / (ticks * 0.001)
            self._seq0, self._phase0 = state.seq, float(state.phase)

    def engage_blockers(self, state, health_names) -> list:
        """Everything: arming and releasing. Empty means the policy may drive."""
        return self.arm_blockers(state, health_names) + self.release_blockers(state)

    def release_blockers(self, state) -> list:
        """
        What stops the policy taking over from a soft-start HOLD: the board
        must be standing at the start pose, at a phase this network was shown
        to walk from, with every joint within POSE_TOL of the reference.

        This is the half of the old single check that a standing robot could
        never pass. The board's ramp is what gets it there now.
        """
        why = []
        lo, hi = self.START_WINDOW
        phase = float(state.phase) % 1.0
        if hi + self.PHASE_EPS < phase < lo - self.PHASE_EPS:
            why.append(f"phase {float(state.phase):.2f} is outside the start window")

        off = np.abs(np.asarray(state.joint_pos, dtype=np.float64)
                     - np.asarray(state.ref_angle, dtype=np.float64))
        if not np.isfinite(off).all() or off.max() > self.POSE_TOL_RAD:
            j = int(np.nanargmax(off)) if np.isfinite(off).any() else 0
            why.append(f"robot is not at the reference pose: {self.p.joint_names[j]} off by "
                       f"{math.degrees(float(off[j])):.1f} deg")
        return why

    def arm_blockers(self, state, health_names) -> list:
        """
        What stops arming at all - into the hold, which moves the robot to the
        start pose. Everything except where the robot is standing.
        """
        why = []
        if not state.stream_flags & 1:          # NexusState.STREAM_GAIT_LIVE
            why.append("gait is not live (STREAM_GAIT_LIVE clear)")
        if state.fused_valid != 2:              # FUSION_OK
            why.append(f"estimator not OK (fused_valid {state.fused_valid}; "
                       "needs ROBOT_CONFIG_CALIBRATED)")
        if state.fk_valid != 3:
            why.append("foot height invalid")
        if state.health:
            why.append("health faults: " + ", ".join(health_names(state.health)))

        err = np.abs(np.asarray(state.ref_angle, dtype=np.float64) - self.p.gait.ref(state.phase))
        if not np.isfinite(err).all() or err.max() > self.REF_TOL_RAD:
            j = int(np.nanargmax(err)) if np.isfinite(err).any() else 0
            why.append(f"stored gait is not the training gait: {self.p.joint_names[j]} off by "
                       f"{math.degrees(float(err[j])):.1f} deg")

        want = 1.0 / self.p.gait.period
        if self.phase_rate is None:
            why.append("still measuring the stride clock")
        elif abs(self.phase_rate - want) > self.PHASE_RATE_TOL * want:
            why.append(f"stride clock {self.phase_rate:.4f} Hz, training {want:.4f} Hz")

        q = np.asarray(state.quat, dtype=np.float64)
        if tilt_of(q / max(float(np.linalg.norm(q)), 1e-6)) > self.p.tilt_stop:
            why.append("torso past the tilt limit")
        return why

    def fallen(self, state, adapter: ObsAdapter) -> Optional[str]:
        """Training's own end-of-episode test. Past it the policy knows nothing."""
        q = np.asarray(state.quat, dtype=np.float64)
        q = q / max(float(np.linalg.norm(q)), 1e-6)
        tilt = tilt_of(q)
        if tilt > self.p.tilt_stop:
            return f"tilt {math.degrees(tilt):.1f} deg"
        z = adapter.base_height(state, q) - np.nanmin(np.asarray(state.foot_z, dtype=np.float64))
        if z < self.p.z_stop:
            return f"base height {z:.2f} m"
        return None
