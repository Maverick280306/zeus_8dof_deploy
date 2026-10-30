#!/usr/bin/env python3
"""
The deployed Pi code in the loop against the training plant.

Runs on the TRAINING machine (MuJoCo, biped_walk_rl). Nothing from SB3 is used:
MuJoCo stands in for the STM32, publishing what the BOARD would publish - height
and velocity AT THE IMU, an arbitrary starting yaw - and only obs_adapter,
policy_runtime and ResidualFilter decide the residual, at the node's 250/50 Hz.

    cd biped_walk_rl/platforms/zeus
    ZEUS_VARIANT=mukesh QUIET_LIMITS=1 python <repo>/zeus_control_interface/sim_check/closed_loop_check.py
"""
import copy
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [os.path.join(REPO, "zeus_control_interface"), os.path.join(REPO, "zeus_link"),
                os.getcwd()]
os.environ.setdefault("QUIET_LIMITS", "1")

import mujoco                                                                  # noqa: E402
import zeus_env as ze                                                          # noqa: E402
import zeus_model as zm                                                        # noqa: E402
from zeus_control_interface.obs_adapter import ObsAdapter, ResidualFilter, rotate  # noqa: E402
from zeus_control_interface.policy_runtime import Policy                       # noqa: E402

PUB_HZ = 250
PUB_PER_POLICY = 5
PHYS_PER_PUB = 2      # 500 Hz physics / 250 Hz publish


def quat_mul(a, b):
    aw, av, bw, bv = a[0], a[1:], b[0], b[1:]
    return np.r_[aw * bw - av @ bv, aw * bv + bw * av + np.cross(av, bv)]


class Sim:
    """ZeusWalkEnv's physics with the action path replaced by the deployed one."""

    def __init__(self, policy, ff=1.0, delay_pub=0, gain=1.0, world_yaw=0.7, res_on=True,
                 zero_err_deg=0.0, imu_pitch_deg=0.0, correct_imu=True):
        self.env = ze.ZeusWalkEnv(LIB, dr=False, seed=7)
        self.p, self.ff, self.gain = policy, ff, gain
        self.yaw = world_yaw
        self.res_on = res_on
        self.delay = [np.zeros(ze.NJ) for _ in range(delay_pub + 1)]
        self.adapter = ObsAdapter(policy)
        if not correct_imu:          # what feeding the board's values in raw would do
            self.adapter.p = copy.copy(policy)
            self.adapter.p.imu_in_base = np.zeros(3)
        # a fixed per-joint calibration error, same draw for every magnitude
        self.zero = np.radians(zero_err_deg) * np.random.default_rng(11).choice([-1.0, 1.0], ze.NJ)
        h = math.radians(imu_pitch_deg) / 2
        self.imu_err = np.array([math.cos(h), 0.0, math.sin(h), 0.0])
        self.filter = ResidualFilter(policy, PUB_HZ)
        self.site = self.env.m.site("imu").id

    def reset(self, phase, at_rest=False):
        e = self.env
        e.set_vcmd(self.p.gait.speed)
        obs, _ = e.reset(seed=3, options={"phase": phase})
        if at_rest:
            e.d.qvel[:] = 0
            mujoco.mj_forward(e.m, e.d)
            e._cl, e._cr, _ = e._contacts()
            e.hist = None
            obs = e._obs()
        self.adapter.reset()
        self.filter.reset()
        return obs

    def board_state(self):
        """What the packet would carry, from true MuJoCo state."""
        e, d = self.env, self.env.d
        q, dq = e._joint_out()
        q = q + self.zero
        quat = quat_mul(d.sensordata[e.s_quat:e.s_quat + 4].copy(), self.imu_err)
        gyro = d.sensordata[e.s_gyro:e.s_gyro + 3].copy()
        v_imu = d.qvel[:3] + rotate(quat, np.cross(gyro, self.p.imu_in_base))
        # the board's world is the sim's turned by an arbitrary yaw
        c, s = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        w, x, y, z = quat
        board_q = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        cy, sy = math.cos(yaw), math.sin(yaw)
        fr, fl = d.site_xpos[e.site_r][2], d.site_xpos[e.site_l][2]
        return SimpleNamespace(
            pelvis_z=d.site_xpos[self.site][2], quat=board_q, gyro=gyro,
            vel_hdg=[-sy * v_imu[0] + cy * v_imu[1], cy * v_imu[0] + sy * v_imu[1], v_imu[2]],
            joint_pos=q, joint_vel=dq, spring_angle=d.qpos[e.sqa].copy(),
            ref_angle=e._ref(e.phase, "q_joints"), contact=[float(e._cl), float(e._cr)],
            foot_z=[fr, fl], phase=e.phase % 1.0)

    def physics(self, residual):
        """Two 500 Hz substeps of zeus_env.step()'s actuator law."""
        e, d, m = self.env, self.env.d, self.env.m
        for _ in range(PHYS_PER_PUB):
            ph = e.phase
            e.phase += ze.DT_PHYS / e.period
            tff = e._ref(ph, "tau_joints") * self.ff
            # a zero error is one frame for the sensor AND the command: the drive
            # believes q + zero, so it holds the true joint off by -zero
            tgt = e._ref(ph, "q_joints") + residual - self.zero
            tgt[e.sea_i] -= np.clip(tff[e.sea_i] / zm.K_SPRING, -zm.SPRING_TRAVEL, zm.SPRING_TRAVEL)
            dqr = e._ref(ph, "dq_joints")
            tau = np.clip(ze.KP * self.gain * (tgt - d.qpos[e.qa])
                          + ze.KD * self.gain * (dqr - d.qvel[e.va]) + tff,
                          -zm.TAU_PEAK, zm.TAU_PEAK)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        e._cl, e._cr, _ = e._contacts()

    def fallen(self):
        d = self.env.d
        return ze._quat_tilt(*d.qpos[3:7]) > ze.TILT_TERM or d.qpos[2] < 0.40

    def run(self, secs=20.0, phase=0.0, at_rest=False, check_obs=False):
        env_obs = self.reset(phase, at_rest)
        tilt, vx, err, x0, res = [], [], [], self.env.d.qpos[0], []
        n = int(secs * PUB_HZ)
        for i in range(n):
            if i % PUB_PER_POLICY == 0:
                st = self.board_state()
                f = self.adapter.frame(st)
                obs = self.adapter.stack(f)
                if check_obs:
                    self.env.a_prev = self.adapter.a_prev.copy()
                    self.env.hist = self.env.hist if i else None
                    env_obs = self.env._obs()
                    err.append(np.abs(obs - env_obs).max())
                a = self.p.act(obs)
                self.adapter.a_prev = a.astype(np.float64)
                self.filter.command(a if self.res_on else a * 0)
                d = self.env.d
                tilt.append(math.degrees(ze._quat_tilt(*d.qpos[3:7])))
                vx.append(d.qvel[0])
            r = self.filter.tick().copy()
            res.append(np.abs(r).max())
            self.delay.append(r)
            self.physics(self.delay.pop(0))
            if self.fallen():
                return dict(alive=(i + 1) / PUB_HZ, tilt95=float(np.percentile(tilt, 95)),
                            vx=float(np.mean(vx)), fell=True, obs_err=max(err, default=0.0))
        return dict(alive=secs, tilt95=float(np.percentile(tilt, 95)),
                    vx=float((self.env.d.qpos[0] - x0) / secs), fell=False,
                    obs_err=max(err, default=0.0), res_max=float(np.degrees(max(res))))


if __name__ == "__main__":
    LIB = "variants/mukesh/libs/M1/gait_library.npz"
    p = Policy(os.path.join(REPO, "zeus_control_interface", "policy", "m1_mukesh_3M"))
    out = {}

    RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "closed_loop_results.json")
    only = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--only" else ""

    def row(name, **kw):
        if only not in name:
            return
        run_kw = {k: kw.pop(k) for k in ("phase", "at_rest", "check_obs", "secs") if k in kw}
        r = Sim(p, **kw).run(**run_kw)
        out[name] = r
        print(f"{name:42s} alive {r['alive']:5.1f}s  tilt95 {r['tilt95']:5.1f}  "
              f"vx {r['vx']:+.3f}  obs_err {r['obs_err']:.1e}", flush=True)

    row("nominal: deployed stack", check_obs=True)
    row("no policy: stored gait, zero residual", res_on=False)
    row("raw board values, no IMU correction", correct_imu=False)
    row("delay 8 ms", delay_pub=2)
    row("delay 20 ms", delay_pub=5)
    row("delay 40 ms", delay_pub=10)
    row("firmware as-is: no torque FF, no pre-deflect", ff=0.0)
    for f in (0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3):
        row(f"torque FF x{f:.1f}", ff=f)
    row("gains x0.33", gain=0.33)
    row("gains x0.5", gain=0.5)
    row("gains x0.7", gain=0.7)
    row("gains x1.5", gain=1.5)
    row("gains x2.0", gain=2.0)
    for z in (0.5, 1.0, 2.0, 3.0, 5.0):
        row(f"joint zero error +-{z:.1f} deg", zero_err_deg=z)
    for i in (1.0, 2.0, 3.0, -1.0, -2.0, -3.0):
        row(f"IMU mount pitch error {i:+.1f} deg", imu_pitch_deg=i)
    for ph in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95):
        row(f"start AT REST, phase {ph:.2f}", phase=ph, at_rest=True)
    if os.path.exists(RESULTS):              # --only keeps the other rows
        out = {**json.load(open(RESULTS)), **out}
    json.dump(out, open(RESULTS, "w"), indent=1)
