"""
zeus_env.py — residual-RL walking env for Zeus, built to the HARDWARE contract.

Observation = exactly what rl_policy_node gets from the board
(zeus_link.convert.policy_block, 44 raw SI values, same order), plus two
Pi-side quantities the node owns anyway (previous action, commanded speed),
stacked over HIST frames:

    pelvis_z 1 | quat wxyz 4 | gyro 3 | vel_hdg (lat, fwd, vert) 3 |
    joint_pos NJ | joint_vel NJ | spring_angle 4 | ref_angle NJ |
    contact (L, R) 2 | foot_z (R, L) 2 | phase 1 | prev_action NJ | vcmd 1

(NJ = 8 on the robot, 10 with ankle roll.)

Action = NJ residuals in [-1, 1] (HW joint order), scaled by ACT_SCALE (well
inside the board's 0.628 rad limit) and passed through the SAME first-order
low-pass the deployed node runs (ACT_FC Hz), then
    motor target = ref + residual - tau_ff/K   (SEA: spring pre-deflection,
                                                clipped to the mechanical travel)
    tau = KP (target - q_motor) + KD (dq_ref - dq_motor) + tau_ff
at 500 Hz, clipped to the drive peak.  The phase clock runs at the library's
own cadence (the board has no clock-rate channel).
"""
import os
import math
import ctypes
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

import zeus_model as zm


def _release_compile_cache():
    """MuJoCo keeps every compiled mesh in a per-process cache (385 MB for the mukesh
    model) that nothing here reuses.  Drop it and hand the pages back to the OS: a
    SubprocVecEnv worker goes from ~1.3 GB to ~0.67 GB.  The model itself is unchanged."""
    if hasattr(mujoco, "mj_getCache"):
        mujoco.mj_clearCache(mujoco.mj_getCache())
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


_release_compile_cache()                 # zeus_model's import already compiled the XML once


def _read_stl_points(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(raw[80:84].view(np.uint32)[0])
    if raw.size != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL")
    rec = raw[84:].reshape(n, 50)[:, 12:48].copy().view("<f4")      # 3 vertices of each triangle
    return rec.reshape(-1, 3).astype(np.float64)


def _hull_table(spec, xml):
    """name -> convex-hull vertices of every mesh asset, cached next to the XML."""
    from scipy.spatial import ConvexHull
    cache = os.path.join(os.path.dirname(os.path.abspath(xml)), "hull_cache.npz")
    old = dict(np.load(cache)) if os.path.exists(cache) else {}
    table, dirty = {}, False
    for mesh in spec.meshes:
        path = mesh.file if os.path.isabs(mesh.file) else os.path.join(spec.meshdir, mesh.file)
        st = os.stat(path)
        sig = np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)
        if mesh.name in old and np.array_equal(old.get("sig_" + mesh.name), sig):
            table[mesh.name] = old[mesh.name]
        else:
            pts = _read_stl_points(path)
            table[mesh.name] = pts[ConvexHull(pts).vertices]
            old[mesh.name], old["sig_" + mesh.name] = table[mesh.name], sig
            dirty = True
    if dirty:
        tmp = f"{cache}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **old)
        os.replace(tmp, cache)
    return table


def _load_model(xml):
    """Compile `xml` with every collision mesh replaced by its own convex hull.

    MuJoCo collides a mesh geom as the convex hull of its vertices, but it also builds and,
    on every mj_collision (10x per env step), clears a bounding-volume tree over ALL its
    triangles: 2.0 M nodes (2 MB) for the 494 k-vertex mukesh meshes, ~75 us per substep
    that no contact ever uses.  Handing MuJoCo the hull vertices instead gives the same
    collision shape (support function agrees to 1.5e-8 m, float32 storage noise), the same
    mass/inertia (the XML sets them explicitly, inertiafromgeom=false) and a 44x smaller
    tree.  ZEUS_FULL_MESH=1 restores the original meshes."""
    if os.environ.get("ZEUS_FULL_MESH", "0") == "1":
        return mujoco.MjModel.from_xml_path(xml)
    spec = mujoco.MjSpec.from_file(xml)
    hulls = _hull_table(spec, xml)
    for mesh in spec.meshes:
        mesh.file = ""
        mesh.uservert = hulls[mesh.name].ravel()
    return spec.compile()

# Geoms that only ever collide with the floor and hang more than 0.4 m above it for
# every pose the episode allows (it terminates at pelvis z < 0.40).  Their plane-vs-hull
# test runs 10x per step against up to 573 hull vertices and can never return a contact,
# so removing the pair cannot change a trajectory (proved by V7_gpu_scaling/dead_pairs.py).
DEAD_FLOOR_GEOMS = ("Base_link_collision", "Hip_pitch_infill_collision",
                    "Hip_pitch_infill_left_collision")

DT_PHYS = 0.002
DECIM = 10                               # 50 Hz policy
DT = DT_PHYS * DECIM
HIST = 3
NJ = zm.NJ                                # 8 actuated joints (10 with ankle roll)
NS = len(zm.SPRING_ORDER)                 # 4 SEA springs
OBS1 = 17 + 4 * NJ + NS                   # 44-value HW block + prev action + vcmd at NJ=8
KP = zm._by(150., 60., 30.)               # SEA / roll / ankle class
KD = zm._by(12., 4., 1.2)
ACT_SCALE = zm._by(0.25, 0.12, 0.15)      # rad
ACT_FC = float(os.environ.get("ACT_FC", 4.0))         # Hz, deployed filter
TILT_TERM = math.radians(20.0)
EP_STEPS = 1000

# reward weights (see README in train_zeus.py)
W = dict(vel=0.20, lat=0.08, lean=0.25, track=0.15, height=0.08, contact=0.10,
         yaw=0.06, rate=float(os.environ.get("RW_RATE", 0.6)),
         acc=float(os.environ.get("RW_ACC", 0.25)), mag=0.10, torque=0.04, collide=0.5,
         clear=0.04, spring=float(os.environ.get("W_SPRING", 0.3)),
         tilt=float(os.environ.get("RW_TILT", 0.0)))


class Library:
    def __init__(self, path):
        L = np.load(path, allow_pickle=True)
        self.v = L["velocities"]
        self.T = L["T"]
        self.keys1 = ["base_z", "base_roll", "base_pitch", "base_vx", "base_vy",
                      "base_vz", "base_wx", "base_wy", "base_wz", "sole_z_r",
                      "sole_z_l", "contact_r", "contact_l", "base_y"]
        self.L = {k: L[k] for k in self.keys1 + ["q_joints", "dq_joints", "tau_joints"]}
        self.n = L["phase"].shape[0]

    def slice(self, v):
        """All library signals at speed v (linear in v), shape (200, .)."""
        if len(self.v) == 1:                 # one-speed probe library
            s = {k: a[0] for k, a in self.L.items()}
            s["T"] = self.T[0]
            return s
        v = float(np.clip(v, self.v[0], self.v[-1]))
        i = int(np.clip(np.searchsorted(self.v, v) - 1, 0, len(self.v) - 2))
        w = (v - self.v[i]) / (self.v[i + 1] - self.v[i])
        s = {k: (1 - w) * a[i] + w * a[i + 1] for k, a in self.L.items()}
        s["T"] = (1 - w) * self.T[i] + w * self.T[i + 1]
        return s


def _quat_tilt(qw, qx, qy, qz):
    # angle between body z and world z
    zz = 1 - 2 * (qx * qx + qy * qy)
    return math.acos(max(-1.0, min(1.0, zz)))


class ZeusWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, library, vrange=(0.20, 0.50), dr=True, seed=None,
                 render_size=(480, 640)):
        self.m = _load_model(zm.XML)
        for g in DEAD_FLOOR_GEOMS:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gid >= 0:
                self.m.geom_contype[gid] = 0
                self.m.geom_conaffinity[gid] = 0
        _release_compile_cache()
        self.d = mujoco.MjData(self.m)
        self.lib = Library(library)
        self.vrange = vrange
        self.dr = dr
        self.rng = np.random.default_rng(seed)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qa = np.array([m.jnt_qposadr[jid(n)] for n in zm.JOINT_NAMES])
        self.va = np.array([m.jnt_dofadr[jid(n)] for n in zm.JOINT_NAMES])
        self.sqa = np.array([m.jnt_qposadr[jid(zm.SPRING_JOINTS[n])] for n in zm.SPRING_ORDER])
        self.sva = np.array([m.jnt_dofadr[jid(zm.SPRING_JOINTS[n])] for n in zm.SPRING_ORDER])
        # spring slot for each HW joint (or -1)
        self.sp_of = np.array([zm.SPRING_ORDER.index(n) if n in zm.SPRING_JOINTS else -1
                               for n in zm.JOINT_NAMES])
        self.sea = self.sp_of >= 0
        self.sea_i = np.flatnonzero(self.sea)
        assert (self.sp_of[self.sea_i] == np.arange(len(self.sqa))).all()
        self.site_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_sole")
        self.site_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.geom_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_sole_box")
        self.geom_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_sole_box")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        gb = m.geom_bodyid
        names = [m.body(b).name for b in range(m.nbody)]
        self.side_of = np.array([1 if names[gb[g]].startswith("left") else
                                 -1 if names[gb[g]].startswith("right") else 0
                                 for g in range(m.ngeom)])
        sa = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
        self.s_quat, self.s_gyro = sa("imu_quat"), sa("imu_gyro")
        self.nom_mass = m.body_mass.copy()
        self.nom_fric = m.geom_friction.copy()
        self.alpha = 1.0 - math.exp(-2 * math.pi * ACT_FC * DT_PHYS)
        self.kp, self.kd = KP.copy(), KD.copy()
        hi = np.full(OBS1 * HIST, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (NJ,), dtype=np.float32)
        self._renderer = None
        self.render_size = render_size
        self.vcmd = 0.35
        self.ff_scale = 1.0
        # pure-Python lookups and scratch buffers: numpy on 10-element arrays costs more in
        # call overhead than in arithmetic, so the hot paths avoid allocating and indexing
        self.side_py = self.side_of.tolist()
        self._qb, self._vb = np.empty(NJ), np.empty(NJ)
        self._tau = np.empty((DECIM, NJ))
        self._spr = np.empty((DECIM, NS))
        self._filt = np.empty((DECIM, NJ))
        self._obs_sd = np.concatenate([[0.01], np.full(3, 0.03), [0.04, 0.04], np.full(NJ, 0.003),
                                       np.full(NJ, 0.05), np.full(NS, 0.0005), np.full(4, 0.004)])
        self._o1 = np.empty(OBS1, np.float32)
        self._ob = np.empty(OBS1 * HIST, np.float32)

    # ── library at the current phase ──────────────────────────────────────
    def _ref(self, phase, key):
        n = self.lib.n
        x = (phase % 1.0) * n
        i = int(x)
        f = x - i
        a = self.S[key]
        return (1 - f) * a[i % n] + f * a[(i + 1) % n]

    # ── sensors exactly as the board reports them ─────────────────────────
    def _joint_out(self):
        qpos, qvel = self.d.qpos, self.d.qvel
        q = qpos[self.qa]
        dq = qvel[self.va]
        q[self.sea_i] += qpos[self.sqa]
        dq[self.sea_i] += qvel[self.sva]
        return q, dq

    def _contacts(self):
        n = self.d.ncon
        if n == 0:
            return False, False, False
        fl, gl, gr, side = self.floor, self.geom_l, self.geom_r, self.side_py
        cl = cr = lr_hit = False
        for g1, g2 in self.d.contact.geom[:n].tolist():       # <= ~9 pairs: plain Python wins
            if g1 == fl or g2 == fl:
                other = g2 if g1 == fl else g1
                if other == gl:
                    cl = True
                elif other == gr:
                    cr = True
            if side[g1] * side[g2] < 0:
                lr_hit = True
        return cl, cr, lr_hit

    def _ref_small(self, phase):
        """Reference joints, base height and both contact flags at `phase`, in one lookup."""
        n = self.lib.n
        x = (phase % 1.0) * n
        i = int(x)
        f = x - i
        T = self.ref_small
        return (1 - f) * T[i % n] + f * T[(i + 1) % n]

    def _obs1(self, q, dq, rs):
        d = self.d
        sd = d.sensordata
        quat = sd[self.s_quat:self.s_quat + 4].copy()
        gyro = sd[self.s_gyro:self.s_gyro + 3].copy()
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        cy, sy = math.cos(yaw), math.sin(yaw)
        vw = d.qvel[0:3]
        v_fwd = cy * vw[0] + sy * vw[1]
        v_lat = -sy * vw[0] + cy * vw[1]
        fz_r = d.site_xpos[self.site_r][2]
        fz_l = d.site_xpos[self.site_l][2]
        pz = d.qpos[2] - min(fz_r, fz_l)
        spring = d.qpos[self.sqa]
        if self.dr:                 # one batched draw == the same 34 normals in the same order
            z = self.rng.normal(0, self._obs_sd)
            pz += z[0]
            gyro += z[1:4]
            v_fwd += z[4]; v_lat += z[5]
            q += z[6:6 + NJ]; dq += z[6 + NJ:6 + 2 * NJ]
            spring = spring + z[6 + 2 * NJ:6 + 2 * NJ + NS]
            quat = quat + z[6 + 2 * NJ + NS:]; quat /= math.sqrt(quat.dot(quat))
        o = self._o1
        o[0] = pz
        o[1:5] = quat
        o[5:8] = gyro
        o[8] = v_lat; o[9] = v_fwd; o[10] = vw[2]
        k = 11
        o[k:k + NJ] = q; k += NJ
        o[k:k + NJ] = dq; k += NJ
        o[k:k + NS] = spring; k += NS
        o[k:k + NJ] = rs[:NJ]; k += NJ
        o[k] = self._cl; o[k + 1] = self._cr
        o[k + 2] = fz_r; o[k + 3] = fz_l
        o[k + 4] = self.phase % 1.0
        k += 5
        o[k:k + NJ] = self.a_prev; k += NJ
        o[k] = self.vcmd
        return o

    def _obs(self, q, dq, rs):
        o = self._obs1(q, dq, rs)
        ob = self._ob
        if self.hist is None:
            self.hist = True
            ob[0:OBS1] = o; ob[OBS1:2 * OBS1] = o; ob[2 * OBS1:] = o
        else:
            ob[:-OBS1] = ob[OBS1:]
            ob[-OBS1:] = o
        return ob.copy()

    # ── reset from the library state (reference state initialisation) ────
    def set_vcmd(self, v):
        self.vcmd_fixed = v

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        m, d = self.m, self.d
        v = getattr(self, "vcmd_fixed", None)
        self.vcmd = float(v if v is not None else r.uniform(*self.vrange))
        self.S = self.lib.slice(self.vcmd)
        self.ref_stack = np.concatenate([self.S["q_joints"], self.S["dq_joints"],
                                         self.S["tau_joints"]], axis=1)
        self.ref_small = np.concatenate([self.S["q_joints"], self.S["base_z"][:, None],
                                         self.S["contact_r"][:, None],
                                         self.S["contact_l"][:, None]], axis=1)
        self.period = float(self.S["T"])
        if self.dr:
            m.body_mass[:] = self.nom_mass * r.uniform(0.92, 1.08, m.nbody)
            m.geom_friction[:, 0] = self.nom_fric[:, 0] * r.uniform(0.7, 1.15)
            self.kp = KP * r.uniform(0.85, 1.15, NJ)
            self.kd = KD * r.uniform(0.8, 1.2, NJ)
            self.ff_scale = r.uniform(0.85, 1.10)
            self.lat_steps = int(r.integers(0, 2))
        else:
            self.lat_steps = 0
        mujoco.mj_resetData(m, d)
        self.phase = float(r.uniform(0, 1)) if (options or {}).get("phase") is None \
            else float(options["phase"])
        ph = self.phase
        qj = self._ref(ph, "q_joints")
        dqj = self._ref(ph, "dq_joints")
        roll, pitch = self._ref(ph, "base_roll"), self._ref(ph, "base_pitch")
        cr_, sr_ = math.cos(roll / 2), math.sin(roll / 2)
        cp_, sp_ = math.cos(pitch / 2), math.sin(pitch / 2)
        d.qpos[3:7] = [cr_ * cp_, sr_ * cp_, cr_ * sp_, -sr_ * sp_]
        d.qpos[self.qa] = qj
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        zmin = min(d.site_xpos[self.site_l][2], d.site_xpos[self.site_r][2])
        d.qpos[2] = 1.0 - zmin + 0.001
        d.qpos[1] = 0.0
        d.qvel[0:3] = [self._ref(ph, "base_vx"), self._ref(ph, "base_vy"),
                       self._ref(ph, "base_vz")]
        d.qvel[3:6] = [self._ref(ph, "base_wx"), self._ref(ph, "base_wy"),
                       self._ref(ph, "base_wz")]
        d.qvel[self.va] = dqj
        if self.dr:
            d.qvel[0:6] += r.normal(0, [0.05, 0.05, 0.02, 0.1, 0.1, 0.05])
        mujoco.mj_forward(m, d)
        self.filt = np.zeros(NJ)
        self.a_prev = np.zeros(NJ)
        self.a_prev2 = np.zeros(NJ)
        self.hist = None
        self.t = 0
        self.yaw0 = 0.0
        self._cl, self._cr, _ = self._contacts()
        self.act_queue = [np.zeros(NJ)] * (self.lat_steps + 1)
        self.ep = dict(tau2=np.zeros(NJ), n=0)
        q, dq = self._joint_out()
        return self._obs(q, dq, self._ref_small(self.phase)), {}

    # ── step ──────────────────────────────────────────────────────────────
    def step(self, action):
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.act_queue.append(a)
        a_eff = self.act_queue.pop(0)
        res_cmd = a_eff * ACT_SCALE
        dph = DT_PHYS / self.period
        rstk, nlib = self.ref_stack, self.lib.n
        mx, mn = np.maximum, np.minimum
        # Everything that does not depend on the physics state is computed for all
        # DECIM substeps at once.  Same arithmetic, same order: golden.py checks it.
        ph = np.empty(DECIM)
        p = self.phase
        for k in range(DECIM):
            ph[k] = p
            p += dph
        self.phase = p
        x = (ph % 1.0) * nlib
        i = x.astype(np.int64)
        f = (x - i)[:, None]
        rs = (1 - f) * rstk[i % nlib] + f * rstk[(i + 1) % nlib]
        filt, FILT = self.filt, self._filt
        for k in range(DECIM):
            filt += self.alpha * (res_cmd - filt)
            FILT[k] = filt
        TFF = rs[:, 2 * NJ:3 * NJ] * self.ff_scale
        TGT = rs[:, 0:NJ] + FILT
        DQR = rs[:, NJ:2 * NJ]
        # spring pre-deflection, never past the spring's mechanical travel
        TGT[:, self.sea_i] -= mn(mx(TFF[:, self.sea_i] / zm.K_SPRING,
                                    -zm.SPRING_TRAVEL), zm.SPRING_TRAVEL)
        # the only part that must run at 500 Hz: PD on the live state, then physics.
        # Same operations in the same order as `kp*(tgt-q) + kd*(dqref-dq) + tff`, but
        # written into preallocated buffers (no allocation, no fancy-index copies).
        qpos, qvel, ctrl = d.qpos, d.qvel, d.ctrl
        qa, va, sqa, kp, kd = self.qa, self.va, self.sqa, self.kp, self.kd
        tpk, ntpk, step = zm.TAU_PEAK, -zm.TAU_PEAK, mujoco.mj_step
        take, sub, mul, add = np.take, np.subtract, np.multiply, np.add
        qb, vb, TAU, SPR = self._qb, self._vb, self._tau, self._spr
        for k in range(DECIM):
            tk = TAU[k]
            take(qpos, qa, out=qb, mode="clip")
            sub(TGT[k], qb, out=qb); mul(kp, qb, out=qb)
            take(qvel, va, out=vb, mode="clip")
            sub(DQR[k], vb, out=vb); mul(kd, vb, out=vb)
            add(qb, vb, out=tk); add(tk, TFF[k], out=tk)
            mx(tk, ntpk, out=tk); mn(tk, tpk, out=tk)
            ctrl[:] = tk
            step(m, d)
            take(qpos, sqa, out=SPR[k], mode="clip")
        self.t += 1
        self.last_tau2 = (TAU * TAU).sum(axis=0) / DECIM
        tau_max = np.abs(TAU).max(axis=0)
        spr_max = np.abs(SPR).max(axis=0)
        self.last_taumax = tau_max
        cl, cr, lr_hit = self._contacts()
        self._cl, self._cr = cl, cr

        # ── reward ────────────────────────────────────────────────────────
        qw, qx, qy, qz = d.qpos[3:7]
        tilt = _quat_tilt(qw, qx, qy, qz)
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        vx, vy = d.qvel[0], d.qvel[1]
        q, dq = self._joint_out()
        ref = self._ref_small(self.phase)
        qref, zref = ref[:NJ], ref[NJ]
        c_r = ref[NJ + 1] > 0.5
        c_l = ref[NJ + 2] > 0.5
        tilt_deg = math.degrees(tilt)
        r_vel = math.exp(-((vx - self.vcmd) / 0.12) ** 2)
        r_lat = math.exp(-(vy / 0.15) ** 2)
        r_lean = math.exp(-(tilt_deg / 2.0) ** 2)
        # r_lean is flat past about 5 deg (1e-7 at 8 deg): this keeps a gradient there
        p_tilt = (tilt_deg / 5.0) ** 2
        r_track = math.exp(-float(((q - qref) ** 2).sum() / NJ) / 0.08 ** 2)
        r_h = math.exp(-((d.qpos[2] - zref) / 0.03) ** 2)
        r_c = 0.5 * (float(cl == c_l) + float(cr == c_r))
        r_yaw = math.exp(-(yaw / 0.15) ** 2)
        da = a - self.a_prev
        dda = a - 2 * self.a_prev + self.a_prev2
        # x.sum()/n == np.mean(x) bit-for-bit, minus numpy's Python-level wrapper
        p_rate = float((da ** 2).sum() / NJ)
        p_acc = float((dda ** 2).sum() / NJ)
        p_mag = float((a ** 2).sum() / NJ)
        p_tau = float((self.last_tau2 / zm.TAU_NOM ** 2).sum() / NJ)
        # SEA recoil: free up to SPRING_SOFT (1 deg), then quadratic, reaching 1
        # per spring at the mechanical end.  Peak over the physics substeps: the
        # touchdown recoil rings at ~9 Hz and a 50 Hz sample would miss its top.
        x = np.maximum(spr_max - zm.SPRING_SOFT, 0.0) / (zm.SPRING_TRAVEL - zm.SPRING_SOFT)
        p_spring = float((x ** 2).sum() / NS)
        # swing toe clearance: swing sole must be off the floor mid-swing
        fz = (d.site_xpos[self.site_l][2], d.site_xpos[self.site_r][2])
        p_clear = 0.0
        if not c_l:
            p_clear += float(cl)
        if not c_r:
            p_clear += float(cr)
        rew = (W["vel"] * r_vel + W["lat"] * r_lat + W["lean"] * r_lean
               + W["track"] * r_track + W["height"] * r_h + W["contact"] * r_c
               + W["yaw"] * r_yaw
               - W["rate"] * p_rate - W["acc"] * p_acc - W["mag"] * p_mag
               - W["torque"] * p_tau - W["collide"] * float(lr_hit)
               - W["clear"] * p_clear - W["spring"] * p_spring - W["tilt"] * p_tilt)
        self.a_prev2 = self.a_prev
        self.a_prev = a

        bad = not np.isfinite(d.qpos).all() or np.abs(d.qvel).max() > 100
        term = bool(bad or tilt > TILT_TERM or d.qpos[2] < 0.40)
        trunc = self.t >= EP_STEPS
        if term:
            rew -= 2.0
        info = dict(vx=vx, vy=vy, tilt_deg=tilt_deg, lr_hit=lr_hit, cl=cl, cr=cr,
                    p_rate=p_rate, p_spring=p_spring,
                    spring_peak_deg=float(np.degrees(spr_max.max())))
        obs = self._obs(q, dq, ref)
        if bad:
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, float(rew), term, trunc, info

    def render(self, cam_azimuth=90.0, track=True):
        if self._renderer is None:
            # draw the original meshes, not the collision hulls the physics runs on
            self._vm = mujoco.MjModel.from_xml_path(zm.XML)
            self._vd = mujoco.MjData(self._vm)
            self._renderer = mujoco.Renderer(self._vm, *self.render_size)
            self._cam = mujoco.MjvCamera()
            self._cam.distance = 1.9
            self._cam.elevation = -12
        self._vd.qpos[:] = self.d.qpos
        mujoco.mj_kinematics(self._vm, self._vd)
        self._cam.azimuth = cam_azimuth
        self._cam.lookat[:] = [self.d.qpos[0], self.d.qpos[1], 0.45]
        self._renderer.update_scene(self._vd, self._cam)
        return self._renderer.render()
