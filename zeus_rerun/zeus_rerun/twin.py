"""
Two robots in 3D, side by side: what was COMMANDED and what HAPPENED.

    twin/commanded   the CAD robot posed at the drive targets (act_target: the
                     stored gait + the policy's residual, after the board's
                     envelope and slew limit). Springs rigid, torso upright,
                     lowest foot switch on the ground. This is the motion the
                     policy asked for - what a perfect, stiff robot would do.
    twin/real        the CAD robot posed at the measured joint angles AND the
                     measured spring deflections, at the estimator's pose.

The difference between the two views is the sim-to-real gap made visible:
tracking error, spring sag, torso tilt the gait never asked for.

WHAT IS AND IS NOT MEASURED in twin/real:
  - joints and springs: measured, every frame.
  - torso orientation and height: the estimator (quat, pelvis_z) while
    fused_valid is OK (2); otherwise the BNO085's own tilt with the last good heading,
    and the lowest foot switch put on the ground. That fallback is a guess -
    "estimator/fused_valid" says which one you are looking at.
  - forward/sideways position: the estimator's fused_pos, held while invalid.

twin/commanded has no physics. It is not a simulation of the policy, it is
the command drawn as a robot. Where nothing is commanded (act_target NaN:
idle, before arming) it copies the real joints so it does not invent a pose.

Drawn at 50 Hz (every 20th board tick), the policy's own rate.

The foot spheres are the switches: green pressed, grey open.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np
import rerun as rr

from zeus_link import nexus_proto as P

RATE_DIV = 20                       # 1 kHz / 20 = 50 Hz
REAL_COLOUR = [150, 170, 200, 255]
CMD_COLOUR = [240, 190, 60, 255]
PRESSED = [60, 200, 90]
OPEN = [140, 140, 140]
CONTACTS = ("left_contact", "right_contact")     # P.CONTACT order: left, right


def _find_description() -> Optional[str]:
    """zeus_description: installed share dir, else next to this source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory("zeus_description")
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for up in ("../..", "../../.."):
        cand = os.path.normpath(os.path.join(here, up, "zeus_description"))
        if os.path.isdir(os.path.join(cand, "urdf")):
            return cand
    return None


def _rpy(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def _axis_rot(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N,) angles about one unit axis -> (N,3,3), Rodrigues."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(N,4) w,x,y,z -> (N,3,3). Normalises; a zero or NaN row becomes identity."""
    q = np.array(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=1)
    bad = ~np.isfinite(n) | (n < 1e-6)
    q[bad] = [1, 0, 0, 0]
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _yaw(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def _Rz(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros((len(yaw), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    return R


class Urdf:
    def __init__(self, desc_dir: str):
        root = ET.parse(os.path.join(desc_dir, "urdf", "zeus.urdf")).getroot()
        self.visuals: Dict[str, List] = {}
        for link in root.findall("link"):
            vis = []
            for v in link.findall("visual"):
                m = v.find("geometry/mesh")
                if m is None:
                    continue
                fn = m.get("filename").split("package://zeus_description/")[-1]
                scale = float(m.get("scale", "1 1 1").split()[0])
                vis.append((os.path.join(desc_dir, fn), _origin(v.find("origin")), scale))
            self.visuals[link.get("name")] = vis
        self.joints = []                      # (name, type, parent, child, T0, axis)
        children = set()
        for j in root.findall("joint"):
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            axis /= np.linalg.norm(axis)
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"),
                                j.find("child").get("link"), _origin(j.find("origin")), axis))
            children.add(j.find("child").get("link"))
        self.root = next(l for l in self.visuals if l not in children)
        # parents before children
        order, known = [], {self.root}
        pending = list(self.joints)
        while pending:
            nxt = []
            for j in pending:
                (order if j[2] in known else nxt).append(j)
                if j[2] in known:
                    known.add(j[3])
            if len(nxt) == len(pending):
                raise ValueError("URDF is not a tree")
            pending = nxt
        self.joints = order

    def fk(self, q: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
        """Joint angles by name, (n,) each -> every link's (n,4,4) in the root frame."""
        T = {self.root: np.broadcast_to(np.eye(4), (n, 4, 4)).copy()}
        for name, typ, parent, child, T0, axis in self.joints:
            Tc = T[parent] @ T0
            if typ in ("revolute", "continuous") and name in q:
                M = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                M[:, :3, :3] = _axis_rot(axis, q[name])
                Tc = Tc @ M
            T[child] = Tc
        return T


def _inv(T: np.ndarray) -> np.ndarray:
    Ti = np.zeros_like(T)
    Rt = np.transpose(T[:, :3, :3], (0, 2, 1))
    Ti[:, :3, :3] = Rt
    Ti[:, :3, 3] = -(Rt @ T[:, :3, 3, None])[:, :, 0]
    Ti[:, 3, 3] = 1
    return Ti


class Twin:
    def __init__(self, desc_dir: Optional[str] = None):
        desc_dir = desc_dir or _find_description()
        if desc_dir is None:
            raise FileNotFoundError("zeus_description not found: no URDF for the 3D twin")
        self.urdf = Urdf(desc_dir)
        self.last_yaw = 0.0
        self.last_xy = np.zeros(2)

    def log_static(self) -> None:
        for robot, colour in (("commanded", CMD_COLOUR), ("real", REAL_COLOUR)):
            for link, vis in self.urdf.visuals.items():
                for k, (path, T, scale) in enumerate(vis):
                    e = f"twin/{robot}/{link}/mesh{k}"
                    rr.log(e, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3] * scale),
                           static=True)
                    rr.log(e, rr.Asset3D(path=path, albedo_factor=colour), static=True)

    def _q(self, joints: np.ndarray, springs: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        q = {n: np.nan_to_num(joints[:, i]) for i, n in enumerate(P.JOINT_NAMES)}
        if springs is not None:
            for i, n in enumerate(P.SPRING_NAMES):
                q[f"{n}_spring"] = np.nan_to_num(springs[:, i])
        return q

    def _place(self, T: Dict[str, np.ndarray], R_imu: np.ndarray, xy: np.ndarray,
               z_imu: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Put imu_link at (xy, z, R) in the world; z None = lowest switch on z=0."""
        n = len(R_imu)
        W = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        W[:, :3, :3] = R_imu
        W[:, :2, 3] = xy
        Wroot = W @ _inv(T["imu_link"])
        out = {k: Wroot @ v for k, v in T.items()}
        low = np.minimum(out[CONTACTS[0]][:, 2, 3], out[CONTACTS[1]][:, 2, 3])
        if z_imu is None:
            dz = -low
        else:
            dz = np.where(np.isfinite(z_imu), z_imu - W[:, 2, 3], -low)
        for v in out.values():
            v[:, 2, 3] += dz
        return out

    def flush(self, pkts, pi_s, timeline: str) -> None:
        sel = [i for i, p in enumerate(pkts) if p.seq % RATE_DIV == 0]
        if not sel:
            return
        ps = [pkts[i] for i in sel]
        n = len(ps)

        def col(name):
            return np.array([getattr(p, name) for p in ps], dtype=np.float64)

        idx = [rr.TimeColumn(timeline, duration=col("seq") / 1000.0),
               rr.TimeColumn("pi", duration=np.asarray(pi_s, dtype=np.float64)[sel])]

        pos, spr, tgt = col("joint_pos"), col("spring_angle"), col("act_target")
        valid = col("fused_valid") == P.FUSION_OK      # not CONVERGING
        R_est = _quat_wxyz_to_R(col("quat"))
        R_imu = _quat_wxyz_to_R(col("imu_quat"))
        fp = col("fused_pos")

        # Heading: the estimator's while valid, else the last good one, on top
        # of the BNO085's own tilt (its yaw is arbitrary).
        yaw = np.empty(n)
        xy = np.empty((n, 2))
        est_yaw = _yaw(R_est)
        for i in range(n):
            if valid[i] and np.all(np.isfinite(fp[i, :2])):
                self.last_yaw, self.last_xy = est_yaw[i], fp[i, :2]
            yaw[i], xy[i] = self.last_yaw, self.last_xy
        R_fallback = _Rz(yaw - _yaw(R_imu)) @ R_imu
        R_real = np.where(valid[:, None, None], R_est, R_fallback)
        z_real = np.where(valid, col("pelvis_z"), np.nan)

        real = self._place(self.urdf.fk(self._q(pos, spr), n), R_real, xy, z_real)

        cmd = np.where(np.isfinite(tgt), tgt, pos)
        commanded = self._place(self.urdf.fk(self._q(cmd, None), n), _Rz(yaw), xy, None)

        for robot, T in (("commanded", commanded), ("real", real)):
            for link in self.urdf.visuals:
                if not self.urdf.visuals[link]:
                    continue
                M = T[link]
                rr.send_columns(f"twin/{robot}/{link}", indexes=idx,
                                columns=rr.Transform3D.columns(
                                    translation=M[:, :3, 3],
                                    mat3x3=M[:, :3, :3].reshape(n, 9)))
            # switch spheres: both robots show the REAL switch state
            contact = col("contact")
            pts = np.stack([T[c][:, :3, 3] for c in CONTACTS], 1)          # n,2,3
            colours = np.where(contact[:, :, None] > 0.5, PRESSED, OPEN)    # n,2,3
            rr.send_columns(f"twin/{robot}/switches", indexes=idx,
                            columns=rr.Points3D.columns(
                                positions=pts.reshape(-1, 3),
                                colors=colours.reshape(-1, 3).astype(np.uint8),
                                radii=np.full(2 * n, 0.025)).partition(lengths=[2] * n))
