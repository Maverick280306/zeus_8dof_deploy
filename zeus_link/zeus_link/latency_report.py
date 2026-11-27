"""
latency_report - from the board measuring a state to the board applying the
command the policy computed from it, measured on a real recording.

Training allowed 0 or 1 policy steps of extra action delay (zeus_env
`lat_steps`, 0..20 ms at 50 Hz). A robot whose real delay is well past that is
being driven by a policy that never saw such a lag, however right everything
else is - so this is measured, not assumed.

Every figure is in BOARD TICKS (1 ms), the one clock both ends share:

    S   the board tick of the state the policy used
    T   the first board tick whose drive target carries the resulting command

A recording alone (the .flt the link node writes) gives:

    usb in     state packet's arrival on the Pi, above the fastest packet of the
               run (the cable's own fixed delay is not observable from one
               clock; what varies is)
    applied    command leaving the Pi -> the board's target carrying it
    total >=   T minus the newest state the Pi had when the command left - a
               lower bound, because the policy computed from an older one

The policy node's timing log (policy_timing_*.csv, same directory, written
while walking) names the exact state each action used, and fills in the rest:

    to policy  link node received the state -> policy node received it (ROS)
    wait       -> the policy's 250 Hz tick picked it up
    inference  -> the network's output
    to link    -> the command reached the link node and left for the board
    total      T - S, exactly

Matching uses the hip roll and ankle joints only: they have no spring, so the
target the board reports minus the gait reference IS the residual - no
pre-deflection to undo. A command counts only where its residual differs from
the one before, so a run of identical commands cannot match the wrong one.

    ros2 run zeus_link latency_report ~/zeus_logs/zeus_20261127_020657.flt
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
from typing import Dict, Optional

import numpy as np

from . import nexus_proto as P
from .flight_log import RX, TX, read_records

CMD = struct.Struct(P.COMMAND_FORMAT)
NON_SEA = (P.JOINT_NAMES.index("left_hip_roll"), P.JOINT_NAMES.index("left_ankle_pitch"),
           P.JOINT_NAMES.index("right_hip_roll"), P.JOINT_NAMES.index("right_ankle_pitch"))
TICK_NS = 1_000_000
TWO_PI = 2.0 * np.pi
MATCH_TOL_RAD = 2e-4        # float32 turns -> rad, and back through the board
NEW_VALUE_RAD = 1e-3        # a command counts only if it moved this much
SEARCH_TICKS = 60           # a command not applied within 60 ms is not found
TRAINED_DELAY_MS = 20.0     # zeus_env: lat_steps in {0, 1} at 50 Hz

TIMING_FIELDS = ("state_seq", "state_rx_ns", "act_start_ns", "act_end_ns", "publish_ns")


def load(path: str) -> Dict[str, np.ndarray]:
    """States and commands out of a recording, with the Pi's receive/send time."""
    _, records = read_records(path)
    buf = bytearray()
    st = {k: [] for k in ("seq", "t", "target", "ref", "armed")}
    cm = {k: [] for k in ("t", "residual", "flags", "last_seq")}
    last = -1
    for d, t, data in records:
        if d == RX:
            buf += data
            while True:
                p, buf = P.NexusState.find_and_parse(buf)
                if p is None:
                    break
                st["seq"].append(p.seq)
                st["t"].append(t)
                st["target"].append(p.act_target)
                st["ref"].append(p.ref_angle)
                st["armed"].append(p.safety_state == P.SAFETY_ARMED)
                last = p.seq
        elif d == TX and len(data) == CMD.size:
            f = CMD.unpack(data)
            if f[0] != P.SYNC or f[1] != P.MSG_COMMAND:
                continue
            n = P.NUM_JOINTS
            cm["t"].append(t)
            cm["residual"].append(f[4:4 + n])
            cm["flags"].append(f[4 + n])
            cm["last_seq"].append(last)
    out = {f"state_{k}": np.asarray(v, dtype=np.float64) for k, v in st.items()}
    out.update({f"cmd_{k}": np.asarray(v, dtype=np.float64) for k, v in cm.items()})
    return out


def clock_offset(seq: np.ndarray, t_ns: np.ndarray) -> float:
    """Pi time of board tick 0, from the fastest-arriving packet. Every other
    packet arrived this much or later; the difference is the delay that varies."""
    return float(np.min(t_ns - seq * TICK_NS))


def applied(rec: Dict[str, np.ndarray], offset_ns: float) -> Dict[str, np.ndarray]:
    """For each command that carried a new value: the board tick it left the Pi,
    the first board tick whose target carries it, and the newest state the Pi
    had when it left."""
    seq, armed = rec["state_seq"], rec["state_armed"] > 0
    res_now = rec["state_target"][:, NON_SEA] - rec["state_ref"][:, NON_SEA]
    cmd_rad = rec["cmd_residual"][:, NON_SEA] * TWO_PI
    sent, found, last = [], [], []
    prev = None
    for k in range(len(cmd_rad)):
        v = cmd_rad[k]
        new = prev is None or np.max(np.abs(v - prev)) > NEW_VALUE_RAD
        prev = v
        if not new:
            continue
        b = (rec["cmd_t"][k] - offset_ns) / TICK_NS
        i0 = int(np.searchsorted(seq, np.floor(b)))
        for i in range(i0, min(i0 + SEARCH_TICKS, len(seq))):
            if armed[i] and np.all(np.abs(res_now[i] - v) < MATCH_TOL_RAD):
                sent.append(b)
                found.append(seq[i])
                last.append(rec["cmd_last_seq"][k])
                break
    return {"sent_tick": np.asarray(sent), "applied_tick": np.asarray(found),
            "last_seq": np.asarray(last)}


def load_timing(path: str) -> Dict[str, np.ndarray]:
    rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    return {name: rows[:, i] for i, name in enumerate(TIMING_FIELDS)}


def find_timing(flt: str, t0_ns: float, t1_ns: float) -> Optional[str]:
    """The policy timing log written during this recording, if there is one."""
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(flt)),
                                              "policy_timing_*.csv"))):
        try:
            tm = load_timing(path)
        except (OSError, ValueError):
            continue
        if len(tm["publish_ns"]) and t0_ns <= tm["publish_ns"][0] <= t1_ns:
            return path
    return None


def stages(rec, offset_ns, app, tm) -> Dict[str, np.ndarray]:
    """Per policy action: every stage from the board's state to the board's target."""
    out = {k: [] for k in ("usb_in", "to_policy", "wait", "inference", "to_link",
                           "applied", "total")}
    seq, t_rx = rec["state_seq"], rec["state_t"]
    for r in range(len(tm["state_seq"])):
        S = tm["state_seq"][r]
        i = int(np.searchsorted(seq, S))
        if i >= len(seq) or seq[i] != S:
            continue
        k = int(np.searchsorted(rec["cmd_t"], tm["publish_ns"][r]))
        if k >= len(rec["cmd_t"]):
            continue
        b = (rec["cmd_t"][k] - offset_ns) / TICK_NS
        j = np.where(np.abs(app["sent_tick"] - b) < 1e-6)[0]
        if not len(j):
            continue            # that command was not a new value, or never found
        out["usb_in"].append((t_rx[i] - S * TICK_NS - offset_ns) / 1e6)
        out["to_policy"].append((tm["state_rx_ns"][r] - t_rx[i]) / 1e6)
        out["wait"].append((tm["act_start_ns"][r] - tm["state_rx_ns"][r]) / 1e6)
        out["inference"].append((tm["act_end_ns"][r] - tm["act_start_ns"][r]) / 1e6)
        out["to_link"].append((rec["cmd_t"][k] - tm["act_end_ns"][r]) / 1e6)
        out["applied"].append(app["applied_tick"][j[0]] - b)
        out["total"].append(app["applied_tick"][j[0]] - S)
    return {k: np.asarray(v) for k, v in out.items()}


def _row(name: str, v: np.ndarray, note: str = "") -> str:
    if not len(v):
        return f"  {name:<12} -"
    return (f"  {name:<12} median {np.median(v):6.1f}  p90 {np.percentile(v, 90):6.1f}  "
            f"max {np.max(v):6.1f} ms   ({len(v)}){'  ' + note if note else ''}")


def report(flt: str, timing: Optional[str] = None) -> int:
    rec = load(flt)
    if not len(rec["state_seq"]) or not len(rec["cmd_t"]):
        print("no states or no commands in this recording")
        return 2
    off = clock_offset(rec["state_seq"], rec["state_t"])
    app = applied(rec, off)
    usb = (rec["state_t"] - rec["state_seq"] * TICK_NS - off) / 1e6
    print(f"{os.path.basename(flt)}: {len(rec['state_seq'])} states, {len(rec['cmd_t'])} "
          f"commands, {len(app['applied_tick'])} new command values found on the board")
    print(_row("usb in", usb, "above the fastest packet"))
    if not len(app["applied_tick"]):
        print("  no command was found applied - was the robot armed and walking?")
        return 2
    print(_row("applied", app["applied_tick"] - app["sent_tick"], "Pi send -> board target"))
    lower = app["applied_tick"] - app["last_seq"]
    print(_row("total >=", lower, "newest state the Pi had -> board target"))

    if timing is None:
        timing = find_timing(flt, rec["state_t"][0], rec["state_t"][-1])
    total = lower
    if timing:
        sg = stages(rec, off, app, load_timing(timing))
        print(f"with {os.path.basename(timing)}:")
        for name in ("usb_in", "to_policy", "wait", "inference", "to_link", "applied"):
            print(_row(name.replace("_", " "), sg[name]))
        print(_row("TOTAL", sg["total"], "state the policy used -> board target"))
        if len(sg["total"]):
            total = sg["total"]
    else:
        print("no policy timing log for this run: the total above is a lower bound")

    p90 = float(np.percentile(total, 90))
    verdict = "inside" if p90 <= TRAINED_DELAY_MS else "OUTSIDE"
    print(f"p90 {p90:.1f} ms: {verdict} the {TRAINED_DELAY_MS:.0f} ms of extra delay the "
          f"policy was trained with")
    return 0 if p90 <= TRAINED_DELAY_MS else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", help="a .flt the link node wrote")
    ap.add_argument("--timing", help="policy_timing_*.csv (default: found next to it)")
    a = ap.parse_args(argv)
    return report(a.recording, a.timing)


if __name__ == "__main__":
    sys.exit(main())
