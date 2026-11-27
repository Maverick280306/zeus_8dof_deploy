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

    usb in     a state packet's arrival on the Pi, above the fastest packet
               nearby (the cable's own fixed delay is not observable from one
               clock; what varies is)
    applied    the tick a command arrived on -> the board's target carrying it
    total >=   T minus the newest state the Pi had when the command left - a
               lower bound, because the policy computed from an older one

The policy node's timing log (policy_timing_*.csv, same directory, written
while walking) names the exact state each action used, and fills in the rest:

    to policy  link node received the state -> policy node received it (ROS)
    wait       -> the policy's 250 Hz tick picked it up
    inference  -> the network's output
    to link    -> the command reached the link node and left for the board
    total      T - S, exactly

What the real board does that a naive match misses (2026-11-27, the first
attempt matched nothing on the robot):

  - The two clocks drift - 145 ppm on Zeus, 32 ms over a four-minute run - so
    board time is pinned to the fastest packet of each second, not of the run.
  - Drive targets go out every ODRV_CMD_DIVIDER (2) ticks, while ref_angle is
    the current tick's: on a fast joint target - ref wobbles 0.1 deg tick to
    tick. The tick the target was computed on is recovered from the data
    (whichever reading is smoothest).
  - The ankles carry a small constant offset on top of the residual. A
    constant per joint is allowed for.

So rather than look for each command exactly, the delay is the lag at which
the applied residual best follows the commanded one, over windows of the walk
- one figure per window, which gives the spread as well. Hip roll and ankle
only: they have no spring, so target - ref is the residual itself.

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
OFFSET_WINDOW_TICKS = 1000  # drift is under 0.2 ms across one of these
LAG_WINDOW_TICKS = 200      # one delay estimate per 200 ms of walking
MIN_MOTION_RAD = 5e-3       # a window whose commands barely move cannot be timed
MAX_LAG_TICKS = 40
FIT_RAD = 1.5e-3            # a window fitted worse than this is not believed
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


def local_offset(seq: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Pi time of board tick 0, per state: the fastest packet of its second
    and the seconds either side. One offset for a whole run reads clock drift
    as delay."""
    d = t_ns - seq * TICK_NS
    w = (seq // OFFSET_WINDOW_TICKS).astype(np.int64)
    uw, inv = np.unique(w, return_inverse=True)
    mins = np.full(len(uw), np.inf)
    np.minimum.at(mins, inv, d)
    m = mins.copy()
    m[1:] = np.minimum(m[1:], mins[:-1])
    m[:-1] = np.minimum(m[:-1], mins[1:])
    return m[inv]


def applied_residual(rec: Dict[str, np.ndarray]) -> np.ndarray:
    """target - ref on the springless joints, each against the reference of
    the tick its target was computed on: every tick, or every second tick on
    either parity - whichever reads smoothest across the armed ticks."""
    tgt = rec["state_target"][:, NON_SEA]
    ref = rec["state_ref"][:, NON_SEA]
    prev = np.vstack([ref[:1], ref[:-1]])
    seq = rec["state_seq"].astype(np.int64)
    use = (rec["state_armed"] > 0) & np.all(np.isfinite(tgt), axis=1)
    best, score = tgt - ref, np.inf
    for cand in (tgt - ref,
                 np.where((seq % 2 == 0)[:, None], tgt - ref, tgt - prev),
                 np.where((seq % 2 == 1)[:, None], tgt - ref, tgt - prev)):
        c = cand[use]
        s = float(np.sum(np.abs(np.diff(c, axis=0)))) if len(c) > 1 else 0.0
        if s < score:
            best, score = cand, s
    return best


def command_ticks(rec, offsets) -> np.ndarray:
    """The board tick each command arrived on, by the local clock."""
    i = np.clip(np.searchsorted(rec["state_t"], rec["cmd_t"]), 0, len(rec["state_t"]) - 1)
    return np.floor((rec["cmd_t"] - offsets[i]) / TICK_NS)


def lag_windows(rec, offsets) -> Dict[str, np.ndarray]:
    """For each stretch of walking: the delay (ticks) from a command's arrival
    to the board's target carrying it, and how well that delay fits."""
    seq = rec["state_seq"].astype(np.int64)
    app = applied_residual(rec)
    cmd = rec["cmd_residual"][:, NON_SEA] * TWO_PI
    arr = command_ticks(rec, offsets)
    ok = (rec["state_armed"] > 0) & np.all(np.isfinite(app), axis=1)
    centre, lag, fit = [], [], []
    idx = np.where(ok)[0]
    for w0 in range(0, len(idx), LAG_WINDOW_TICKS):
        rows = idx[w0:w0 + LAG_WINDOW_TICKS]
        if len(rows) < LAG_WINDOW_TICKS // 2:
            continue
        best = (np.inf, None)
        for L in range(MAX_LAG_TICKS + 1):
            k = np.searchsorted(arr, seq[rows] - L, side="right") - 1
            good = k >= 0
            if good.sum() < len(rows) // 2:
                continue
            c = cmd[k[good]]
            if np.max(np.ptp(c, axis=0)) < MIN_MOTION_RAD:
                best = (np.inf, None)
                break                       # nothing moving: no delay to see
            e = app[rows][good] - c
            e = e - np.median(e, axis=0)    # a constant offset per joint is allowed
            s = float(np.median(np.abs(e)))
            if s < best[0]:
                best = (s, L)
        if best[1] is not None and best[0] < FIT_RAD:
            centre.append(seq[rows[len(rows) // 2]])
            lag.append(best[1])
            fit.append(best[0])
    return {"tick": np.asarray(centre), "lag": np.asarray(lag), "fit": np.asarray(fit)}


def lag_at(win, tick: float) -> Optional[float]:
    """The delay of the window nearest this tick, if one is within a window."""
    if not len(win["tick"]):
        return None
    j = int(np.argmin(np.abs(win["tick"] - tick)))
    return float(win["lag"][j]) if abs(win["tick"][j] - tick) <= LAG_WINDOW_TICKS else None


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


def stages(rec, offsets, win, tm) -> Dict[str, np.ndarray]:
    """Per policy action: every stage from the board's state to the board's target."""
    out = {k: [] for k in ("usb_in", "to_policy", "wait", "inference", "to_link",
                           "applied", "total")}
    seq, t_rx = rec["state_seq"], rec["state_t"]
    arr = command_ticks(rec, offsets)
    for r in range(len(tm["state_seq"])):
        S = tm["state_seq"][r]
        i = int(np.searchsorted(seq, S))
        if i >= len(seq) or seq[i] != S:
            continue
        k = int(np.searchsorted(rec["cmd_t"], tm["publish_ns"][r]))
        if k >= len(rec["cmd_t"]):
            continue
        L = lag_at(win, arr[k])
        if L is None:
            continue
        out["usb_in"].append((t_rx[i] - S * TICK_NS - offsets[i]) / 1e6)
        out["to_policy"].append((tm["state_rx_ns"][r] - t_rx[i]) / 1e6)
        out["wait"].append((tm["act_start_ns"][r] - tm["state_rx_ns"][r]) / 1e6)
        out["inference"].append((tm["act_end_ns"][r] - tm["act_start_ns"][r]) / 1e6)
        out["to_link"].append((rec["cmd_t"][k] - tm["act_end_ns"][r]) / 1e6)
        out["applied"].append(L)
        out["total"].append(arr[k] + L - S)
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
    off = local_offset(rec["state_seq"], rec["state_t"])
    win = lag_windows(rec, off)
    usb = (rec["state_t"] - rec["state_seq"] * TICK_NS - off) / 1e6
    print(f"{os.path.basename(flt)}: {len(rec['state_seq'])} states, {len(rec['cmd_t'])} "
          f"commands, {len(win['lag'])} stretches of walking timed")
    print(_row("usb in", usb, "above the fastest packet nearby"))
    if not len(win["lag"]):
        print("  no stretch could be timed - was the robot armed and walking, with the "
              "policy moving its residual?")
        return 2
    print(_row("applied", win["lag"], "tick a command arrived -> board target"))

    arr = command_ticks(rec, off)
    lower = []
    for k in range(len(arr)):
        L = lag_at(win, arr[k])
        if L is not None and rec["cmd_last_seq"][k] >= 0:
            lower.append(arr[k] + L - rec["cmd_last_seq"][k])
    lower = np.asarray(lower)
    print(_row("total >=", lower, "newest state the Pi had -> board target"))

    if timing is None:
        timing = find_timing(flt, rec["state_t"][0], rec["state_t"][-1])
    total = lower
    if timing:
        sg = stages(rec, off, win, load_timing(timing))
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
