"""
A flight recording -> a Rerun recording, every packet, at the full 1 kHz.

    ros2 run zeus_rerun flight_to_rrd ~/zeus_logs/zeus_20261121_142530.flt
    python3 -m zeus_rerun.flight_to_rrd run.flt --out run.rrd --spawn

Run it on the LAPTOP. The robot records raw bytes because that is cheap; the
conversion is done afterwards, where it has the time. Nothing is decimated by
default - the reason for recording every packet is to look at every packet.

WHY COLUMNS. Logging packet by packet, the way the live view does, costs ~20
Rerun calls a packet and converted an hour of running in about an hour and a
half. Here the packets are parsed a minute at a time and each signal goes to
Rerun as one column per minute (rr.send_columns), which is the same data on
the same entity paths - so the live view's layout and colours apply - at a
small fraction of the cost.

What you get on top of the live view (zeus_rerun/viz.py):

  link/arrival_jitter_ms   when each packet reached the Pi, against when the
                           board stamped it, minus the best case. A flat line
                           near 0 is a healthy link; spikes are the Pi or USB
                           holding packets back, and they are what makes a
                           policy act on stale data.
  link/seq_gap             packets that never arrived (0 = none lost)
  policy/residual          every command the Pi sent, in rad
  policy/enable            1 while the Pi was authorising motion
  events                   gains messages, board restarts, unknown messages

Timelines: "robot" is the board's own clock (seq / 1000, as everywhere else).
"pi" is seconds since the recording started, on the Pi's clock - the only one
that keeps running across a board reboot, so use it when the board restarted.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

from zeus_link import nexus_proto as P
from zeus_link.flight_log import RX, TX, read_records

from . import viz

import rerun as rr     # viz has already explained how to install it if missing

TWO_PI = 2.0 * math.pi
RAD_TO_DEG = 180.0 / math.pi
_CMD = struct.Struct(P.COMMAND_FORMAT)

# Packets per block sent to Rerun: one minute of board time. Big enough that
# the per-call overhead vanishes, small enough that an hour never sits in RAM.
BLOCK = 60_000


def _decode_tx(data: bytes):
    """('cmd', seq, residual_turns, flags) | ('gains', seq) | None."""
    if len(data) == _CMD.size:
        f = _CMD.unpack(data)
        if f[0] == P.SYNC and f[1] == P.MSG_COMMAND and P.crc16(data[:-2]) == f[-1]:
            n = P.NUM_JOINTS
            return ("cmd", f[3], list(f[4:4 + n]), f[4 + n])
    if len(data) == P.GAINS_SIZE and data[:2] == P.SYNC_BYTES and data[2] == P.MSG_GAINS:
        (seq,) = struct.unpack_from("<I", data, 4)
        return ("gains", seq)
    return None


def _send(path: str, idx, values) -> None:
    rr.send_columns(path, indexes=idx, columns=rr.Scalars.columns(scalars=values))


def _index(robot_s, pi_s):
    return [rr.TimeColumn(viz.TIMELINE, duration=robot_s),
            rr.TimeColumn("pi", duration=pi_s)]


def _flush_states(pkts, pi_s, gaps, jitter, degrees: bool, twin=None) -> None:
    """One block of packets -> one send_columns per entity. Paths and series
    order match viz.log_state exactly, so the static styles line up."""
    if not pkts:
        return
    k = RAD_TO_DEG if degrees else 1.0

    def col(name):
        return np.array([getattr(p, name) for p in pkts], dtype=np.float64)

    idx = _index(np.array([p.seq for p in pkts], dtype=np.float64) / 1000.0,
                 np.asarray(pi_s, dtype=np.float64))

    pos, ref, cmd = col("joint_pos") * k, col("ref_angle") * k, col("act_target") * k
    for j, n in enumerate(P.JOINT_NAMES):
        _send(f"joints/{n}", idx, np.column_stack([pos[:, j], cmd[:, j], ref[:, j]]))
    _send("tracking", idx, cmd - pos)
    _send("velocity", idx, col("joint_vel") * k)
    _send("torque", idx, col("act_torque"))

    _send("estimator/pelvis_z", idx, col("pelvis_z"))
    _send("estimator/vel_hdg", idx, col("vel_hdg"))
    _send("estimator/fused_valid", idx, col("fused_valid"))
    _send("imu/gyro", idx, col("imu_gyro"))
    _send("imu/accel", idx, col("imu_accel"))
    _send("contacts", idx, col("contact"))
    _send("feet", idx, col("foot_z"))
    _send("springs", idx, col("spring_angle"))
    _send("gait/phase", idx, col("phase"))
    _send("status/safety_state", idx, col("safety_state"))
    _send("status/health_bits", idx, col("health"))
    _send("status/loop_us_max", idx, col("loop_us_max"))
    _send("status/overruns", idx, col("overruns"))

    _send("link/seq_gap", idx, np.asarray(gaps, dtype=np.float64))
    _send("link/arrival_jitter_ms", idx, np.asarray(jitter, dtype=np.float64))

    # 3D: pelvis pose (packet is w,x,y,z; Rerun wants x,y,z,w) and the feet at
    # their estimated heights under the hips, as in the live view.
    q = col("quat")
    fp = col("fused_pos")
    z = col("pelvis_z")
    rr.send_columns("world/pelvis", indexes=idx, columns=rr.Transform3D.columns(
        translation=np.column_stack([fp[:, 0], fp[:, 1], z]),
        quaternion=q[:, [1, 2, 3, 0]]))

    if twin is not None:
        twin.flush(pkts, pi_s, viz.TIMELINE)

    fz = col("foot_z")
    pts, lengths = [], []
    for i in range(len(pkts)):
        n = 0
        for f, dy in ((fz[i, 0], -viz.HIP_OFFSET_Y), (fz[i, 1], viz.HIP_OFFSET_Y)):
            if math.isfinite(f):
                pts.append((fp[i, 0], fp[i, 1] + dy, f))
                n += 1
        lengths.append(n)
    if pts:
        rr.send_columns("world/feet", indexes=idx, columns=rr.Points3D.columns(
            positions=np.array(pts)).partition(lengths=lengths))


def _flush_commands(cmds) -> None:
    if not cmds:
        return
    a = np.array(cmds, dtype=np.float64)       # robot_s, pi_s, enable, residual...
    idx = _index(a[:, 0], a[:, 1])
    _send("policy/residual", idx, a[:, 3:] * TWO_PI)
    _send("policy/enable", idx, a[:, 2])


def convert(path: str, out: str = "", mode: str = "save", decimate: int = 1,
            degrees: bool = False, host: str = "", port: int = viz.VIEWER_PORT,
            twin_3d: bool = True) -> dict:
    header, records = read_records(path)
    if header.get("proto_version") not in (None, 0, P.PROTO_VERSION):
        print(f"warning: recorded at protocol v{header['proto_version']}, this parser is "
              f"v{P.PROTO_VERSION}; packets of the other version will not parse",
              file=sys.stderr)

    out = out or os.path.splitext(path)[0] + ".rrd"
    print(viz.open_sink(mode, host=host, port=port, path=out, app_id="zeus_flight"))
    rr.log("link/arrival_jitter_ms", rr.SeriesLines(names=["jitter"]), static=True)
    rr.log("link/seq_gap", rr.SeriesLines(names=["lost"]), static=True)
    rr.log("policy/enable", rr.SeriesLines(names=["enable"]), static=True)

    # The CAD robot twice: commanded and real (twin.py). Off with --no-3d.
    twin = None
    if twin_3d:
        try:
            from .twin import Twin
            twin = Twin()
            twin.log_static()
        except (OSError, ValueError) as exc:
            print(f"3D twin skipped: {exc}", file=sys.stderr)

    t0 = int(header.get("start_mono_ns", 0)) or None
    buf = bytearray()
    last_seq = None
    best_offset = None      # min(host ms - board ms): the zero-latency line
    stats = dict(packets=0, logged=0, lost=0, commands=0, gains=0,
                 restarts=0, rx_bytes=0, junk_bytes=0)

    pkts, pi_s, gaps, jitter = [], [], [], []
    cmds = []

    def event(text, t_ns, level="INFO"):
        rr.set_time("pi", duration=(t_ns - t0) / 1e9)
        if last_seq is not None:
            viz.set_time(last_seq)
        rr.log("events", rr.TextLog(text, level=level))

    for direction, t_ns, data in records:
        if t0 is None:
            t0 = t_ns
        if direction == RX:
            stats["rx_bytes"] += len(data)
            buf += data
            while True:
                pkt, buf = P.NexusState.find_and_parse(buf)
                if pkt is None:
                    break
                stats["packets"] += 1

                gap = 0
                if last_seq is not None:
                    d = (pkt.seq - last_seq) & 0xFFFFFFFF
                    if pkt.seq < last_seq and d > 1_000_000:
                        stats["restarts"] += 1
                        best_offset = None
                        event(f"board restarted (seq {last_seq} -> {pkt.seq})", t_ns, "WARN")
                    elif d != 1:
                        gap = d - 1
                        stats["lost"] += gap
                last_seq = pkt.seq

                # Several packets can arrive in one read, all stamped with the
                # read's time, so jitter is measured against the best case.
                offset = t_ns / 1e6 - pkt.seq
                if best_offset is None or offset < best_offset:
                    best_offset = offset

                # Decimating never hides a gap: the packet after one is kept.
                if stats["packets"] % decimate != 0 and gap == 0:
                    continue
                pkts.append(pkt)
                pi_s.append((t_ns - t0) / 1e9)
                gaps.append(gap)
                jitter.append(offset - best_offset)
                if len(pkts) >= BLOCK:
                    _flush_states(pkts, pi_s, gaps, jitter, degrees, twin)
                    stats["logged"] += len(pkts)
                    pkts, pi_s, gaps, jitter = [], [], [], []

        elif direction == TX:
            m = _decode_tx(data)
            if m is None:
                event(f"unrecognised {len(data)}-byte message to the board", t_ns, "WARN")
            elif m[0] == "cmd":
                stats["commands"] += 1
                _, _, turns, flags = m
                robot_s = (last_seq if last_seq is not None else 0) / 1000.0
                cmds.append([robot_s, (t_ns - t0) / 1e9,
                             float(bool(flags & P.CMD_ENABLE))] + list(turns))
                if len(cmds) >= BLOCK:
                    _flush_commands(cmds)
                    cmds = []
            else:
                stats["gains"] += 1
                event(f"gains message sent, seq {m[1]}", t_ns)

    _flush_states(pkts, pi_s, gaps, jitter, degrees, twin)
    stats["logged"] += len(pkts)
    _flush_commands(cmds)
    rr.disconnect()

    # Everything that arrived and did not become a packet: boot text, corrupted
    # packets, a protocol version this parser cannot read.
    stats["junk_bytes"] = stats["rx_bytes"] - stats["packets"] * P.STATE_SIZE - len(buf)
    stats["out"] = out
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flight_log", help=".flt file written by link_node")
    ap.add_argument("--out", default="", help=".rrd to write (default: next to the .flt)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--spawn", action="store_true", help="open a viewer here instead of saving")
    g.add_argument("--connect", metavar="HOST", help="stream to a viewer on HOST")
    ap.add_argument("--decimate", type=int, default=1,
                    help="log every Nth packet (default 1: all of them). The packet "
                         "after a gap is always logged.")
    ap.add_argument("--degrees", action="store_true")
    ap.add_argument("--no-3d", action="store_true",
                    help="skip the commanded-vs-real CAD robots (smaller, faster)")
    a = ap.parse_args(argv)

    mode = "spawn" if a.spawn else "connect" if a.connect else "save"
    try:
        s = convert(a.flight_log, out=a.out, mode=mode, decimate=max(1, a.decimate),
                    degrees=a.degrees, host=a.connect or "",
                    twin_3d=not a.no_3d)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"{s['packets']} packets ({s['packets'] / 1000.0:.1f} s of board time), "
          f"{s['logged']} logged, {s['lost']} lost, {s['junk_bytes']} junk bytes, "
          f"{s['commands']} commands, {s['gains']} gains messages, "
          f"{s['restarts']} board restarts")
    if mode == "save":
        print(f"open it with:  rerun {s['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
