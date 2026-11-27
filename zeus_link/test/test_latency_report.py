"""
latency_report measures how stale the policy's view of the robot is by the time
its command reaches the drives.

What a failure here would mean on the robot: a delay reported short sends
everyone looking for the fall in the policy when the robot was acting on a
state 40 ms old; reported long, it sends them rewiring a link that is fine.
Every case below is built with the answer written down first - including the
three things the real board does that made the first version match nothing on
the robot: targets every second tick against a moving reference, a constant
offset on the ankles, and clocks that drift apart.
"""

import math
import struct

import numpy as np

from zeus_link import flight_log, latency_report as L, nexus_proto as P
from zeus_link.synthetic import pack_state, state_values

OFF = 5_000_000_000          # Pi monotonic ns at board tick 0
APPLY_TICKS = 4              # a command arriving on tick s is in the target from s + 4
USED_BEFORE = 6              # each action used the state 6 ticks before its command left
ANKLES = (P.JOINT_NAMES.index("left_ankle_pitch"), P.JOINT_NAMES.index("right_ankle_pitch"))


def ref_at(s, moving):
    base = [0.05 * (j + 1) for j in range(P.NUM_JOINTS)]
    if not moving:
        return base
    return [b + 0.3 * math.sin(s / 40.0 + j) for j, b in enumerate(base)]   # ~0.4 deg/tick


def residual_turns(k):
    """Command k: every joint moving, 0.06 rad of travel, so every window times."""
    return [0.01 * math.sin(k / 5.0 + j) for j in range(P.NUM_JOINTS)]


def build(tmp_path, n_cmds=250, divider=False, moving_ref=False, ankle_offset=0.0,
          drift_ppm=0.0, slow_every=0):
    """A walk: a state every tick, a command every 4 ticks (sent half a tick
    after an even state), each in the target APPLY_TICKS later. Returns the
    recording, the policy's timing log, and how many commands were sent."""
    path = tmp_path / "walk.flt"
    rec = flight_log.FlightRecorder(str(path), port="/dev/x", proto_version=P.PROTO_VERSION)
    sends = {1010 + 4 * k: k for k in range(n_cmds)}
    live = {s + APPLY_TICKS: k for s, k in sends.items()}

    def pi_time(s, extra=0.0):
        return int(OFF + (s * L.TICK_NS + extra) * (1.0 + drift_ppm * 1e-6))

    current = [0.0] * P.NUM_JOINTS
    target = None
    events = []
    for s in range(1000, 1010 + 4 * n_cmds + 20):
        if s in live:
            current = residual_turns(live[s])
        ref = ref_at(s, moving_ref)
        if target is None or not divider or s % 2 == 0:
            target = [ref[j] + current[j] * 2 * math.pi + (ankle_offset if j in ANKLES else 0.0)
                      for j in range(P.NUM_JOINTS)]
        late = 3_000_000 if slow_every and s % slow_every == 0 else 0
        events.append((pi_time(s, late), flight_log.RX,
                       pack_state(state_values(seq=s, ref_angle=ref, act_target=target,
                                               safety_state=P.SAFETY_ARMED))))
        if s in sends:
            k = sends[s]
            events.append((pi_time(s, 500_000), flight_log.TX,
                           P.NexusCommand(seq=k, residual=residual_turns(k)).pack()))
    events.sort(key=lambda e: e[0])
    for _, d, data in events:
        rec.write(d, data)
    rec.close()
    _restamp(path, [t for t, _, _ in events])

    # each action: its state reached the policy 1.5 ms after the link node had
    # it, waited 1 ms for the tick, took 2 ms; published 0.1 ms before it left
    timing = tmp_path / "policy_timing_20261127_000000.csv"
    with open(timing, "w") as f:
        f.write(",".join(L.TIMING_FIELDS) + "\n")
        for s in sends:
            S = s - USED_BEFORE
            rx = pi_time(S) + 1_500_000
            f.write(f"{S},{rx},{rx + 1_000_000},{rx + 3_000_000},{pi_time(s, 400_000)}\n")
    return str(path), str(timing), n_cmds


def _restamp(path, stamps):
    """Rewrite each record's host time with the one the test chose."""
    raw = bytearray(open(path, "rb").read())
    (hlen,) = struct.unpack_from("<I", raw, len(flight_log.MAGIC))
    i = len(flight_log.MAGIC) + 4 + hlen
    for t in stamps:
        d, _, n = flight_log._REC.unpack_from(raw, i)
        flight_log._REC.pack_into(raw, i, d, t, n)
        i += flight_log._REC.size + n
    open(path, "wb").write(bytes(raw))


def lags(flt):
    rec = L.load(flt)
    off = L.local_offset(rec["state_seq"], rec["state_t"])
    return L.lag_windows(rec, off)["lag"]


def test_the_delay_is_found_on_a_plain_board(tmp_path):
    flt, _, _ = build(tmp_path)
    got = lags(flt)
    assert len(got) >= 4 and np.all(got == APPLY_TICKS)


def test_the_delay_is_found_on_the_real_board_s_habits(tmp_path):
    # Targets every second tick against a reference that moves 0.4 deg a tick,
    # a 0.4 deg constant on the ankles, 145 ppm of clock drift: all three at
    # once, as on Zeus 2026-11-27. The answer must not move.
    flt, _, _ = build(tmp_path, divider=True, moving_ref=True, ankle_offset=0.0065,
                      drift_ppm=145.0)
    got = lags(flt)
    assert len(got) >= 4 and np.all(got == APPLY_TICKS)


def test_clock_drift_is_not_reported_as_delay(tmp_path):
    # 145 ppm over this walk is 0.15 ms; over a four-minute run it was 32 ms,
    # which a single offset for the whole run reported as USB delay.
    flt, _, _ = build(tmp_path, drift_ppm=145.0, slow_every=7)
    rec = L.load(flt)
    off = L.local_offset(rec["state_seq"], rec["state_t"])
    usb = (rec["state_t"] - rec["state_seq"] * L.TICK_NS - off) / 1e6
    on_time = usb[rec["state_seq"] % 7 != 0]
    late = usb[rec["state_seq"] % 7 == 0]
    assert np.max(on_time) < 0.5            # drift stays under half a millisecond
    assert np.all(np.abs(late - 3.0) < 0.5)  # a late packet still reads as 3 ms late


def test_the_policy_log_gives_every_stage_and_the_exact_total(tmp_path):
    flt, timing, n = build(tmp_path, divider=True, moving_ref=True)
    rec = L.load(flt)
    off = L.local_offset(rec["state_seq"], rec["state_t"])
    sg = L.stages(rec, off, L.lag_windows(rec, off), L.load_timing(timing))
    assert len(sg["total"]) > 0.8 * n
    assert np.allclose(sg["to_policy"], 1.5) and np.allclose(sg["wait"], 1.0)
    assert np.allclose(sg["inference"], 2.0)
    # inference ended 4.5 ms after the state; the command left 6.5 ms after it
    assert np.allclose(sg["to_link"], 2.0)
    assert np.all(sg["applied"] == APPLY_TICKS)
    assert np.all(sg["total"] == USED_BEFORE + APPLY_TICKS)


def test_a_residual_that_never_moves_cannot_be_timed(tmp_path):
    # The hold and dry runs send the same residual for seconds. There is no
    # delay to see; reporting one would be inventing it.
    flt, _, _ = build(tmp_path)
    rec = L.load(flt)
    rec["cmd_residual"][:] = rec["cmd_residual"][:1]
    off = L.local_offset(rec["state_seq"], rec["state_t"])
    assert len(L.lag_windows(rec, off)["lag"]) == 0


def test_report_passes_inside_the_trained_delay(tmp_path, capsys):
    flt, timing, _ = build(tmp_path, divider=True, moving_ref=True, drift_ppm=145.0)
    assert L.report(flt, timing) == 0           # 10 ms total, trained for 20
    assert "inside" in capsys.readouterr().out
    assert L.find_timing(flt, 0, 10 ** 19) == timing
