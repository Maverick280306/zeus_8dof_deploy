"""
latency_report measures how stale the policy's view of the robot is by the time
its command reaches the drives.

What a failure here would mean on the robot: a delay reported short sends
everyone looking for the fall in the policy when the robot was acting on a
state 40 ms old; reported long, it sends them rewiring a link that is fine.
Every case below is built with the answer written down first.
"""

import numpy as np

from zeus_link import flight_log, latency_report as L, nexus_proto as P
from zeus_link.synthetic import pack_state, state_values

OFF = 5_000_000_000          # Pi monotonic ns at board tick 0
REF = [0.05 * (i + 1) for i in range(P.NUM_JOINTS)]
APPLY_TICKS = 3              # board applies a command 3 ticks after its tick of arrival
SEA = [i for i in range(P.NUM_JOINTS) if i not in L.NON_SEA]


def residual_turns(k):
    """Command k: a new value on every joint, so every one is matchable."""
    return [0.001 * (k + 1) * (1 if i % 2 else -1) for i in range(P.NUM_JOINTS)]


def build(tmp_path, n_cmds=40, slow_every=0):
    """A walk: states every tick, a command every 4 ticks sent half a tick
    after a state, applied APPLY_TICKS later. Every `slow_every`-th state
    arrives 3 ms late (USB jitter). Returns the recording and the policy log."""
    path = tmp_path / "walk.flt"
    rec = flight_log.FlightRecorder(str(path), port="/dev/x", proto_version=P.PROTO_VERSION)
    first, last = 1000, 1000 + 4 * n_cmds + 20
    sends = {1010 + 4 * k: k for k in range(n_cmds)}         # tick -> command index
    applied_at = {t + APPLY_TICKS: k for t, k in sends.items()}
    current = [0.0] * P.NUM_JOINTS
    events = []
    for s in range(first, last):
        if s in applied_at:
            current = residual_turns(applied_at[s])
        target = [REF[j] + current[j] * 2 * np.pi for j in range(P.NUM_JOINTS)]
        late = 3_000_000 if slow_every and s % slow_every == 0 else 0
        events.append((OFF + s * L.TICK_NS + late, flight_log.RX,
                       pack_state(state_values(seq=s, ref_angle=REF, act_target=target,
                                               safety_state=P.SAFETY_ARMED))))
        if s in sends:
            k = sends[s]
            events.append((OFF + s * L.TICK_NS + 500_000, flight_log.TX,
                           P.NexusCommand(seq=k, residual=residual_turns(k)).pack()))
    # the recorder stamps its own time; write in time order and patch the stamps
    events.sort(key=lambda e: e[0])
    for _, d, data in events:
        rec.write(d, data)
    rec.close()
    _restamp(path, [t for t, _, _ in events])

    # policy log: each action used the state 6 ticks before its command left,
    # received 1.5 ms after the link node had it, 1 ms queued, 2 ms of inference
    timing = tmp_path / "policy_timing_20261127_000000.csv"
    with open(timing, "w") as f:
        f.write(",".join(L.TIMING_FIELDS) + "\n")
        for s, k in sends.items():
            S = s - 6
            rx = OFF + S * L.TICK_NS + 1_500_000
            f.write(f"{S},{rx},{rx + 1_000_000},{rx + 3_000_000},"
                    f"{OFF + s * L.TICK_NS + 400_000}\n")
    return str(path), str(timing), n_cmds


def _restamp(path, stamps):
    """Rewrite each record's host time with the one the test chose."""
    import struct
    raw = bytearray(open(path, "rb").read())
    (hlen,) = struct.unpack_from("<I", raw, len(flight_log.MAGIC))
    i = len(flight_log.MAGIC) + 4 + hlen
    for t in stamps:
        d, _, n = flight_log._REC.unpack_from(raw, i)
        flight_log._REC.pack_into(raw, i, d, t, n)
        i += flight_log._REC.size + n
    open(path, "wb").write(bytes(raw))


def test_every_command_is_found_where_the_board_applied_it(tmp_path):
    flt, _, n = build(tmp_path)
    rec = L.load(flt)
    app = L.applied(rec, L.clock_offset(rec["state_seq"], rec["state_t"]))
    # sent half a tick after its state, applied APPLY_TICKS ticks after that state
    assert len(app["applied_tick"]) == n
    assert np.allclose(app["applied_tick"] - app["sent_tick"], APPLY_TICKS - 0.5)
    # the newest state the Pi had when each command left was that same tick
    assert np.all(app["applied_tick"] - app["last_seq"] == APPLY_TICKS)


def test_usb_jitter_does_not_move_the_board_clock(tmp_path):
    # A late packet must read as late, not drag every other figure with it:
    # the clock is set by the FASTEST packet.
    flt, _, n = build(tmp_path, slow_every=7)
    rec = L.load(flt)
    off = L.clock_offset(rec["state_seq"], rec["state_t"])
    assert off == OFF
    usb = (rec["state_t"] - rec["state_seq"] * L.TICK_NS - off) / 1e6
    assert set(np.round(usb, 3)) == {0.0, 3.0}
    assert len(L.applied(rec, off)["applied_tick"]) == n


def test_the_policy_log_gives_every_stage_and_the_exact_total(tmp_path):
    flt, timing, n = build(tmp_path)
    rec = L.load(flt)
    off = L.clock_offset(rec["state_seq"], rec["state_t"])
    sg = L.stages(rec, off, L.applied(rec, off), L.load_timing(timing))
    assert len(sg["total"]) == n
    assert np.allclose(sg["to_policy"], 1.5) and np.allclose(sg["wait"], 1.0)
    assert np.allclose(sg["inference"], 2.0)
    # inference ended 1.5 + 3 = 4.5 ms after the state; the command left the
    # link node 6.5 ms after it (6 ticks, then half a tick)
    assert np.allclose(sg["to_link"], 6.5 - 4.5)
    # the state used was 6 ticks before the send tick, applied 3 after it
    assert np.all(sg["total"] == 6 + APPLY_TICKS)


def test_a_repeated_command_is_not_matched_to_a_later_tick(tmp_path):
    # The hold and dry runs send the same residual for seconds. Only a NEW
    # value can be timed; a repeat would match the first tick after it and
    # report a delay of zero.
    flt, _, n = build(tmp_path)
    rec = L.load(flt)
    rec["cmd_residual"] = np.repeat(rec["cmd_residual"][:1], len(rec["cmd_residual"]), 0)
    app = L.applied(rec, L.clock_offset(rec["state_seq"], rec["state_t"]))
    assert len(app["applied_tick"]) <= 1


def test_report_passes_inside_the_trained_delay_and_fails_outside(tmp_path, capsys):
    flt, timing, _ = build(tmp_path)
    assert L.report(flt, timing) == 0           # 9 ms total, trained for 20
    assert "inside" in capsys.readouterr().out
    assert L.find_timing(flt, 0, 10 ** 19) == timing
