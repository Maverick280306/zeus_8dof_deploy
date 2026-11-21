"""
The flight recorder keeps every byte, in order, in both directions.

What a failure here would mean on the robot: the one run worth looking at
afterwards - the fall, the fault, the link that stopped parsing - comes back
from the Pi with holes in it, or does not open at all.
"""

import struct
import time

import serial

from zeus_link import flight_log, nexus_link, nexus_proto as P
from zeus_link.synthetic import pack_state, state_values as build_state


def test_round_trip_keeps_every_record_in_order(tmp_path):
    path = tmp_path / "run.flt"
    rec = flight_log.FlightRecorder(str(path), port="/dev/x", proto_version=P.PROTO_VERSION)
    rec.write(flight_log.RX, b"boot text\r\n")
    rec.write(flight_log.RX, b"\xa5\xa5abc")
    rec.write(flight_log.TX, b"cmd")
    rec.write(flight_log.RX, b"")              # nothing to record, nothing recorded
    rec.close()

    header, records = flight_log.read_records(str(path))
    got = list(records)
    assert header["proto_version"] == P.PROTO_VERSION and header["port"] == "/dev/x"
    assert [(d, data) for d, _, data in got] == [
        (flight_log.RX, b"boot text\r\n"), (flight_log.RX, b"\xa5\xa5abc"),
        (flight_log.TX, b"cmd")]
    times = [t for _, t, _ in got]
    assert times == sorted(times)              # host time never runs backwards


def test_a_record_cut_off_by_a_power_cut_ends_the_log_quietly(tmp_path):
    # The Pi losing power mid-write leaves half a record. Everything before
    # it must still read; the torn tail is the one part allowed to be missing.
    path = tmp_path / "torn.flt"
    rec = flight_log.FlightRecorder(str(path))
    rec.write(flight_log.RX, b"first")
    rec.write(flight_log.RX, b"second, about to be torn")
    rec.close()
    raw = path.read_bytes()
    path.write_bytes(raw[:-5])

    _, records = flight_log.read_records(str(path))
    assert [data for _, _, data in records] == [b"first"]


def test_a_failed_write_stops_the_recorder_not_the_caller(tmp_path):
    # Disk full on the Pi: the link's reader thread calls write() at 1 kHz,
    # and an exception there would kill the thread that publishes /zeus/state.
    rec = flight_log.FlightRecorder(str(tmp_path / "full.flt"))

    class Full:
        def write(self, _):
            raise OSError(28, "No space left on device")

        def close(self):
            pass

    rec._f = Full()
    rec.write(flight_log.RX, b"x")            # must not raise
    rec.write(flight_log.RX, b"y")
    assert not rec.active and "No space" in rec.error


def test_the_link_records_rx_before_parsing_and_every_command_sent(monkeypatch, tmp_path):
    # Junk, then two good packets: all three must be in the recording, because
    # the run where parsing broke is the run that most needs the raw bytes.
    loop = serial.serial_for_url("loop://", timeout=0.005)
    monkeypatch.setattr(nexus_link.serial, "Serial", lambda *a, **k: loop)
    path = tmp_path / "link.flt"
    rec = flight_log.FlightRecorder(str(path))

    link = nexus_link.NexusLink("loop://", recorder=rec)
    link.start()
    try:
        sent = b"junk!" + pack_state(build_state(seq=1)) + pack_state(build_state(seq=2))
        loop.write(sent)
        deadline = time.monotonic() + 2.0
        while link.stats.packets < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert link.stats.packets == 2
    finally:
        link.stop()

    # A command, written with the reader stopped so the loopback does not
    # echo it back in as RX.
    link._ser = type("S", (), {"write": lambda self, b: None})()
    link.send_command([0.02] * P.NUM_JOINTS, enable=True)
    rec.close()

    _, records = flight_log.read_records(str(path))
    rx = b"".join(d for k, _, d in records if k == flight_log.RX)
    assert rx == sent

    _, records = flight_log.read_records(str(path))
    tx = [d for k, _, d in records if k == flight_log.TX]
    assert len(tx) == 1 and len(tx[0]) == P.COMMAND_SIZE
    f = struct.unpack(P.COMMAND_FORMAT, tx[0])
    assert abs(f[4] - 0.02) < 1e-7 and f[-2] & P.CMD_ENABLE


def test_default_path_is_one_sortable_file_per_run(tmp_path):
    p = flight_log.default_path(str(tmp_path))
    assert p.startswith(str(tmp_path)) and p.endswith(".flt") and "zeus_" in p
