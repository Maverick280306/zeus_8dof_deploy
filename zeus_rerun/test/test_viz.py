"""
The rerun logging runs end to end and produces a recording.

Skipped where rerun-sdk is not installed (it is a pip package, not a rosdep
key, so a plain `colcon test` on a fresh machine may not have it).
"""

import os
import sys

import pytest

rr = pytest.importorskip("rerun")

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WS, "zeus_rerun"))
sys.path.insert(0, os.path.join(WS, "zeus_link"))

from types import SimpleNamespace  # noqa: E402

from zeus_link import convert, nexus_proto as P  # noqa: E402
from zeus_link.synthetic import state_packet  # noqa: E402
from zeus_rerun import viz  # noqa: E402


def _packets(n):
    return [P.NexusState.parse(state_packet(seq=s)) for s in range(n)]


def test_log_packets_and_commands_to_a_file(tmp_path):
    path = tmp_path / "zeus.rrd"
    where = viz.open_sink("save", path=str(path))
    assert str(path) in where

    for pkt in _packets(50):
        viz.log_state(pkt)
        viz.log_residual(pkt.seq, [0.01] * 10)
    rr.disconnect()

    assert path.exists() and path.stat().st_size > 1000


def test_logs_from_a_ros_message_too(tmp_path):
    # Same function, fed the message type the ROS node receives.
    viz.open_sink("save", path=str(tmp_path / "msg.rrd"))
    msg = SimpleNamespace()
    convert.fill_state_msg(msg, _packets(1)[0])
    viz.log_state(msg)
    rr.disconnect()


def test_connect_needs_a_host():
    with pytest.raises(ValueError, match="host"):
        viz.open_sink("connect")


def test_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        viz.open_sink("bogus")


def test_a_flight_log_converts_with_every_packet_and_command(tmp_path):
    # Every packet logged, the lost one counted, the junk counted, the
    # command decoded. A converter that quietly dropped any of these would
    # make a recording look healthier than the run was.
    from zeus_link import flight_log
    from zeus_link.synthetic import pack_state, state_values
    from zeus_rerun import flight_to_rrd

    flt = tmp_path / "run.flt"
    rec = flight_log.FlightRecorder(str(flt), proto_version=P.PROTO_VERSION)
    rec.write(flight_log.RX, b"boot\r\n")                         # 6 junk bytes
    for seq in (100, 101, 102, 104):                               # 103 lost
        rec.write(flight_log.RX, pack_state(state_values(seq=seq)))
    rec.write(flight_log.TX, P.NexusCommand(seq=0, residual=[0.01] * P.NUM_JOINTS,
                                            flags=P.CMD_ENABLE).pack())
    rec.write(flight_log.TX, P.NexusGains(seq=1).pack())
    rec.close()

    s = flight_to_rrd.convert(str(flt), out=str(tmp_path / "run.rrd"))
    assert s["packets"] == 4 and s["logged"] == 4
    assert s["lost"] == 1 and s["junk_bytes"] == 6
    assert s["commands"] == 1 and s["gains"] == 1 and s["restarts"] == 0
    assert (tmp_path / "run.rrd").stat().st_size > 1000
