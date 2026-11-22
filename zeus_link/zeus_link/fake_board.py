"""
A fake STM32 on a pseudo-terminal: run the whole Pi stack with no robot.

    ros2 run zeus_link fake_board                        # prints the port to use
    ros2 launch zeus_bringup walk.launch.py port:=/tmp/zeus_fake_board enable:=true

Linux and macOS only (it needs a pty).

It streams byte-exact 444-byte state packets at 1 kHz - a smooth fake stride
in joint_pos and ref_angle, the stride clock in phase - and parses every
command packet that comes back, so everything between the USB port and your
policy runs for real: link_node, the messages, QoS, the policy, rerun.

It models the parts of the firmware's safety.c the Pi has to get right:

  BOOT -> IDLE     commands with enable false arrive; clears the re-arm latch
  IDLE -> ARMED    enable true, with the latch clear and |residual| <= 0.1 turn
  IDLE/ARMED -> FAULT   200 ms without a command; sets the latch
  sequence         an enabled command must advance seq past the last accepted
                   one; enable false starts a new session (a restarted link
                   node counts from 0 again). 10 rejections in a row -> FAULT.

It does NOT model the robot: joint_pos does not follow your residual, and the
estimator values are fixed. It tests plumbing and protocol, not a policy's
control quality.

With --summary it writes what it saw on exit - commands, CRC errors, sequence
gaps, every safety transition - which is what the CI end-to-end test checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import struct
import sys
import threading
import time
import tty

from . import nexus_proto as P
from .synthetic import pack_state, state_values

DEFAULT_LINK = "/tmp/zeus_fake_board"
RESIDUAL_LIMIT_TURNS = 0.1
LINK_TIMEOUT_S = 0.2
MAX_REJECTS = 10
SOFT_RAMP_S = 3.0      # soft_start.h SOFT_START_RAMP_S


class FakeBoard:

    def __init__(self, link_path: str, error_at: float = -1.0, gait: str = "",
                 ankle_offset_deg: float = 0.0) -> None:
        self.master, slave = os.openpty()
        tty.setraw(slave)
        self._slave = slave
        self.link_path = link_path
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(os.ttyname(slave), link_path)

        self.error_at = error_at
        # --gait: play the policy's own stride (gait_q, gait_period from its
        # policy.npz) instead of the synthetic one, and model what the soft
        # start does to the legs - standing at zero while idle, the ramp, the
        # hold, then the stride. Without it the synthetic stride is kept, which
        # is what the "wrong stored gait" end-to-end case relies on.
        self.gait_q = self.gait_period = None
        if gait:
            import numpy as np
            z = np.load(gait, allow_pickle=False)
            self.gait_q = np.asarray(z["gait_q"], dtype=np.float64)
            self.gait_period = float(z["gait_period"])
        self.ramp_from = [0.0] * P.NUM_JOINTS
        # --ankle-offset: a right ankle whose zero was lost, as seen on the
        # robot (6.5 and 10 deg). What the ankle pre-check must catch.
        self.ankle_offset = math.radians(ankle_offset_deg)
        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        self.safety = P.SAFETY_BOOT
        self.gains_seq = 0
        self.needs_rearm = False
        self.last_cmd = None
        self.gains = None
        self.stats = dict(commands=0, crc_bad=0, seq_gaps=0, enables=0, disables=0,
                          gains=0, gains_refused=0,
                          rejected=0, packets_sent=0)
        self.events = []
        self._last_cmd_seq = None
        self._accepted_seq = None       # None = new session, like s_seq_valid
        self._reject_run = 0
        self._stop = threading.Event()
        # Soft start (v11), the board's soft_start.c in miniature.
        self.soft = "idle"              # idle | ramp | hold | run | return | home
        self.soft_hold = False          # HOLD as it stands
        self.soft_ramp_held = False     # the ramp heads for the frozen pose
        self.soft_t0 = 0.0
        self.zero_prev = False
        self.ankles_zeroed = False

    # ---- state machine ------------------------------------------------------

    def _set(self, new: int, why: str) -> None:
        if self.safety != new:
            t = round(time.monotonic() - self.t0, 3)
            self.events.append([t, P.SAFETY_NAMES[self.safety], P.SAFETY_NAMES[new], why])
            print(f"  {t:8.3f} s  {P.SAFETY_NAMES[self.safety]} -> {P.SAFETY_NAMES[new]}  ({why})",
                  flush=True)
            self.safety = new

    def _on_command(self, frame: bytes) -> None:
        f = struct.unpack(P.COMMAND_FORMAT, frame)
        seq, residual, flags = f[3], f[4:4 + P.NUM_JOINTS], f[4 + P.NUM_JOINTS]
        s = self.stats
        if self._last_cmd_seq is not None and seq != (self._last_cmd_seq + 1) & 0xFFFFFFFF:
            s["seq_gaps"] += 1
        self._last_cmd_seq = seq
        s["commands"] += 1
        self.last_cmd = time.monotonic()

        zero = bool(flags & P.CMD_ZERO_ANKLES)
        if zero and not self.zero_prev and self.safety != P.SAFETY_ARMED:
            self.ankles_zeroed = True
            self.ankle_offset = 0.0     # Set_Absolute_Position(0): it reads 0 now
            s.setdefault("ankle_zeroings", 0)
            s["ankle_zeroings"] += 1
        self.zero_prev = zero

        if not flags & P.CMD_ENABLE:
            s["disables"] += 1
            self.needs_rearm = False
            self._accepted_seq = None
            self._set(P.SAFETY_IDLE, "enable false")
            return

        s["enables"] += 1
        if self.needs_rearm or self.safety in (P.SAFETY_BOOT, P.SAFETY_FAULT):
            return
        stale = (self._accepted_seq is not None and
                 struct.unpack("<i", struct.pack("<I", (seq - self._accepted_seq) & 0xFFFFFFFF))[0] <= 0)
        bad = any(not math.isfinite(r) or abs(r) > RESIDUAL_LIMIT_TURNS for r in residual)
        if stale or bad:
            s["rejected"] += 1
            self._reject_run += 1
            if self._reject_run >= MAX_REJECTS:
                self.needs_rearm = True
                self._set(P.SAFETY_FAULT, f"{MAX_REJECTS} rejected commands")
            return
        self._reject_run = 0
        self._accepted_seq = seq
        hold = bool(flags & P.CMD_HOLD)
        if self.safety != P.SAFETY_ARMED:
            self.soft, self.soft_t0 = "ramp", time.monotonic()
            self.soft_ramp_held = hold
        self.soft_hold = hold
        if self.soft == "hold" and not hold:
            self.soft = "run"
        held = self.soft == "hold" or (self.soft == "ramp" and self.soft_ramp_held)
        if flags & P.CMD_RETURN and held and self.safety == P.SAFETY_ARMED:
            self.soft, self.soft_t0 = "return", time.monotonic()
            self.return_from = list(getattr(self, "_last_pos", self.ramp_from))
        self._set(P.SAFETY_ARMED, "enable true" + (", HOLD" if hold else ""))

    def _on_gains(self, frame: bytes) -> None:
        """Gains land only between runs, exactly as the board treats them."""
        f = struct.unpack(P.GAINS_FORMAT, frame)
        seq = f[3]
        self.stats["gains"] += 1

        if self.safety == P.SAFETY_ARMED:
            self.stats["gains_refused"] += 1
            return

        self.gains = dict(pos=f[4:4 + P.NUM_JOINTS],
                          vel=f[4 + P.NUM_JOINTS:4 + 2 * P.NUM_JOINTS],
                          vel_int=f[4 + 2 * P.NUM_JOINTS:4 + 3 * P.NUM_JOINTS])
        self.gains_seq = seq & 0xFF

    # ---- threads --------------------------------------------------------------

    def _reader(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            try:
                buf += os.read(self.master, 4096)
            except OSError:
                return
            while True:
                i = buf.find(P.SYNC_BYTES)
                if i < 0:
                    del buf[:-1]
                    break
                if len(buf) - i < 3:
                    del buf[:i]
                    break
                # Two message types, two lengths - the real board decides the
                # same way, from the id in the third byte.
                size = {P.MSG_COMMAND: P.COMMAND_SIZE, P.MSG_GAINS: P.GAINS_SIZE}.get(buf[i + 2])
                if size is None:
                    del buf[:i + 1]
                    continue
                if len(buf) - i < size:
                    del buf[:i]
                    break
                frame = bytes(buf[i:i + size])
                if P.crc16(frame[:-2]) != struct.unpack_from("<H", frame, size - 2)[0]:
                    self.stats["crc_bad"] += 1
                    del buf[:i + 1]
                    continue
                del buf[:i + size]
                with self.lock:
                    if frame[2] == P.MSG_COMMAND:
                        self._on_command(frame)
                    else:
                        self._on_gains(frame)

    def run(self, duration: float) -> None:
        threading.Thread(target=self._reader, daemon=True).start()
        seq, nxt = 0, time.monotonic()
        while not self._stop.is_set() and (duration <= 0 or time.monotonic() - self.t0 < duration):
            now = time.monotonic()
            with self.lock:
                link_ok = self.last_cmd is not None and now - self.last_cmd < LINK_TIMEOUT_S
                if not link_ok and self.safety in (P.SAFETY_IDLE, P.SAFETY_ARMED):
                    self.needs_rearm = True
                    self._set(P.SAFETY_FAULT, "no command for 200 ms")
                safety = self.safety

            with self.lock:
                if safety != P.SAFETY_ARMED:
                    self.soft = "idle"
                elif self.soft == "ramp" and now - self.soft_t0 >= SOFT_RAMP_S:
                    self.soft = "hold" if (self.soft_ramp_held and self.soft_hold) else "run"
                elif self.soft == "return" and now - self.soft_t0 >= SOFT_RAMP_S:
                    self.soft = "home"
                soft = self.soft
                frozen = ((soft == "ramp" and self.soft_ramp_held)
                          or soft in ("hold", "return", "home"))
                stream = (P.STREAM_GAIT_LIVE
                          | (P.STREAM_RAMP if soft in ("ramp", "return") else 0)
                          | (P.STREAM_RETURN if soft in ("return", "home") else 0)
                          | (P.STREAM_HOLD if soft == "hold" else 0)
                          | (P.STREAM_ANKLES_ZEROED if self.ankles_zeroed else 0))
            if frozen:
                clock_seq = 0.0
            else:
                clock_seq = getattr(self, "_clock", 0.0) + 1.0
            self._clock = clock_seq
            period = self.gait_period if self.gait_q is not None else 1.26
            phase = (clock_seq / (1000.0 * period)) % 1.0
            w = 2.0 * math.pi * phase
            errors = [0] * P.NUM_JOINTS
            if self.error_at >= 0 and self.error_at <= now - self.t0 < self.error_at + 1.0:
                errors[P.JOINT_INDEX["right_ankle_pitch"]] = 0x08000200
            if self.gait_q is None:
                ref = [0.2 * math.sin(w + j + 0.05) for j in range(P.NUM_JOINTS)]
                # Standing still until armed, like the robot.
                pos = ([0.2 * math.sin(w + j) for j in range(P.NUM_JOINTS)]
                       if safety == P.SAFETY_ARMED else [0.0] * P.NUM_JOINTS)
                extra = {}
            else:
                n = self.gait_q.shape[0]
                x = phase * n
                i = int(x) % n
                f = x - int(x)
                ref = list((1.0 - f) * self.gait_q[i] + f * self.gait_q[(i + 1) % n])
                if soft == "idle":
                    pos = [0.0] * P.NUM_JOINTS          # standing, the zero pose
                    self.ramp_from = list(pos)
                elif soft == "ramp":
                    xr = min(1.0, (now - self.soft_t0) / SOFT_RAMP_S)
                    sm = xr * xr * (3.0 - 2.0 * xr)
                    pos = [a + sm * (b - a) for a, b in zip(self.ramp_from, ref)]
                elif soft == "return":
                    xr = min(1.0, (now - self.soft_t0) / SOFT_RAMP_S)
                    sm = xr * xr * (3.0 - 2.0 * xr)
                    pos = [a + sm * (b - a) for a, b in zip(self.return_from, self.ramp_from)]
                elif soft == "home":
                    pos = list(self.ramp_from)
                else:
                    pos = list(ref)
                self._last_pos = list(pos)
                # A healthy, converged estimator over a robot on its feet -
                # what the policy's arm guards need to see.
                extra = dict(fused_valid=P.FUSION_OK, fk_valid=3,
                             quat=[1.0, 0.0, 0.0, 0.0], pelvis_z=0.77,
                             foot_z=[0.0, 0.0], contact=[1.0, 1.0])
            pos = list(pos)
            pos[P.JOINT_INDEX["right_ankle_pitch"]] += self.ankle_offset
            pkt = pack_state(state_values(
                seq=seq, timestamp_us=(seq * 1000) & 0xFFFFFFFF, phase=phase,
                joint_pos=pos,
                joint_vel=[0.2 * w * math.cos(w + j) for j in range(P.NUM_JOINTS)],
                ref_angle=ref,
                health=0 if link_ok else P.HEALTH_LINK, safety_state=safety,
                act_state=[8 if safety == P.SAFETY_ARMED else 1] * P.NUM_JOINTS,
                # What the drives were told: the reference the board is playing
                # while armed, and NaN while nothing is being driven.
                act_target=(list(ref) if safety == P.SAFETY_ARMED
                            else [float("nan")] * P.NUM_JOINTS),
                gains_seq=self.gains_seq,
                # Every drive reporting, telemetry and heartbeat. The synthetic
                # template's own act_flags are deliberately distinct per joint
                # (to catch a field swap) and mark one drive silent.
                act_flags=[P.ACT_TELEM_FRESH | P.ACT_HB_FRESH] * P.NUM_JOINTS,
                stream_flags=stream,
                act_error=errors, **extra))
            try:
                os.write(self.master, pkt)
            except OSError:
                pass
            seq += 1
            nxt += 0.001
            time.sleep(max(0.0, nxt - time.monotonic()))
        self.stats["packets_sent"] = seq

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if os.path.lexists(self.link_path):
            os.remove(self.link_path)
        os.close(self.master)
        os.close(self._slave)

    def summary(self) -> dict:
        return dict(self.stats, events=self.events, final=P.SAFETY_NAMES[self.safety],
                    elapsed_s=round(time.monotonic() - self.t0, 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", default=DEFAULT_LINK, help="symlink to create for the pty")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until ctrl-c")
    ap.add_argument("--summary", help="write a JSON summary here on exit")
    ap.add_argument("--ankle-offset", type=float, default=0.0,
                    help="deg added to the right ankle's reading: a lost zero")
    ap.add_argument("--gait", default="",
                    help="a policy.npz: play its stride, and model the soft start on the legs")
    ap.add_argument("--error-at", type=float, default=-1.0,
                    help="report an ODrive error on right_ankle_pitch for 1 s from this time")
    a = ap.parse_args(argv)

    board = FakeBoard(a.link, error_at=a.error_at, gait=a.gait,
                      ankle_offset_deg=a.ankle_offset)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: board.stop())
    print(f"fake STM32 streaming 1 kHz on {a.link}\n"
          f"  ros2 launch zeus_bringup walk.launch.py port:={a.link}", flush=True)
    try:
        board.run(a.duration)
    finally:
        s = board.summary()
        board.close()
        if a.summary:
            with open(a.summary, "w") as f:
                json.dump(s, f, indent=1)
        print(f"sent {s['packets_sent']} packets, received {s['commands']} commands, "
              f"{s['crc_bad']} bad, final state {s['final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
