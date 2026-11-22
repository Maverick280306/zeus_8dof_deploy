"""
The operator's window on the robot: a web page served by the Pi.

    ros2 launch zeus_bringup walk.launch.py policy:=rl      # starts this node too
    # the log prints the address, with its key:
    #   GUI: http://nexus.local:8080/?key=Xy3...
    # open that on the laptop

What it does, and what it deliberately does not:

  * It shows everything the robot reports, live: board and soft-start state,
    every joint's actual / commanded / reference angle, the estimator, the
    springs, the feet, health, and what the commander (rl_policy_node or the
    gait passthrough) says is blocking it.
  * It has four buttons - ZERO ANKLES, START, WALK, STOP - and each one only
    CALLS A SERVICE. It never publishes a command itself, so there is still
    exactly one thing commanding the robot: the commander node. Two command
    publishers would interleave enable true and false and fight each other.
  * STOP calls /zeus/stand_down on the link node first - which forwards every
    command with enable false from then on, whatever the commander does - and
    then the commander's own stop. START undoes the stand-down and asks the
    commander to arm into the HOLD. WALK is accepted by the commander only when
    it reports "ready".
  * DEAD-MAN. While the robot is armed from this page, the page must keep
    saying so: it sends a heartbeat every 200 ms. Nothing for HEARTBEAT_LOSS_S
    and this node stops the robot exactly as the STOP button does - a closed
    tab, a sleeping laptop, a dropped Wi-Fi link. The physical motor cut-off is
    still the first stop; this is the second.
  * A KEY. Anyone on the network could otherwise open the page and arm the
    robot. Every launch makes a random key (or takes the `key` parameter); the
    page must be opened with it, and every request without it is refused.

Python standard library only - http.server and Server-Sent Events - so nothing
has to be installed on the Pi.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from zeus_msgs.msg import NexusState

from .nexus_proto import (FUSION_OK, HEALTH_NAMES, JOINT_NAMES, SAFETY_ARMED, SAFETY_NAMES,
                          SPRING_NAMES, STREAM_ANKLES_ZEROED, STREAM_ARM_BLOCKED, STREAM_HOLD,
                          STREAM_RAMP)
from .odrive_names import axis_state_name, error_names
from .precheck import ANKLE_ZERO_TOL_DEG, ankle_zero_problems
from .qos import STATE_QOS

# No heartbeat this long while armed from the page -> STOP. Not 0.5: browsers
# throttle timers in a background tab to about once a second, so a shorter
# limit stops the robot every time the operator switches tabs.
HEARTBEAT_LOSS_S = 1.5
WINDOW = 50                 # zeus/state_viz samples kept for freshness, ~1 s at 50 Hz
SERVICE_TIMEOUT_S = 3.0
GRAVITY = 9.81


def _deg(v: float) -> Optional[float]:
    return round(math.degrees(v), 2) if math.isfinite(v) else None


def _tilt_deg(q) -> Optional[float]:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(n) or n < 1e-6:
        return None
    x, y = x / n, y / n
    c = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return round(math.degrees(math.acos(c)), 2)


class GuiNode(Node):

    def __init__(self) -> None:
        super().__init__("zeus_gui")
        self.port = int(self.declare_parameter("port", 8080).value)
        self.key = str(self.declare_parameter("key", "").value) or secrets.token_urlsafe(9)

        self.lock = threading.Lock()
        self.state: Optional[NexusState] = None
        self.state_t = 0.0
        self.window = collections.deque(maxlen=WINDOW)   # (act_flags, enc_valid)
        self.rate_seq = collections.deque(maxlen=200)     # (monotonic, seq)
        self.commander: Optional[dict] = None
        self.commander_t = 0.0

        self.armed_by_gui = False
        self.last_hb = 0.0
        self.events = collections.deque(maxlen=40)
        self.run = None              # the run in progress, while ARMED
        self.last_run = None         # its summary, once it ends
        self.was_armed = False

        cb = ReentrantCallbackGroup()
        # The 50 Hz display copy, NOT zeus/state: handling 1000 messages a second
        # in Python starved this node's web server (heartbeats answered in up to
        # 2.2 s - the dead-man fired on a page that was still open).
        self.create_subscription(NexusState, "zeus/state_viz", self._on_state, STATE_QOS,
                                 callback_group=cb)
        self.create_subscription(String, "zeus/commander_status", self._on_commander, 10,
                                 callback_group=cb)
        self.cli = {name: self.create_client(Trigger, srv, callback_group=cb) for name, srv in (
            ("cmd_start", "zeus/commander/start"), ("cmd_walk", "zeus/commander/walk"),
            ("cmd_stop", "zeus/commander/stop"), ("stand_down", "zeus/stand_down"),
            ("resume", "zeus/resume"), ("zero_ankles", "zeus/zero_ankles"))}
        self.create_timer(0.05, self._watchdog, callback_group=cb)

        host = socket.gethostname()
        self.url = f"http://{host}.local:{self.port}/?key={self.key}"
        self.get_logger().warn(f"GUI: {self.url}")
        self.event("GUI started")

    # ---- inputs --------------------------------------------------------------

    def _on_state(self, msg: NexusState) -> None:
        now = time.monotonic()
        with self.lock:
            self.state, self.state_t = msg, now
            self.window.append((tuple(msg.act_flags), int(msg.enc_valid)))
            self.rate_seq.append((now, int(msg.seq)))
            armed = msg.safety_state == SAFETY_ARMED
            if armed and not self.was_armed:
                self.run = dict(t0=now, max_err=[0.0] * len(JOINT_NAMES), max_tilt=0.0,
                                rec=self.recording(), soft_seen=set())
                self.event("board ARMED")
            if armed and self.run is not None:
                for j in range(len(JOINT_NAMES)):
                    e = float(msg.act_target[j]) - float(msg.joint_pos[j])
                    if math.isfinite(e):
                        self.run["max_err"][j] = max(self.run["max_err"][j], abs(e))
                t = _tilt_deg(msg.quat)
                if t is not None:
                    self.run["max_tilt"] = max(self.run["max_tilt"], t)
                if msg.stream_flags & STREAM_RAMP:
                    self.run["soft_seen"].add("ramp")
                if msg.stream_flags & STREAM_HOLD:
                    self.run["soft_seen"].add("hold")
            if not armed and self.was_armed and self.run is not None:
                why = (self.commander or {}).get("stop_reason") or SAFETY_NAMES.get(
                    msg.safety_state, "?").lower()
                health = [n for b, n in HEALTH_NAMES.items() if msg.health & b]
                self.last_run = dict(
                    duration_s=round(now - self.run["t0"], 2),
                    max_err_deg={JOINT_NAMES[j]: round(math.degrees(v), 2)
                                 for j, v in enumerate(self.run["max_err"])},
                    max_tilt_deg=round(self.run["max_tilt"], 2),
                    reached=sorted(self.run["soft_seen"]),
                    stopped_because=why + (f" (health: {', '.join(health)})" if health else ""),
                    recording=self.run["rec"])
                self.run = None
                self.event(f"board disarmed: {why}")
            self.was_armed = armed

    def _on_commander(self, msg: String) -> None:
        try:
            st = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            old = (self.commander or {}).get("mode")
            self.commander, self.commander_t = st, time.monotonic()
        if st.get("mode") != old:
            self.event(f"{st.get('node', 'commander')}: {st.get('mode')}")

    def event(self, text: str) -> None:
        self.events.appendleft([time.strftime("%H:%M:%S"), text])

    @staticmethod
    def recording() -> Optional[str]:
        files = glob.glob(os.path.expanduser("~/zeus_logs/*.flt"))
        return max(files, key=os.path.getmtime) if files else None

    # ---- the dead-man --------------------------------------------------------

    def _watchdog(self) -> None:
        if not self.armed_by_gui:
            return
        if time.monotonic() - self.last_hb > HEARTBEAT_LOSS_S:
            self.armed_by_gui = False
            self.get_logger().error(f"GUI heartbeat lost for {HEARTBEAT_LOSS_S} s while armed "
                                    "- STOPPING the robot")
            self.event("DEAD-MAN: page went quiet - robot stopped")
            threading.Thread(target=self.do_stop, args=("GUI heartbeat lost",),
                             daemon=True).start()

    # ---- actions -------------------------------------------------------------

    def call(self, name: str) -> tuple:
        """Never raises: this is on the STOP path, where an exception escaping
        would skip the rest of the stop. Seen at shutdown - the dead-man firing
        while ROS was being torn down - as an RCLError on a dead context."""
        try:
            cli = self.cli[name]
            if not cli.wait_for_service(timeout_sec=1.0):
                return False, f"service for {name} is not running"
            fut = cli.call_async(Trigger.Request())
            deadline = time.monotonic() + SERVICE_TIMEOUT_S
            while not fut.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not fut.done():
                return False, f"{name}: no answer within {SERVICE_TIMEOUT_S:g} s"
            r = fut.result()
            return bool(r.success), str(r.message)
        except Exception as exc:     # noqa: BLE001 - reported, never raised
            return False, f"{name}: {exc}"

    def do_stop(self, why: str = "operator STOP") -> tuple:
        # The link node first: from here every command reaches the board with
        # enable false, whatever the commander is doing.
        ok1, m1 = self.call("stand_down")
        ok2, m2 = self.call("cmd_stop")
        self.armed_by_gui = False
        self.event(f"STOP ({why})")
        return ok1, f"{m1} | {m2}" if ok2 else f"{m1} | commander: {m2}"

    def do_start(self) -> tuple:
        ok, msg = self.call("resume")
        if not ok:
            return False, f"resume refused: {msg}"
        ok, msg = self.call("cmd_start")
        if ok:
            self.last_hb = time.monotonic()
            self.armed_by_gui = True
            self.event("START")
        return ok, msg

    def do_walk(self) -> tuple:
        ok, msg = self.call("cmd_walk")
        self.event("WALK" if ok else f"WALK refused: {msg}")
        return ok, msg

    def do_zero(self) -> tuple:
        ok, msg = self.call("zero_ankles")
        self.event("ankles zeroed" if ok else f"zero ankles failed: {msg}")
        return ok, msg

    # ---- the picture ---------------------------------------------------------

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            s, st_age = self.state, now - self.state_t
            win = list(self.window)
            rs = list(self.rate_seq)
            cmd, cmd_age = self.commander, now - self.commander_t
            run, last_run = self.run, self.last_run
            events = list(self.events)
        out = dict(url=self.url, events=events, last_run=last_run,
                   deadman=dict(armed_by_gui=self.armed_by_gui,
                                heartbeat_age_ms=round(1000 * (now - self.last_hb))
                                if self.armed_by_gui else None,
                                limit_ms=int(1000 * HEARTBEAT_LOSS_S)))
        out["commander"] = dict(cmd or {}, age_s=round(cmd_age, 1)) if cmd else None
        if s is None or st_age > 1.0:
            out["link"] = dict(connected=False, age_s=None if s is None else round(st_age, 1))
            return out

        rate = 0.0
        if len(rs) >= 2 and rs[-1][0] > rs[0][0]:
            rate = (rs[-1][1] - rs[0][1]) / (rs[-1][0] - rs[0][0])
        n = max(1, len(win))
        fresh = [sum(f[j] & 1 for f, _ in win) / n for j in range(len(JOINT_NAMES))]
        enc = [sum((e >> i) & 1 for _, e in win) / n for i in range(len(SPRING_NAMES))]
        health = [name for b, name in HEALTH_NAMES.items() if s.health & b]

        out["link"] = dict(connected=True, rate_hz=round(rate, 1), age_ms=round(1000 * st_age))
        soft = "RAMP" if s.stream_flags & STREAM_RAMP else "HOLD" if s.stream_flags & STREAM_HOLD else ""
        out["board"] = dict(
            safety=SAFETY_NAMES.get(s.safety_state, "?"), soft=soft,
            ankles_zeroed=bool(s.stream_flags & STREAM_ANKLES_ZEROED),
            arm_blocked=bool(s.stream_flags & STREAM_ARM_BLOCKED),
            health=health, loop_us_max=int(s.loop_us_max), overruns=int(s.overruns),
            phase=round(float(s.phase), 3))
        out["joints"] = [dict(
            name=name, pos=_deg(s.joint_pos[j]), target=_deg(s.act_target[j]),
            ref=_deg(s.ref_angle[j]),
            err=_deg(float(s.act_target[j]) - float(s.joint_pos[j])),
            fresh=round(100 * fresh[j]), state=axis_state_name(s.act_state[j]),
            error=error_names(s.act_error[j])) for j, name in enumerate(JOINT_NAMES)]
        out["springs"] = [dict(name=name, deg=_deg(s.spring_angle[i]), valid=round(100 * enc[i]))
                          for i, name in enumerate(SPRING_NAMES)]
        acc = math.sqrt(sum(float(v) ** 2 for v in s.imu_accel))
        out["estimator"] = dict(
            fused_valid=int(s.fused_valid), ok=s.fused_valid == FUSION_OK,
            pelvis_z=round(float(s.pelvis_z), 3) if math.isfinite(s.pelvis_z) else None,
            vel=[round(float(v), 3) if math.isfinite(v) else None for v in s.vel_hdg],
            tilt=_tilt_deg(s.quat),
            foot_z=[round(float(v), 3) if math.isfinite(v) else None for v in s.foot_z],
            contacts=[bool(c > 0.5) for c in s.contact], accel=round(acc, 2),
            gyro=round(max(abs(float(v)) for v in s.imu_gyro), 3))
        if run is not None:
            out["run"] = dict(duration_s=round(now - run["t0"], 1))

        # Pre-flight, the same rules as `zeus check`.
        checks = []

        def chk(ok, good, bad, warn=False):
            checks.append(dict(ok=bool(ok), text=good if ok else bad,
                               level="ok" if ok else ("warn" if warn else "fail")))

        chk(rate >= 900, f"board ticking at {rate:.0f} Hz", f"board at only {rate:.0f} Hz")
        bad = [f"{JOINT_NAMES[j]}: {', '.join(error_names(s.act_error[j])) or 'not reporting'}"
               for j in range(len(JOINT_NAMES)) if s.act_error[j] or fresh[j] < 0.5]
        chk(not bad, "all 8 drives reporting, no errors", "; ".join(bad))
        # A drive can report, error-free, a position that is not a number - the
        # ankles after a reboot, and the left knee at times. The board will not
        # arm with one; say which, before START is pressed.
        nan = [JOINT_NAMES[j] for j in range(len(JOINT_NAMES))
               if not math.isfinite(float(s.joint_pos[j]))]
        chk(not nan, "every joint has a position",
            "no position (NaN): " + ", ".join(nan) + " - the board will not arm")
        why = ankle_zero_problems(s.joint_pos, [1 if f >= 0.5 else 0 for f in fresh])
        chk(not why, f"ankles at zero (within {ANKLE_ZERO_TOL_DEG:g} deg) - feet flat",
            "; ".join(why))
        chk(abs(acc - GRAVITY) < 0.5, f"IMU reads {acc:.2f} m/s^2",
            f"IMU reads {acc:.2f} m/s^2 (expected ~9.81 standing still)", warn=True)
        low = [f"{SPRING_NAMES[i]} {100 * v:.0f}%" for i, v in enumerate(enc) if v < 0.95]
        chk(not low, "spring encoders all reading", "spring encoders: " + ", ".join(low), warn=True)
        bad_health = [h for h in health if h != "pi-link" or (cmd and cmd.get("enabled"))]
        chk(not bad_health, "no board health faults", "health: " + ", ".join(bad_health))
        chk(s.fused_valid == FUSION_OK, "estimator OK",
            "estimator not OK - the policy will not arm until it is", warn=True)
        out["checks"] = checks
        return out


def make_handler(node: GuiNode):

    class Handler(BaseHTTPRequestHandler):
        # Keep-alive: a browser sends every heartbeat down one connection
        # instead of opening a new one five times a second. With HTTP/1.0 each
        # heartbeat was a fresh connection, the accept queue overflowed while
        # the server waited its turn, and the kernel's retries stretched single
        # heartbeats to 5 s - long enough to trip the dead-man on an open page.
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def _authorised(self) -> bool:
            q = parse_qs(urlparse(self.path).query)
            got = self.headers.get("X-Zeus-Key") or (q.get("key") or [""])[0]
            return secrets.compare_digest(got, node.key)

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if not self._authorised():
                self._json(403, dict(error="wrong or missing key: open the address the "
                                           "GUI printed in the launch log"))
                return
            if path == "/":
                data = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/state":
                self._json(200, node.snapshot())
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while rclpy.ok():
                        self.wfile.write(b"data: " + json.dumps(node.snapshot()).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self._json(404, dict(error="not found"))

        def do_POST(self):
            path = urlparse(self.path).path
            if not self._authorised():
                self._json(403, dict(error="wrong or missing key"))
                return
            if path == "/api/heartbeat":
                node.last_hb = time.monotonic()
                self._json(200, dict(ok=True))
                return
            act = {"/api/start": node.do_start, "/api/walk": node.do_walk,
                   "/api/stop": node.do_stop, "/api/zero_ankles": node.do_zero}.get(path)
            if act is None:
                self._json(404, dict(error="not found"))
                return
            ok, msg = act()
            self._json(200, dict(success=ok, message=msg))

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64        # the default 5 overflowed: see Handler


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GuiNode()
    try:
        server = Server(("0.0.0.0", node.port), make_handler(node))
    except OSError as exc:
        node.get_logger().fatal(f"GUI cannot open port {node.port}: {exc}. Another GUI or "
                                "launch still running? Stop it, or pass gui_port:=8081.")
        node.destroy_node()
        rclpy.try_shutdown()
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # One executor thread is enough at 50 Hz, and every extra one competes with
    # the web server for the interpreter. Service calls are still answered: the
    # HTTP threads wait on futures this thread completes.
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Ctrl-C can surface from inside an rclpy read as a RuntimeError while
        # the context is being torn down (Jazzy). After shutdown has begun
        # it is noise; while ROS is still running it is a real error.
        if rclpy.ok():
            raise
    finally:
        server.shutdown()
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zeus Control</title>
<style>
:root{--bg:#0f1214;--panel:#171b1e;--line:#262c30;--ink:#e6e9ea;--dim:#8b949a;
--ok:#3fb27f;--warn:#e0a83a;--fail:#e5534b;--acc:#4ea1d3;--hold:#9b7be0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:12px 16px;
border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:16px;margin:0;letter-spacing:.04em}
.pill{padding:3px 10px;border-radius:99px;font-weight:600;font-size:13px;background:var(--line)}
.ARMED{background:var(--fail);color:#fff}.IDLE{background:var(--ok);color:#06150e}
.FAULT{background:var(--warn);color:#1b1300}.BOOT{background:var(--line)}
.RAMP{background:var(--acc);color:#04121b}.HOLD{background:var(--hold);color:#120a24}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;padding:12px 16px}
@media (max-width:900px){main{grid-template-columns:minmax(0,1fr)}}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 8px}
.controls{display:flex;gap:10px;flex-wrap:wrap}
button{font:600 15px system-ui,sans-serif;padding:12px 18px;border-radius:8px;border:0;cursor:pointer}
button:disabled{opacity:.35;cursor:not-allowed}button:focus-visible{outline:3px solid var(--acc)}
#start{background:var(--acc);color:#04121b}#walk{background:var(--ok);color:#06150e}
#zero{background:var(--line);color:var(--ink)}
#stop{background:var(--fail);color:#fff;font-size:20px;padding:16px 36px;margin-left:auto}
.mode{font-size:22px;font-weight:700;margin:8px 0 2px}.dim{color:var(--dim)}
ul.checks{list-style:none;margin:0;padding:0}ul.checks li{padding:3px 0}
.ok::before{content:"✓ ";color:var(--ok)}.warn::before{content:"! ";color:var(--warn)}
.fail::before{content:"✗ ";color:var(--fail)}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--dim);font-weight:500}
.bad{color:var(--fail)}.meh{color:var(--warn)}
.banner{margin:8px 0;padding:8px 10px;border-radius:6px;background:#3a1d1b;color:#ffd7d4}
.blockers{margin:6px 0 0;padding-left:18px;color:var(--warn)}
.full{grid-column:1/-1}
#events{max-height:180px;overflow:auto;font:12px ui-monospace,Menlo,monospace}
</style></head><body>
<header>
  <h1>ZEUS</h1>
  <span id="safety" class="pill">—</span><span id="soft" class="pill" hidden></span>
  <span id="link" class="dim">connecting…</span>
  <span id="deadman" class="dim"></span>
</header>
<main>
<section class="full">
  <div class="controls">
    <button id="zero">ZERO ANKLES</button>
    <button id="start">START</button>
    <button id="walk">WALK</button>
    <button id="stop" title="space bar">STOP</button>
  </div>
  <div class="mode" id="mode">—</div>
  <div class="dim" id="modeinfo"></div>
  <ul class="blockers" id="blockers"></ul>
  <div class="banner" id="banner" hidden></div>
</section>
<section><h2>Pre-flight</h2><ul class="checks" id="checks"></ul></section>
<section><h2>Estimator &amp; feet</h2><div id="est"></div></section>
<section class="full"><h2>Joints (deg)</h2><div class="tablewrap"><table id="joints"></table></div></section>
<section><h2>Springs</h2><div class="tablewrap"><table id="springs"></table></div></section>
<section><h2>Last run</h2><div id="run" class="dim">no run yet</div></section>
<section class="full"><h2>Events</h2><div id="events"></div></section>
</main>
<script>
const key = new URLSearchParams(location.search).get("key") || "";
const H = {"X-Zeus-Key": key};
const $ = id => document.getElementById(id);
let S = null, armedHere = false;
const f = (v, d=2) => v === null || v === undefined ? "—" : Number(v).toFixed(d);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

async function act(path, confirmText){
  if (confirmText && !confirm(confirmText)) return;
  try{
    const r = await fetch(path, {method:"POST", headers:H});
    const j = await r.json();
    if (!j.success) alert(j.message || j.error || "refused");
  }catch(e){ alert("no answer from the robot: " + e); }
}
$("zero").onclick = () => act("/api/zero_ankles",
  "ZERO ANKLES: both feet must be FLAT right now. Zero them?");
$("start").onclick = () => act("/api/start");
$("walk").onclick = () => act("/api/walk", S && S.commander && S.commander.dry_run ?
  "DRY RUN: this releases the STORED GAIT with no policy - it falls within a second in " +
  "simulation. Release anyway?" : null);
$("stop").onclick = () => act("/api/stop");
document.addEventListener("keydown", e => { if (e.code === "Space"){ e.preventDefault(); act("/api/stop"); }});

setInterval(() => { fetch("/api/heartbeat", {method:"POST", headers:H}).catch(()=>{}); }, 200);

const MODES = {handshake:"Connecting to the board", watching:"Ready to START",
  ankle_check:"Ankle pre-check failing", blocked:"Cannot arm yet", arming:"Arming the drives",
  arm_refused:"Board refuses to arm", ramping:"Ramping to the start pose",
  holding:"Holding the start pose", ready:"Holding — READY TO WALK", walking:"WALKING",
  running:"Running", stopped:"Stopped", fallen:"FALLEN — press STOP to acknowledge",
  no_state:"No state from the board"};

function render(){
  if (!S) return;
  const L = S.link || {};
  $("link").textContent = L.connected ? `link ${f(L.rate_hz,0)} Hz` : "NO LINK TO THE BOARD";
  $("link").className = L.connected ? "dim" : "bad";
  const B = S.board || {};
  $("safety").textContent = B.safety || "—"; $("safety").className = "pill " + (B.safety||"");
  $("soft").hidden = !B.soft; $("soft").textContent = B.soft; $("soft").className = "pill " + B.soft;
  const D = S.deadman || {};
  $("deadman").textContent = D.armed_by_gui ? `dead-man armed (${D.heartbeat_age_ms} ms / ${D.limit_ms})` : "";

  const C = S.commander;
  const mode = C ? C.mode : null;
  $("mode").textContent = C ? (MODES[mode] || mode) : "No commander running";
  $("modeinfo").textContent = C ? `${C.node} · ${C.policy}` + (C.dry_run ? " · DRY RUN" : "") +
     (C.residual_peak_deg ? ` · residual peak ${f(C.residual_peak_deg,1)}°` : "") +
     (C.stop_reason ? ` · last stop: ${C.stop_reason}` : "") : "start walk.launch.py";
  $("blockers").innerHTML = C && C.blockers ? C.blockers.map(b => `<li>${esc(b)}</li>`).join("") : "";
  const ban = [];
  if (C && C.dry_run) ban.push("DRY RUN: the policy computes but sends zero. A released dry run is the stored gait alone.");
  if (C && C.fall) ban.push("FALL: " + C.fall);
  if (B.arm_blocked) ban.push("The board refuses to arm: a joint has no position.");
  $("banner").hidden = !ban.length; $("banner").textContent = ban.join(" ");

  const armed = B.safety === "ARMED";
  $("zero").disabled = !L.connected || armed;
  $("start").disabled = !L.connected || !C || C.enabled || mode === "fallen";
  $("walk").disabled = !C || !C.can_walk || mode !== "ready";

  $("checks").innerHTML = (S.checks||[]).map(c => `<li class="${c.level}">${esc(c.text)}</li>`).join("");

  const E = S.estimator || {};
  $("est").innerHTML = L.connected ? `
    <div class="${E.ok ? "" : "meh"}">estimator ${["INVALID","CONVERGING","OK"][E.fused_valid] || E.fused_valid}</div>
    <div>height ${f(E.pelvis_z,3)} m · tilt ${f(E.tilt,1)}°</div>
    <div>velocity lat/fwd/up ${(E.vel||[]).map(v=>f(v,3)).join(" / ")} m/s</div>
    <div>feet: left ${E.contacts&&E.contacts[0]?"DOWN":"up"}, right ${E.contacts&&E.contacts[1]?"DOWN":"up"} ·
         foot z R/L ${(E.foot_z||[]).map(v=>f(v,3)).join(" / ")} m</div>
    <div class="dim">IMU |a| ${f(E.accel,2)} m/s² · |gyro| ${f(E.gyro,3)} rad/s</div>` : "";

  const J = S.joints || [];
  $("joints").innerHTML = "<tr><th>joint</th><th>actual</th><th>commanded</th><th>gait ref</th><th>error</th><th>fresh</th><th>drive</th></tr>" +
    J.map(j => `<tr><td>${j.name}</td><td>${f(j.pos)}</td><td>${f(j.target)}</td><td>${f(j.ref)}</td>
      <td class="${j.err!==null && Math.abs(j.err)>5?"bad":j.err!==null && Math.abs(j.err)>2?"meh":""}">${f(j.err)}</td>
      <td class="${j.fresh<50?"bad":""}">${j.fresh}%</td>
      <td class="${j.error.length?"bad":""}">${esc(j.state)}${j.error.length?" · "+esc(j.error.join(", ")):""}</td></tr>`).join("");
  $("springs").innerHTML = "<tr><th>spring</th><th>deflection</th><th>valid</th></tr>" +
    (S.springs||[]).map(s => `<tr><td>${s.name}</td><td>${f(s.deg,3)}</td><td class="${s.valid<95?"meh":""}">${s.valid}%</td></tr>`).join("");

  const R = S.last_run;
  $("run").innerHTML = R ? `<div>${f(R.duration_s,1)} s armed · reached ${esc(R.reached.join(", ")||"—")} · max tilt ${f(R.max_tilt_deg,1)}°</div>
    <div>stopped because: ${esc(R.stopped_because)}</div>
    <div class="tablewrap"><table>${Object.entries(R.max_err_deg).map(([k,v]) =>
      `<tr><td>${k}</td><td class="${v>5?"bad":v>2?"meh":""}">max error ${f(v)}°</td></tr>`).join("")}</table></div>
    <div class="dim">recording: ${esc(R.recording||"—")}</div>` : "no run yet";
  $("events").innerHTML = (S.events||[]).map(e => `<div>${esc(e[0])}  ${esc(e[1])}</div>`).join("");
}

function connect(){
  const es = new EventSource("/events?key=" + encodeURIComponent(key));
  es.onmessage = m => { S = JSON.parse(m.data); render(); };
  es.onerror = () => { $("link").textContent = "GUI connection lost - retrying"; $("link").className = "bad"; };
}
connect();
</script></body></html>
"""


if __name__ == "__main__":
    main()
