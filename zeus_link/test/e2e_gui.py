"""
End-to-end: the web GUI driving the real link node and commander against
fake_board, over HTTP exactly as a browser does.

    source install/setup.bash && python3 zeus_link/test/e2e_gui.py

What a failure would mean on the robot: a button that does not do what it
says, a STOP that does not stop, a page that arms the robot for anyone on the
network, or a closed laptop that leaves the robot walking.
"""

import json
import re
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from ament_index_python.packages import get_package_prefix, get_package_share_directory

LINK = os.path.join(get_package_prefix("zeus_link"), "lib", "zeus_link")
RL = os.path.join(get_package_prefix("zeus_control_interface"), "lib",
                  "zeus_control_interface", "rl_policy_node")
GAIT = os.path.join(get_package_share_directory("zeus_control_interface"),
                    "policy", "z1_softland_tol05", "policy.npz")
KEY = "e2e-test-key"
failures = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        failures.append(what)


tmp = tempfile.mkdtemp(prefix="zeus_gui_e2e_")
ENV = dict(os.environ, ROS_DOMAIN_ID=str(150 + os.getpid() % 50),
           ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", PYTHONUNBUFFERED="1")


def http(port, method, path, key=KEY):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 headers={"X-Zeus-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def state(port):
    return http(port, "GET", "/api/state")[1]


def wait_for(port, pred, secs):
    deadline = time.monotonic() + secs
    s = None
    while time.monotonic() < deadline:
        try:
            s = state(port)
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.2)     # the GUI's server is still coming up
            continue
        if pred(s):
            return True, s
        time.sleep(0.1)
    return False, s or {}


def mode(s):
    return (s.get("commander") or {}).get("mode")


def safety(s):
    return (s.get("board") or {}).get("safety")


class Heartbeat:
    """What the open page does: a POST every 200 ms."""

    def __init__(self, port):
        self.port, self.on = port, True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.on:
            try:
                http(self.port, "POST", "/api/heartbeat")
            except Exception:
                pass
            time.sleep(0.2)


def bring_up(name, commander, port):
    link = os.path.join(tmp, name)
    log = open(os.path.join(tmp, name + ".log"), "w+")
    procs = [subprocess.Popen([os.path.join(LINK, "fake_board"), "--link", link, "--gait", GAIT,
                               "--duration", "90"], stdout=log, stderr=subprocess.STDOUT, env=ENV)]
    for _ in range(50):
        if os.path.exists(link):
            break
        time.sleep(0.1)
    procs.append(subprocess.Popen([os.path.join(LINK, "link_node"), "--ros-args",
                                   "-p", f"port:={link}", "-p", "record:=false"],
                                  stdout=log, stderr=subprocess.STDOUT, env=ENV))
    procs.append(subprocess.Popen(commander, stdout=log, stderr=subprocess.STDOUT, env=ENV))
    procs.append(subprocess.Popen([os.path.join(LINK, "zeus_gui"), "--ros-args",
                                   "-p", f"port:={port}", "-p", f"key:={KEY}"],
                                  stdout=log, stderr=subprocess.STDOUT, env=ENV))
    return procs, log


# The test's own SIGINT landing inside a garbage-collector finalizer during
# teardown: Python prints it and carries on ("Exception ignored in"). Not an
# exception in any node's code; anything else still fails the check.
_GC_SIGINT = re.compile(r"Exception ignored in: <function [^>]*__del__[^>]*>\n"
                        r"Traceback \(most recent call last\):\n(?:  .*\n)*?\n?KeyboardInterrupt: ?\n")


def tear_down(procs, log):
    for p in reversed(procs):
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    log.seek(0)
    return _GC_SIGINT.sub("", log.read())


# ---------------------------------------------------------------- the policy
PORT = 18080 + os.getpid() % 500
procs, log = bring_up("policy", [RL, "--ros-args", "-p", "enable:=false",
                                 "-p", "dry_run:=false"], PORT)
try:
    print("the page and its key:", flush=True)
    ok, s = wait_for(PORT, lambda s: (s.get("link") or {}).get("connected")
                     and mode(s) == "watching", 20)
    check(ok, f"connected, commander watching (mode {mode(s)})")
    code, _ = http(PORT, "GET", "/api/state", key="")
    check(code == 403, f"no key: refused ({code})")
    code, _ = http(PORT, "POST", "/api/start", key="wrong")
    check(code == 403 and safety(state(PORT)) != "ARMED", "wrong key: START refused, nothing armed")

    print("START -> ramp -> ready:", flush=True)
    hb = Heartbeat(PORT)
    code, r = http(PORT, "POST", "/api/start")
    check(code == 200 and r["success"], f"START accepted: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: (s.get("board") or {}).get("soft") == "RAMP", 5)
    check(ok, "the board is ramping")
    code, r = http(PORT, "POST", "/api/walk")
    check(not r["success"], f"WALK during the ramp refused: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: mode(s) == "ready", 8)
    check(ok and (s["board"]["soft"] == "HOLD"), f"holding, ready to walk (mode {mode(s)})")

    print("WALK -> STOP:", flush=True)
    code, r = http(PORT, "POST", "/api/walk")
    check(r["success"], f"WALK accepted: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: mode(s) == "walking" and s["board"]["soft"] == "", 3)
    check(ok, "walking: hold released, stride running")
    time.sleep(1.0)
    code, r = http(PORT, "POST", "/api/stop")
    check(r["success"], f"STOP accepted: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: safety(s) != "ARMED", 2)
    check(ok, f"the board disarmed within 2 s ({safety(s)})")
    ok, s = wait_for(PORT, lambda s: s.get("last_run") is not None, 2)
    check(ok and s["last_run"]["duration_s"] > 3 and "hold" in s["last_run"]["reached"],
          "a run summary: duration, reached the hold, per-joint error")
    # STOP idles the board through the link node first; the commander's own
    # status follows within a tick or two.
    ok, s = wait_for(PORT, lambda s: mode(s) in ("watching", "stopped"), 2)
    check(ok, f"commander back to watching ({mode(s)})")
    hb.on = False

    print("BACK TO STANDING from the hold:", flush=True)
    time.sleep(0.5)
    hb = Heartbeat(PORT)
    code, r = http(PORT, "POST", "/api/return")
    check(r["success"], f"accepted while stopped: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: (s.get("board") or {}).get("soft") == "HOME", 6)
    check(ok and safety(s) == "ARMED", f"armed straight into the way back, home ({mode(s)})")
    http(PORT, "POST", "/api/stop")
    wait_for(PORT, lambda s: safety(s) != "ARMED", 2)
    http(PORT, "POST", "/api/start")
    ok, s = wait_for(PORT, lambda s: mode(s) == "ready", 8)
    check(ok, f"at the start pose again ({mode(s)})")
    code, r = http(PORT, "POST", "/api/return")
    check(r["success"], f"BACK TO STANDING accepted: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: (s.get("board") or {}).get("soft") == "RETURN"
                     and mode(s) == "returning", 2)
    check(ok, f"the board is going back ({mode(s)})")
    code, r = http(PORT, "POST", "/api/walk")
    check(not r["success"], "WALK refused on the way back")
    ok, s = wait_for(PORT, lambda s: (s.get("board") or {}).get("soft") == "HOME"
                     and mode(s) == "home", 5)
    check(ok and mode(s) == "home" and safety(s) == "ARMED", f"home, still holding ({mode(s)})")
    worst = max(abs(j["pos"]) for j in s["joints"])
    check(worst < 0.5, f"every joint back at the standing pose (worst {worst:.2f} deg)")
    http(PORT, "POST", "/api/stop")
    ok, s = wait_for(PORT, lambda s: safety(s) != "ARMED", 2)
    check(ok, "STOP idled it")
    hb.on = False

    print("BACK TO STANDING while walking:", flush=True)
    time.sleep(0.5)
    hb = Heartbeat(PORT)
    http(PORT, "POST", "/api/start")
    wait_for(PORT, lambda s: mode(s) == "ready", 8)
    http(PORT, "POST", "/api/walk")
    ok, s = wait_for(PORT, lambda s: mode(s) == "walking", 3)
    code, r = http(PORT, "POST", "/api/return")
    check(ok and r["success"], f"accepted mid-stride: {r.get('message')}")
    ok, s = wait_for(PORT, lambda s: (s.get("board") or {}).get("soft") == "HOME"
                     and mode(s) == "home", 6)
    check(ok and safety(s) == "ARMED", f"walked back to the standing pose ({mode(s)})")
    http(PORT, "POST", "/api/stop")
    wait_for(PORT, lambda s: safety(s) != "ARMED", 2)
    hb.on = False

    print("the dead-man: page goes quiet while armed:", flush=True)
    time.sleep(0.5)
    hb = Heartbeat(PORT)
    code, r = http(PORT, "POST", "/api/start")
    check(r["success"], "START again after a STOP")
    ok, s = wait_for(PORT, lambda s: safety(s) == "ARMED", 5)
    check(ok, "armed again")
    hb.on = False
    t0 = time.monotonic()
    ok, s = wait_for(PORT, lambda s: safety(s) != "ARMED", 4)
    took = time.monotonic() - t0
    check(ok and took < 3.0, f"stopped by itself {took:.1f} s after the last heartbeat")
    check(any("DEAD-MAN" in e[1] for e in s.get("events", [])), "and said why")
finally:
    out = tear_down(procs, log)
check("Traceback" not in out, "no exception in any node")

# ------------------------------------------------------- the stand test
print("the stand test (gait passthrough) never walks:", flush=True)
PORT2 = PORT + 1
procs, log = bring_up("stand", [os.path.join(LINK, "gait_passthrough_node")], PORT2)
try:
    ok, s = wait_for(PORT2, lambda s: mode(s) == "watching", 20)
    check(ok, f"passthrough watching ({mode(s)})")
    hb = Heartbeat(PORT2)
    http(PORT2, "POST", "/api/start")
    ok, s = wait_for(PORT2, lambda s: mode(s) == "holding", 8)
    check(ok, f"armed, ramped, holding the start pose ({mode(s)})")
    code, r = http(PORT2, "POST", "/api/walk")
    check(not r["success"], f"WALK refused: {r.get('message', '')[:60]}...")
    code, r = http(PORT2, "POST", "/api/return")
    check(r["success"], f"BACK TO STANDING accepted: {r.get('message')}")
    ok, s = wait_for(PORT2, lambda s: mode(s) == "home", 6)
    check(ok and safety(s) == "ARMED", f"back at the standing pose, holding ({mode(s)})")
    http(PORT2, "POST", "/api/stop")
    ok, s = wait_for(PORT2, lambda s: safety(s) != "ARMED", 2)
    check(ok, "STOP disarmed it")
    hb.on = False
finally:
    out = tear_down(procs, log)
check("Traceback" not in out, "no exception in any node")

print(f"\nlogs in {tmp}")
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("GUI END-TO-END PASSED")
