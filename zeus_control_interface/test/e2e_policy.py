"""
End-to-end: rl_policy_node and the real link node against fake_board, over a pty.

Needs a built, sourced workspace, like zeus_link/test/e2e_fake_board.py:

    source install/setup.bash && python3 zeus_control_interface/test/e2e_policy.py

  1. enable false   the board never arms
  2. enable true, default dry_run    arms, and says it is sending zero
  3. enable true, dry_run false      arms, residuals move, none rejected
  4. enable true, release false     arms into the soft-start hold, never walks
  5. a lost right-ankle zero          zeroed by the link node on start, then arms
  6. wrong gait (fake_board's default stride)  the guards never let it arm

Every arm goes through the v11 soft start: arm into HOLD, ramp to the start
pose, then (unless release:=false) release into the gait.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory

PKG = "zeus_control_interface"
LINK = os.path.join(get_package_prefix("zeus_link"), "lib", "zeus_link")
NODE = os.path.join(get_package_prefix(PKG), "lib", PKG, "rl_policy_node")
GAIT = os.path.join(get_package_share_directory(PKG), "policy", "z1_softland_tol05", "policy.npz")
failures = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        failures.append(what)


tmp = tempfile.mkdtemp(prefix="zeus_policy_e2e_")
ENV = dict(os.environ, ROS_DOMAIN_ID=str(100 + os.getpid() % 50),
           ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", PYTHONUNBUFFERED="1")


def scenario(name, seconds, *params, gait=True, board_extra=()):
    link, summary = os.path.join(tmp, name), os.path.join(tmp, name + ".json")
    log = open(os.path.join(tmp, name + ".log"), "w+")
    board_args = ["--link", link, "--summary", summary, "--duration", str(seconds + 8),
                  *board_extra]
    board = subprocess.Popen([os.path.join(LINK, "fake_board"), *board_args,
                              *(["--gait", GAIT] if gait else [])],
                             stdout=log, stderr=subprocess.STDOUT, env=ENV)
    for _ in range(50):
        if os.path.exists(link):
            break
        time.sleep(0.1)
    ln = subprocess.Popen([os.path.join(LINK, "link_node"), "--ros-args", "-p", f"port:={link}"],
                          stdout=log, stderr=subprocess.STDOUT, env=ENV)
    args = [x for p in params for x in ("-p", p)]
    pol = subprocess.Popen([NODE, "--ros-args", *args], stdout=log, stderr=subprocess.STDOUT, env=ENV)

    time.sleep(seconds)
    for p in (pol, ln):
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    board.send_signal(signal.SIGTERM)
    board.wait(timeout=10)
    log.seek(0)
    return json.load(open(summary)), log.read()


print("enable false:", flush=True)
s, log = scenario("idle", 8, "enable:=false")
check(s["enables"] == 0 and s["commands"] > 500, f"{s['commands']} commands, none enabled")
check(not [e for e in s["events"] if e[2] == "ARMED"], "board never armed")

print("enable true, dry run:", flush=True)
s, log = scenario("dry", 9, "enable:=true")
check(any(e[2] == "ARMED" for e in s["events"]), "board armed")
check("guards clear" in log, "guards cleared")
check("DRY RUN" in log, "node says it is sending zero")
check("Traceback" not in log, "no exception")

print("enable true, live:", flush=True)
s, log = scenario("live", 9, "enable:=true", "dry_run:=false")
check(any(e[2] == "ARMED" for e in s["events"]), "board armed")
check(s["rejected"] == 0 and s["crc_bad"] == 0, f"{s['enables']} enabled, none rejected")
check("Traceback" not in log, "no exception")
peak = [l for l in log.splitlines() if "residual peak" in l]
check(bool(peak) and "DRY RUN" not in peak[-1] and "peak 0.0 deg" not in peak[-1],
      "the residual is really going out: " + (peak[-1].split("]: ")[-1] if peak else "none"))
# Warnings are throttled to one per 2 s, so this counts bad windows. One is a
# loaded machine; a run of them is inference not fitting in the tick.
late = log.count("tick late")
check(late <= 1, f"250 Hz timer kept up with inference ({late} late windows)")

print("enable true, release false - the first thing to run on the robot:", flush=True)
s, log = scenario("hold", 9, "enable:=true", "release:=false")
check(any(e[2] == "ARMED" and "HOLD" in e[3] for e in s["events"]), "board armed INTO A HOLD")
check("HOLDING the start pose" in log, "node reports it is holding the start pose")
check("released at phase" not in log, "and never released into the gait")
check("Traceback" not in log, "no exception")

print("right ankle reads 6.5 deg at launch: the link node zeroes it, then it arms:", flush=True)
s, log = scenario("ankle", 9, "enable:=true", "release:=false",
                  board_extra=("--ankle-offset", "6.5"))
check(s.get("ankle_zeroings", 0) >= 1, "the link node zeroed the ankles on start")
check(any(e[2] == "ARMED" and "HOLD" in e[3] for e in s["events"]),
      "and the robot then armed into the hold from the new zero")

print("wrong stored gait:", flush=True)
s, log = scenario("wrong", 8, "enable:=true", "dry_run:=false", gait=False)
check(not [e for e in s["events"] if e[2] == "ARMED"], "guards kept the board from arming")
check("not arming" in log, "and the node said why")

print(f"\nlogs in {tmp}")
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("POLICY END-TO-END PASSED")
