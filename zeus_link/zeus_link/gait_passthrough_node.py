"""
The simplest possible "policy": a zero residual at 250 Hz.

With enable false (the default) the board keeps receiving commands, so the link
stays healthy, but stays IDLE - nothing moves. That is the state to watch
sensors in.

With enable true the STM32 arms the drives and plays its stored gait with no
correction at all. That is the first thing to run on the full robot, and the
baseline any RL policy has to beat.

    ros2 run zeus_link gait_passthrough_node                      # watch only
    ros2 run zeus_link gait_passthrough_node --ros-args -p enable:=true
                                    # arm, ramp to the start pose, and STAND there
    ros2 run zeus_link gait_passthrough_node --ros-args -p enable:=true -p hold:=false
                                    # arm, ramp into the running gait, and walk it

hold (default true) is the soft start's HOLD flag (protocol v11): the board
freezes its gait clock at the start phase, ramps every joint over 3 s from where
it stands to that phase's pose, and holds it. With hold false the arm still
ramps - into the stride in progress - but then the gait plays. Arming can never
jump straight to the gait any more; see stm32_zeuss Appli/App/soft_start.h.

On start it sends enable false until the handshake is CONFIRMED: the command
publisher is matched to the link node, and the board has reported IDLE or ARMED
for HANDSHAKE_SEC. The STM32 latches every fault and will not arm again until it
has seen a command with enable false, so a node restarted after a fault with
enable:=true would otherwise be ignored indefinitely with nothing to say why.

It is closed-loop, not a timer, on purpose. The first version sent enable false
for a fixed 0.2 s from startup - but DDS discovery takes longer than that, so
every handshake frame could be published before the link node was listening and
silently dropped, leaving the board latched in FAULT. Tested and seen against
the fake board.

It is also the template for a real policy: subscribe zeus/state with STATE_QOS,
publish zeus/command with COMMAND_QOS, do the handshake, then fill residual_rad
and enable.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from zeus_msgs.msg import NexusCommand, NexusState

from .nexus_proto import (HEALTH_NAMES, NUM_JOINTS, SAFETY_ARMED, SAFETY_IDLE,
                          SAFETY_NAMES, STREAM_ARM_BLOCKED, STREAM_HOLD, STREAM_RAMP,
                          STREAM_RETURN)
from .precheck import return_refusal
from .qos import COMMAND_QOS, STATE_QOS

HANDSHAKE_SEC = 0.2



class GaitPassthroughNode(Node):

    def __init__(self) -> None:
        super().__init__("gait_passthrough")

        rate = float(self.declare_parameter("rate_hz", 250.0).value)
        self._enable = bool(self.declare_parameter("enable", False).value)
        self._hold = bool(self.declare_parameter("hold", True).value)
        self._pos = None
        self._act_flags = None

        self._pub = self.create_publisher(NexusCommand, "zeus/command", COMMAND_QOS)
        self.create_subscription(NexusState, "zeus/state", self._on_state, STATE_QOS)

        self._cmd = NexusCommand()
        self._cmd.residual_rad = [0.0] * NUM_JOINTS
        self._cmd.enable = False
        self._cmd.hold = self._hold
        self._safety = None
        self._soft = None
        self._health = 0
        self._handshake_ticks = max(1, int(round(HANDSHAKE_SEC * rate)))
        self._confirmed = 0
        self._handshake_done = False

        self._stop_reason = None
        self._status_pub = self.create_publisher(String, "zeus/commander_status", 10)
        # The same operator interface as rl_policy_node, so the GUI drives
        # either. START arms into the hold; WALK is refused here - the stored
        # gait alone falls within a second (sim_check/RESULTS.md: 0.7 s).
        self.create_service(Trigger, "zeus/commander/start", self._on_start)
        self.create_service(Trigger, "zeus/commander/walk", self._on_walk)
        self.create_service(Trigger, "zeus/commander/stop", self._on_stop)
        # Back to standing (protocol v12 RETURN): from the ramp or the hold,
        # ramp back to the pose the robot armed from and stand there.
        self.create_service(Trigger, "zeus/commander/return", self._on_return)
        self.create_timer(0.1, self._publish_status)
        self._flags = 0

        self.create_timer(1.0 / rate, self._tick)

        if self._enable and self._hold:
            self.get_logger().warn(
                f"ENABLED with HOLD - zero residual at {rate:g} Hz. The robot will arm, "
                "ramp to the gait's start pose over 3 s, and stand there.")
        elif self._enable:
            self.get_logger().warn(
                f"ENABLED - zero residual at {rate:g} Hz. The robot will arm, ramp "
                "into the running gait over 3 s, and walk it.")
        else:
            self.get_logger().info(
                f"zero residual at {rate:g} Hz with enable false: the board stays "
                "IDLE. Set -p enable:=true to walk the stored gait.")

    def _tick(self) -> None:
        if not self._handshake_done:
            self._handshake()
        elif not self._enable:
            self._cmd.enable = False
        else:
            # No ankle-zero pre-check any more (removed 2026-11-23): the robot
            # is not always upright with its feet flat, so "both ankles read
            # zero" was a condition the builder often could not meet. The
            # ankles are zeroed when the link node starts and whenever ZERO
            # ANKLES is pressed.
            self._cmd.enable = True
        self._cmd.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._cmd)

    def _handshake(self) -> None:
        self._cmd.enable = False
        listening = self._pub.get_subscription_count() > 0
        ready = self._safety in (SAFETY_IDLE, SAFETY_ARMED)

        self._confirmed = self._confirmed + 1 if (listening and ready) else 0
        if self._confirmed >= self._handshake_ticks:
            self._handshake_done = True
            self._cmd.enable = self._enable
            self.get_logger().info(
                f"handshake confirmed, board IDLE; enable -> {str(self._enable).lower()}")
            return

        if not listening:
            why = "no link node is subscribed to zeus/command yet"
        elif self._safety is None:
            why = "no zeus/state received yet"
        else:
            faults = [n for bit, n in HEALTH_NAMES.items() if self._health & bit]
            why = (f"board is {SAFETY_NAMES.get(self._safety, self._safety)}"
                   + (f", health faults: {', '.join(faults)}" if faults else ""))
        self.get_logger().info(f"sending enable false until the board is ready: {why}",
                               throttle_duration_sec=2.0)

    def _on_start(self, _req, resp):
        self._cmd.return_home = False
        self._enable = True
        self._stop_reason = None
        self.get_logger().warn("START: arming into the "
                               + ("HOLD" if self._hold else "running gait"))
        resp.success = True
        resp.message = ("arming into the hold - the robot ramps to the start pose and stands"
                        if self._hold else "arming into the running gait")
        return resp

    def _on_walk(self, _req, resp):
        resp.success = False
        resp.message = ("the stand test does not walk: the stored gait without the policy falls "
                        "within a second in simulation. Use policy:=rl to walk.")
        return resp

    def _on_return(self, _req, resp):
        resp.success, resp.message = return_refusal(self._enable, self._safety, self._flags)
        if resp.success:
            self._cmd.return_home = True
            self.get_logger().warn("BACK TO STANDING: ramping every joint to zero")
        return resp

    def _on_stop(self, _req, resp):
        self._cmd.return_home = False
        self._enable = False
        self._stop_reason = "operator STOP"
        self.stand_down()
        self._handshake_done = True
        self.get_logger().warn("STOP: enable off, the board idles the drives")
        resp.success, resp.message = True, "stopped: drives idle"
        return resp

    def _publish_status(self) -> None:
        if not self._handshake_done:
            mode, blockers = "handshake", []
        elif not self._enable:
            mode, blockers = "watching", []
        elif self._flags & STREAM_ARM_BLOCKED:
            mode, blockers = "arm_refused", ["the board refuses to arm: a joint has no position"]
        elif self._flags & STREAM_RETURN:
            mode, blockers = ("returning" if self._flags & STREAM_RAMP else "home"), []
        elif self._flags & STREAM_RAMP:
            mode, blockers = "ramping", []
        elif self._flags & STREAM_HOLD:
            mode, blockers = "holding", []
        elif self._safety == SAFETY_ARMED:
            mode, blockers = "running", []
        else:
            mode, blockers = "arming", []
        msg = String()
        msg.data = json.dumps({
            "node": "gait_passthrough", "mode": mode, "blockers": blockers,
            "enabled": bool(self._enable), "release": False, "dry_run": False,
            "can_walk": False, "residual_peak_deg": 0.0, "fall": None,
            "stop_reason": self._stop_reason, "policy": "stored gait (stand test)",
        })
        self._status_pub.publish(msg)

    def _on_state(self, msg: NexusState) -> None:
        self._health = msg.health
        self._pos = list(msg.joint_pos)
        self._act_flags = list(msg.act_flags)
        self._flags = msg.stream_flags

        soft = ("going back to the standing pose"
                if msg.stream_flags & STREAM_RETURN and msg.stream_flags & STREAM_RAMP else
                "back at the standing pose - STOP to idle the drives"
                if msg.stream_flags & STREAM_RETURN else
                "ramping to the start pose" if msg.stream_flags & STREAM_RAMP else
                "holding the start pose" if msg.stream_flags & STREAM_HOLD else
                "arm refused: a joint has no position (an ankle not zeroed?)"
                if msg.stream_flags & STREAM_ARM_BLOCKED else None)
        if soft != self._soft:
            if soft is not None:
                self.get_logger().info(f"board: {soft}")
            self._soft = soft
        if msg.safety_state != self._safety:
            self._safety = msg.safety_state
            name = SAFETY_NAMES.get(msg.safety_state, str(msg.safety_state))
            self.get_logger().info(f"board safety state: {name}")
            if name == "FAULT" and self._enable and self._handshake_done:
                # Only a fault AFTER our handshake: a node just started sees the
                # board's leftover fault from whatever ran before it, and its
                # own enable-false handshake is what clears that one.
                # A faulted board ignores enable:true until it has seen
                # enable:false. Carrying on sending true left the GUI showing
                # "arming" forever and looking like START did nothing. Drop
                # back to watching, say why, and let the next START re-arm.
                faults = [n for bit, n in HEALTH_NAMES.items() if self._health & bit]
                self._enable = False
                self._cmd.return_home = False
                self._stop_reason = ("board fault"
                                     + (f" ({', '.join(faults)})" if faults else ""))
                self.get_logger().warn(
                    f"the board faulted and has idled the drives: {self._stop_reason}. "
                    "Press START again to re-arm.")

    def stand_down(self) -> None:
        self._handshake_done = True
        self._cmd.enable = False
        self._cmd.header.stamp = self.get_clock().now().to_msg()
        for _ in range(3):
            self._pub.publish(self._cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GaitPassthroughNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Ctrl-C can surface from inside an rclpy read as a RuntimeError while
        # the context is being torn down (Jazzy). After shutdown has begun
        # it is noise; while ROS is still running it is a real error.
        if rclpy.ok():
            raise
    finally:
        # Only possible if the context is still up. On ctrl-c it is not, and
        # the robot is idled another way: commands stop, and 200 ms later the
        # STM32's link fault takes the drives - or sooner, if the link node is
        # shut down too, since it sends enable false on its way out.
        if rclpy.ok():
            try:
                node.stand_down()
            except Exception:   # noqa: BLE001
                # The context can go invalid between ok() and publish() as
                # ctrl-c lands. The link node's own enable:false on exit, or
                # the board's 200 ms link fault, still idles the drives.
                pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
