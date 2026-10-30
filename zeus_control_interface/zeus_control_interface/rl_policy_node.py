"""
The m1_mukesh policy on the robot.

    ros2 launch zeus_bringup walk.launch.py policy:=rl                        # watch: never arms
    ros2 launch zeus_bringup walk.launch.py policy:=rl enable:=true           # arms, ZERO residual
    ros2 launch zeus_bringup walk.launch.py policy:=rl enable:=true dry_run:=false   # the policy

Two rates, both from training. The command stream runs at 250 Hz, because the
board interpolates over it and faults if it stops. The network runs at 50 Hz on
every fifth tick. Between those the residual passes through the same 4 Hz filter
that sat inside the training environment, so it is part of the plant the policy
learned, not a smoothing nicety: remove it and the policy is driving a robot it
has never met.

Inference is on the timer thread and takes about 2 ms, well inside the 4 ms tick.
If `tick late` warnings appear on the real Pi, move it to its own thread.

Enable is withheld until every Guards check passes, and a fall stands the board
down and LATCHES: a fallen robot is not one to re-arm automatically, so it takes a
restart of this node. dry_run runs everything but sends zero, which is the way to
watch what the policy WOULD do on a stand.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from zeus_link.convert import RESIDUAL_LIMIT_RAD
from zeus_link.nexus_proto import (HEALTH_NAMES, JOINT_NAMES, NUM_JOINTS, SAFETY_ARMED,
                                   SAFETY_IDLE)
from zeus_link.qos import COMMAND_QOS, STATE_QOS
from zeus_msgs.msg import NexusCommand, NexusState

from .obs_adapter import Guards, ObsAdapter, ResidualFilter
from .policy_runtime import Policy

RATE_HZ = 250.0
HANDSHAKE_TICKS = 50        # board must read IDLE/ARMED this long before enabling
STALE_SEC = 0.02            # no new state for 20 ms -> stop enabling


def health_names(mask: int) -> list:
    return [n for bit, n in HEALTH_NAMES.items() if mask & bit]


class RLPolicyNode(Node):

    def __init__(self):
        super().__init__('rl_policy')
        self.enable = bool(self.declare_parameter('enable', False).value)
        self.dry_run = bool(self.declare_parameter('dry_run', True).value)

        self.policy = Policy()
        # Where the IMU really is, in base_link. Default is the training model's,
        # the top of the torso, which is unconfirmed on the robot.
        self.policy.imu_in_base = np.array(self.declare_parameter(
            'imu_in_base', self.policy.imu_in_base.tolist()).value, dtype=np.float64)
        if self.policy.joint_names != tuple(JOINT_NAMES):
            raise RuntimeError(f"policy joints {self.policy.joint_names} are not the "
                               f"protocol's {JOINT_NAMES}: a residual would drive the wrong joint")
        self.adapter = ObsAdapter(self.policy)
        self.filter = ResidualFilter(self.policy, RATE_HZ)
        self.gate = Guards(self.policy)
        self.decim = int(round(RATE_HZ / self.policy.policy_hz))

        self.state = None
        self.state_time = 0.0       # when seq last advanced
        self.confirmed = 0
        self.handshake_done = False
        self.engaged = False
        self.tick = 0
        self.fall = None
        self.last_tick = None
        self.peak_rad = 0.0

        self.create_subscription(NexusState, 'zeus/state', self.on_state, STATE_QOS)
        self.pub = self.create_publisher(NexusCommand, 'zeus/command', COMMAND_QOS)
        self.create_timer(1.0 / RATE_HZ, self.step)

        m = self.policy.manifest
        self.get_logger().info(
            f"{m['run']} at {m['num_timesteps']} steps, {self.policy.gait.speed:.2f} m/s, stride "
            f"{self.policy.gait.period:.3f} s. enable {str(self.enable).lower()}, dry_run "
            f"{str(self.dry_run).lower()}")
        if self.enable and not self.dry_run:
            self.get_logger().warn("LIVE: the policy will command the robot once every guard passes")

    def on_state(self, msg):
        if self.state is None or msg.seq != self.state.seq:
            self.state_time = time.monotonic()
        self.state = msg
        self.gate.watch_clock(msg)

    def disengage(self):
        self.engaged = False
        self.adapter.reset()
        self.filter.reset()

    def step(self):
        now = time.monotonic()
        if self.last_tick is not None and now - self.last_tick > 3.0 / RATE_HZ:
            self.get_logger().warn(f"tick late by {1e3 * (now - self.last_tick):.1f} ms",
                                   throttle_duration_sec=2.0)
        self.last_tick = now

        cmd = NexusCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.residual_rad = [0.0] * NUM_JOINTS
        cmd.enable = False

        s = self.state
        fresh = s is not None and (now - self.state_time) < STALE_SEC

        if not self.handshake_done:
            # enable false until the link node is listening AND the board says
            # IDLE or ARMED - which is what clears a latched fault
            ready = (fresh and self.pub.get_subscription_count() > 0
                     and s.safety_state in (SAFETY_IDLE, SAFETY_ARMED))
            self.confirmed = self.confirmed + 1 if ready else 0
            self.handshake_done = self.confirmed >= HANDSHAKE_TICKS
        elif self.fall is not None or not fresh:
            self.disengage()
        elif not self.enable:
            pass
        elif not self.engaged:
            why = self.gate.engage_blockers(s, health_names)
            if why:
                self.get_logger().info("not engaging: " + "; ".join(why),
                                       throttle_duration_sec=2.0)
            else:
                self.engaged, self.tick = True, 0
                self.get_logger().info(f"guards clear at phase {s.phase:.3f}: engaged")
                self.run_policy(s, cmd)
        else:
            self.run_policy(s, cmd)

        self.pub.publish(cmd)

    def run_policy(self, s, cmd):
        why = self.gate.fallen(s, self.adapter)
        if why is None and self.tick % self.decim == 0:
            frame = self.adapter.frame(s)
            if frame is None:
                why = "state went non-finite"
            else:
                action = self.policy.act(self.adapter.stack(frame))
                self.adapter.a_prev = action.astype(np.float64)
                self.filter.command(action)
        if why is not None:
            self.fall = why
            self.get_logger().error(f"STOP: {why}. Drives idled; restart this node to go again.")
            self.disengage()
            return

        self.tick += 1
        residual = np.clip(self.filter.tick(), -RESIDUAL_LIMIT_RAD, RESIDUAL_LIMIT_RAD)
        self.peak_rad = max(self.peak_rad, float(np.abs(residual).max()))
        cmd.enable = True
        if not self.dry_run:
            cmd.residual_rad = residual.astype(np.float32)
        if s.safety_state == SAFETY_ARMED:
            self.get_logger().info(
                f"{'DRY RUN, sending zero; ' if self.dry_run else ''}"
                f"residual peak {np.degrees(self.peak_rad):.1f} deg, phase {s.phase:.2f}",
                throttle_duration_sec=2.0)

    def stand_down(self):
        cmd = NexusCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.residual_rad = [0.0] * NUM_JOINTS
        cmd.enable = False
        for _ in range(3):
            self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stand_down()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
