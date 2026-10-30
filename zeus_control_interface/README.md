# zeus_control_interface

**Where the RL policy lives.** This package runs `m1_mukesh` (3 M steps) on the
robot. Start with [docs/MUKESH_DEPLOYMENT.md](../docs/MUKESH_DEPLOYMENT.md).

```
zeus_control_interface/
├── zeus_control_interface/
│   ├── rl_policy_node.py       the ROS node: handshake, guards, 250/50 Hz loop
│   ├── obs_adapter.py          board state -> training observation; guards; filter
│   ├── policy_runtime.py       the network in plain numpy, no torch on the Pi
│   └── hardcoded_actuator_test_node.py   from the old ros2_control stack, unused
├── policy/m1_mukesh_3M/        policy.npz, golden.npz, manifest.json, source/
├── scripts/export_policy.py    trained run -> bundle + firmware gait table
├── sim_check/                  deployed code in the loop with MuJoCo, and results
└── test/                       runs on any machine; e2e_policy.py needs ROS
```

## What the policy does

Every 4 ms (250 Hz):

1. read the newest `/zeus/state`
2. every fifth tick, build the observation and run the network (50 Hz)
3. pass the residual through a 4 Hz first-order filter
4. publish `/zeus/command`: a **residual** in radians per joint, and `enable`

The STM32 adds that residual to the gait it is already playing. Without a policy
this gait falls over in 0.7 s in simulation, so the residual is what balances the
robot rather than a small correction.

## The observation

`zeus_link.convert.policy_block(msg)` gives the 50 values of the packet's policy
block as one `np.float32` array, in this order:

| slice | field | size | units |
|---|---|---:|---|
| 0 | `pelvis_z` | 1 | m, **at the IMU** on this board; the adapter moves it to the hips |
| 1–4 | `quat` | 4 | w, x, y, z — body → world |
| 5–7 | `gyro` | 3 | rad/s, body frame |
| 8–10 | `vel_hdg` | 3 | m/s: lateral, forward, vertical, **at the IMU** likewise |
| 11–20 | `joint_pos` | 10 | rad, output side |
| 21–30 | `joint_vel` | 10 | rad/s |
| 31–34 | `spring_angle` | 4 | rad, spring deflection |
| 35–44 | `ref_angle` | 10 | rad, the stored gait right now |
| 45–46 | `contact` | 2 | 0/1: left, right (one switch per foot) |
| 47–48 | `foot_z` | 2 | m: right, left — may be NaN |
| 49 | `phase` | 1 | 0..1 stride clock |

Do not copy these numbers into code. `nexus_proto.POLICY_INDEX` gives each
field's offset and length by name, and `test_this_table_matches_the_protocol`
fails if this table ever drifts from it again - which it did, silently, across
two protocol versions.

**Raw SI values.** The STM32 applies no scaling, clipping or normalisation.
`ObsAdapter` turns the block into the network's input: the block with the IMU
corrected to the hip centre and the starting yaw removed, plus the previous action
(10) and the commanded speed (1). That is a 61-value frame, stacked three deep at
50 Hz, oldest first, 183 inputs. `Policy.act` applies the frozen training
normaliser.

Joint arrays follow the [joint map](../README.md#joint-map). An idle command is
`[0.0] * 10`.

## Guards

`enable` stays false until all of these hold, and the log gives the reason every 2 s:

| check | why |
|---|---|
| gait live, no health faults, foot heights valid | the observation is real |
| `fused_valid` is OK | needs `ROBOT_CONFIG_CALIBRATED` on the board |
| `ref_angle` within 1° of the policy's own gait | the board must play what the policy trained on |
| stride clock within 2% | the same table at the wrong rate is the wrong gait |
| torso under 20° | the training limit |
| phase in 0.95..0.30 | from rest, the policy fell from other phases in simulation |
| each joint within 5° of `ref_angle` | arming sends it straight there |

Once running, 20° of tilt, a base height under 0.40 m or a non-finite state stands
the board down and latches until the node restarts.

`dry_run` (default true in the launch file) runs all of it and sends zero residuals.

## Parameters

| parameter | default | |
|---|---|---|
| `enable` | false | ask the board to arm once the guards pass |
| `dry_run` | true | compute the residual, send zero |
| `imu_in_base` | `[0.0002, 0, 0.2403]` | IMU origin in base_link, m. Unconfirmed on the robot |

## Notes on the choices

- **The handshake.** The STM32 latches every fault and refuses to arm until it
  sees `enable` false. The node keeps sending it until the board *reports* IDLE or
  ARMED, so restarting the node is always enough to recover. Do not shorten it to a
  fixed count from startup: that finishes before DDS discovery has connected the
  node, the frames are dropped, and the board stays latched. This was found against
  `fake_board` in `gait_passthrough_node`, which had exactly that bug.
- **Staleness.** If `/zeus/state` stops updating for 20 ms the node keeps publishing
  with `enable` false, so the board stands its drives down instead of acting on an
  old observation.
- **Node attribute names.** `rclpy.Node` owns `guards`, as a read-only property.
  Assigning to it fails at start-up, and only a real ROS run shows it.

## hardcoded_actuator_test_node.py

Left as it was. It publishes to `/single_actuator_command_controller/commands`,
a ros2_control topic that no longer exists, so it does nothing on this stack.
The equivalent now is `zeus_link`'s `gait_passthrough_node`.
