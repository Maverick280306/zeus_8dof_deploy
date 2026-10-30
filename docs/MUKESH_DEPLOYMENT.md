# Deploying the Mukesh walking policy

For the team bringing the robot up. It says what is in this repo, what has to
happen before the policy can run, and how to run it safely. Firmware detail is in
[firmware_handoff/FIRMWARE_CHANGES.md](../firmware_handoff/FIRMWARE_CHANGES.md).

## Status

**The Pi side is complete. It cannot run on the robot until the firmware is
changed.** This policy has only ever run in simulation.

| | |
|---|---|
| Robot | Mukesh: 10 joints with ankle roll, 11.507 kg ([mukesh_description](../mukesh_description)) |
| Policy | `m1_mukesh`, 3 M training steps, 50 Hz residual on a stored gait |
| Deployed speed | 0.30 m/s, stride 1.565 s |
| Simulated result at 0.30 m/s | walks 20 s, 0.31 m/s, peak residual under 7° |
| Known shortfall | torso tilt p95 is **4.8°** against a 2° target; it passes the other seven gates at this speed |

0.30 m/s was chosen because it is the lowest-risk speed: no leg contacts, stable
limit cycle, smooth actions. The board holds one stride, so the robot walks at
that one speed.

## Confirm these first

Two facts in the model were assumed from CAD. If either is wrong the policy is
trained on the wrong robot, and the first one cannot be fixed by tuning.

1. **Which way the foot points.** The model says forward is +Y in the CAD, which
   makes the foot 99 mm behind the ankle and 62 mm in front (long heel). If the
   long side is really the toe, the robot was modelled walking backwards and the
   policy needs retraining.
2. **Where the IMU is.** The model puts it at the centre of the torso top, 240 mm
   above the hips, with axes matching the robot's. A different position is a
   launch parameter (`imu_in_base`). A different *orientation* needs correcting in
   `robot_config`.

## What is in the repo

| path | what |
|---|---|
| [mukesh_description/](../mukesh_description) | URDF and meshes the policy trained on |
| [zeus_control_interface/](../zeus_control_interface) | the policy node, observation adapter, guards |
| [zeus_control_interface/policy/m1_mukesh_3M/](../zeus_control_interface/policy/m1_mukesh_3M) | the exported policy (`policy.npz`) and everything it came from in `source/` |
| [firmware_handoff/](../firmware_handoff) | generated `gait_ref.c/.h`, `link_proto_v10.patch`, change list |
| [zeus_control_interface/sim_check/RESULTS.md](../zeus_control_interface/sim_check/RESULTS.md) | how sensitive the policy is, measured |
| `policy/m1_mukesh_3M/source/walk_v0.30.mp4` | the simulated walk: what success looks like |

The rest of the workspace is the existing Zeus stack, moved to protocol v10.

**This branch needs v10 firmware.** Against today's v9 board the link refuses to
talk, by design.

## Calibration tolerances

From closed-loop simulation with the deployed code. Survival was over 20 s. These
are one model and one scenario each, so treat them as upper bounds, not as
guarantees.

| quantity | fine | degraded | falls |
|---|---|---|---|
| Joint zero error | ±1° | ±2° (0.25 m/s), ±3° (0.21 m/s, 8.5° tilt) | ±5° |
| Torque feedforward scale | x0.8 to x1.3 | | x0.7 and below, or none |
| Drive stiffness scale | x0.7 to x2 | | x0.5 and below |
| IMU mount pitch error | ±3° | | not reached |
| Total command latency | up to 40 ms | | not reached |

## Bring-up order

Each step proves the one before it. Keep the robot on the gantry until the last.

1. **Flash v10 and build the Pi.** `ros2 run zeus_link link_check --joints` shows
   ~1000 Hz, 0% lost, 10 joints.
2. **Check every joint by hand**, drives idle. Positive direction and zero against
   the URDF, with the ankle roll joints especially. Fit within ±1°.
3. **Check the sensors.** `ros2 run zeus_bringup zeus check`. Press each foot
   switch, and tilt the torso to confirm the IMU axes.
4. **Measure the torque units** (FIRMWARE_CHANGES item 5) on one joint, unloaded.
5. **Tune each joint on the stand**, one at a time, to the stiffness in that same
   item, feet off the ground.
6. **Check timing on the Pi.** Run the policy with `enable:=false` and confirm
   no `tick late` warnings. It took about 2 ms on the laptop; the Pi has not been
   measured.
7. **Dry run, feet off the ground.** Runs the policy but sends zero, so the legs
   play the gait:

   ```bash
   ros2 launch zeus_bringup walk.launch.py policy:=rl enable:=true
   ```

   The log gives the reason for every refusal, for example `stored gait is not the
   training gait` or `phase is outside the start window`.
8. **Live, feet off the ground.** Add `dry_run:=false`. Watch the residual peak in
   the log. Simulation stays under about 7°; much more than that means the
   observation is wrong somewhere, so stop and look.
9. **Lowered onto the ground on the gantry, taking weight**, then released only as
   far as the gantry allows.

`policy:=passthrough` is a stand test only. The stored gait alone falls in 0.7 s
in simulation, so it must never be the first thing tried on the floor.

## Stopping

- `ros2 run zeus_bringup zeus stop` idles the drives, and the joints go limp.
- Killing the SSH session stops commands, and the board idles within 200 ms.
- The node stops on its own at 20° tilt or base height under 0.40 m, the same limits
  as training. That latches: restart the node to go again.

Idling means a limp robot, so the gantry is what catches it.

## Known limits

- Trained with randomised mass, friction, gains and sensor noise, but on flat
  ground and without pushes. The contact model is MuJoCo's.
- Sensing is simplified in simulation: contact is the whole foot box, while the
  real switch is at the sole centre with 3 ms/8 ms debounce, which has not been
  measured against these switches.
- Standing, stopping and turning do not exist. The policy walks forward at one
  speed from a gantry start.
- `m2_mukesh` is a longer run still training on the same gait, without domain
  randomisation, and has not been fully evaluated. If it replaces this policy it
  needs a re-export of `policy.npz` and no firmware change.

## Regenerating

On the training machine, from `biped_walk_rl/platforms/zeus`:

```bash
ZEUS_VARIANT=mukesh QUIET_LIMITS=1 python <repo>/zeus_control_interface/scripts/export_policy.py \
    --run m1_mukesh --speed 0.30 \
    --out <repo>/zeus_control_interface/policy/m1_mukesh_3M --firmware <repo>/firmware_handoff
```
