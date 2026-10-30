# Firmware changes for the Mukesh policy (`stm32_zeuss`)

Written against `stm32_zeuss` at `14953f5`. **Nothing here has been built or
flashed.** Only `link_proto_v10.patch` has been checked, with that repo's own
`tools/check_proto.py`, which passes against this workspace's `nexus_proto.py`.

Items 1 to 5 are required. In simulation the policy falls within 2 s without
item 3 and from most start phases without item 4.

## 1. Protocol v10: ten joints

Apply `link_proto_v10.patch`, then copy `zeus_link/zeus_link/nexus_proto.py` over
`pi/nexus_proto.py`. Board and Pi must change together: the version byte makes
either side refuse the other.

| | v9 | v10 |
|---|---|---|
| joints | 8 | 10, `index = bus * 5 + (node - 1)` |
| state packet | 426 B | 478 B |
| command packet | 44 B | 52 B |
| gains message | 106 B | 130 B |
| policy block | 44 floats | 50 floats |

| index | bus | node | joint |
|---:|---:|---:|---|
| 0 | 0 | 1 | left_hip_pitch |
| 1 | 0 | 2 | left_hip_roll |
| 2 | 0 | 3 | left_knee_pitch |
| 3 | 0 | 4 | left_ankle_pitch |
| 4 | 0 | 5 | **left_ankle_roll** |
| 5 | 1 | 1 | right_hip_pitch |
| 6 | 1 | 2 | right_hip_roll |
| 7 | 1 | 3 | right_knee_pitch |
| 8 | 1 | 4 | right_ankle_pitch |
| 9 | 1 | 5 | **right_ankle_roll** |

The patch covers the header and `check_proto.py` only. Everything sized by the
old map still has to follow:

- `act_odrive.c`: `ODRV_NODES_PER_BUS` 5.
- `robot_config.c/.h`: `ROBOT_LEG_MOTORS` 5, `g_leg_joints`, `g_drive_gains`.
- `app.c`: `build_reference()`, see item 2.
- `tools/hosttest/*`, `tools/sim/*`, `leg_stream.c`, `gen_gait*.py`: all reference
  the joint map or `GAIT_JOINTS`.

The tag `waist-10-actuators` is the last commit with five nodes per bus and shows
what changed when the count went down.

## 2. Replace the stored gait

Replace `Appli/App/gait_ref.c` and `.h` with the generated files in this folder.
The current table is `final_humangait.xlsx`, stride 1.26 s. This one is the
library row the policy trained on, stride **1.5646 s**, 0.30 m/s.

```c
/* build_reference(): five columns now, and BOTH roll joints flip */
{ NEXUS_J_L_HIP_ROLL,   NEXUS_J_R_HIP_ROLL,   GAIT_COL_HIP_ROLL,   -1.0f },
{ NEXUS_J_L_ANKLE_ROLL, NEXUS_J_R_ANKLE_ROLL, GAIT_COL_ANKLE_ROLL, -1.0f },
```

Keep the existing `gait_sample(phase + 0.5f, left)`. The exporter asserts the
library is half-stride mirror-symmetric, for position, velocity and torque.
Keep `APP_GAIT_SPEED` at 1.0. `rl_policy_node` compares `ref_angle` with its own
copy (1° tolerance) and measures the clock rate (2%), and refuses to enable if
either is off.

## 3. Drive command: add spring pre-deflection and torque feedforward

Today the board sends `ref + residual` with a velocity feedforward and
`Torque_FF = 0`. The policy was trained on:

```
tau_ff  = g_gait_tau at phase, for this joint            (output Nm)
target  = ref + residual
if SEA:   target -= clamp(tau_ff / K, -1.7 deg, +1.7 deg)    K = 3437.75 Nm/rad
vel_ff  = g_gait_vel at phase                            (not the slope of target)
Set_Input_Pos(target, vel_ff, tau_ff)
```

SEA joints are hip pitch and knee pitch. The residual is still bounded alone in
`safety.c`; the bounds there should see the final `target`.

**Measured in simulation (20 s = survived the full run):**

| torque feedforward | result |
|---|---|
| none (today's firmware) | **falls at 1.5 s** |
| x0.5 | falls at 1.7 s |
| x0.7 | falls at 4.1 s |
| x0.8 to x1.3 | survives; tilt grows from 3.6° to 7.6° |

So the scale must be right to about ±20%, which makes the units in item 5 a
safety matter. `Torque_FF` is `int16` in 0.001 Nm and saturates at ±32.7 Nm; the
peak at this speed is 24.5 Nm, so it fits **if the drive's torque is in output Nm**.

## 4. Start: hold the clock, ramp to the pose

`s_phase` free-runs from boot, so the arm happens at an arbitrary point in the
stride, and arming jumps straight to `ref`. Both have to change:

1. Hold `s_phase` at 0 until the drives are armed.
2. On arm, ramp each joint from where it is to `ref(0)`, then hold.
3. Release the clock on a start condition.

The policy never trained a transition from standing. From rest at the reference
pose it survived from phases 0.95 to 0.30 and **fell within 1.5 s from 0.40, 0.50,
0.60, 0.75 and 0.90**. `ref(0)` has the left foot in the air (single support), so
the robot cannot stand there unaided: **the start needs a gantry**, with the robot
lowered as the clock releases. `rl_policy_node` will not engage outside that
window or with any joint more than 5° from `ref`.

Start pose `ref(0)`, degrees:

| | hip pitch | hip roll | knee | ankle pitch | ankle roll |
|---|---:|---:|---:|---:|---:|
| left | +6.26 | -6.19 | +19.00 | -26.18 | +7.13 |
| right | -17.01 | +6.76 | +18.14 | -1.22 | -6.27 |

## 5. Drive stiffness and units

The ODrive law, with `tau_ff` and `vel_ff` from item 3, matches training exactly
when, in output-shaft units:

```
vel_gain = KD * 2 pi    [Nm per turn/s]
pos_gain = KP / KD      [turn/s per turn]
vel_integrator_gain = 0     (training had no integrator)
```

| joint class | trained KP Nm/rad | trained KD Nm s/rad | `pos_gain` | `vel_gain` |
|---|---:|---:|---:|---:|
| hip pitch, knee (47:1) | 150 | 12 | 12.5 | 75.4 |
| hip roll (26:1) | 60 | 4 | 15.0 | 25.1 |
| ankle pitch, ankle roll | 30 | 1.2 | 25.0 | 7.54 |

These are not ready to type in. Check first whether `vel_gain` and `Torque_FF`
are in motor Nm or output Nm on each drive, and which shaft each encoder is on:
`docs/ODRIVE_COMMANDS.md` says the hip and knee read the load and the ankle reads
the motor. If torque is motor-side, divide `vel_gain` and `tau_ff` by the gear
ratio.

| gains relative to training | result |
|---|---|
| x0.33 | falls at 2.5 s |
| x0.5 | falls at 6.8 s |
| x0.7 to x2.0 | survives |

Today's hip default (30 / 10 / 50) is about x0.32 of the training stiffness if the
units are output Nm.

**Ankle gear conflict.** Training assumed 15:1 ankles (25.2 Nm peak, 8.7 Nm RMS).
`ODRIVE_COMMANDS.md` and `gains.yaml` say 9:1. At the same motor, 9:1 gives 15.1
Nm peak and 5.2 Nm RMS against about 11.6 and 4.1 Nm used in simulation, so it
may fit, but the reflected inertia differs from what was trained.

**Ankle roll drive.** The sim assumes the ankle-pitch class. The real drive,
gear and encoder placement need to be confirmed.

## 6. Command envelope

`SAFETY_POS_MIN/MAX_TURNS` are ±10 turns, placeholders. Set them per joint. Below
is what the gait plus the largest possible residual reaches; clamp to the
mechanical stop where that is tighter. Ankle roll runs into the model's ±25° limit
by design at full residual.

| joint | reference deg | with full residual deg | peak speed rad/s | peak FF Nm |
|---|---|---|---:|---:|
| hip pitch | -19.4 .. +8.1 | -33.7 .. +22.5 | 1.18 | 13.6 |
| left hip roll | -11.3 .. +10.8 | -18.2 .. +17.6 | 1.47 | 24.5 |
| right hip roll | -10.8 .. +11.3 | -17.6 .. +18.2 | 1.47 | 24.5 |
| knee pitch | +16.4 .. +23.4 | +2.0 .. +37.7 | 0.47 | 10.8 |
| ankle pitch | -29.4 .. +3.3 | -38.0 .. +11.9 | 1.48 | 6.5 |
| left ankle roll | +5.2 .. +20.1 | -3.4 .. +28.6 | 1.27 | 6.5 |
| right ankle roll | -20.1 .. -5.2 | -28.6 .. +3.4 | 1.27 | 6.5 |

## 7. Estimator

- Add `ZEUS_KIN_Q_ANKLE_ROLL` and run `tools/gen_kinematics.py --urdf
  mukesh_description/urdf/mukesh.urdf`. The foot chain gained a joint, and
  `foot_z`, `pelvis_z` and contact-aided velocity all depend on it. The firmware
  CI's sha check must point at `mukesh.urdf`.
- `pelvis_z` and `vel_hdg` stay the IMU's. `obs_adapter.py` moves them to the hip
  centre. Do **not** also correct them on the board.
- `ROBOT_CONFIG_CALIBRATED` must be 1. The node refuses to engage until
  `fused_valid` is OK, which that flag gates.
- `foot_z` comments still say "lower of toe and heel". Training used the sole
  centre, which is also where the switch is.

## Quick reference

| | value |
|---|---|
| Stride | 1.5646 s, 0.30 m/s, 200 samples, linear interpolation |
| Spring K | 3437.75 Nm/rad (60 Nm/deg), travel ±1.7° |
| Residual | ±14.3° hip pitch and knee, ±6.9° hip roll, ±8.6° ankles; 4 Hz low-pass on the Pi |
| Policy | 50 Hz; commands 250 Hz |
