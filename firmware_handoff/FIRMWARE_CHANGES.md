# Firmware for the Zeus 8-DOF policy (`stm32_zeuss`)

**Everything in this folder is DONE and in the firmware.** This is a record of
what was changed and why, not a list of work outstanding. The firmware repo is
private; ask for access if these files need regenerating.

> This file previously described the **Mukesh** robot's changes — protocol v10,
> ten joints, ankle roll — and shipped `link_proto_v10.patch` alongside. None of
> that applies to this robot and the patch has been deleted. Applying it would
> have taken an 8-joint board to a 10-joint protocol.

## The headline: almost nothing needed changing

`docs/ZEUS_DEPLOYMENT.md` recorded a blocker — this workspace spoke v10 while
the hardware is v9, and the v9 Python "no longer exists in this repo… no
earlier history to recover it from", so rebuilding it was judged too risky.

**It did not have to be rebuilt.** The canonical copy has always lived in the
firmware repo at `stm32_zeuss/pi/nexus_proto.py`, and that repo's
`tools/check_proto.py` verifies it field by field against the C header. It was
run against the v9 `link_proto.h` to confirm agreement before the file was
restored here, so what this workspace now carries is *checked*, not
reconstructed.

And the policy needs no protocol change at all. `zeus_env.py` says its
observation is *"exactly what rl_policy_node gets from the board
(zeus_link.convert.policy_block, 44 raw SI values, same order)"* with NJ = 8 —
which is the v9 policy block unaltered.

Every constant the policy trains against was already in the firmware:

| | policy (`zeus_model.py`) | firmware |
|---|---|---|
| joint order | `hip_pitch, hip_roll, knee_pitch, ankle_pitch` × L,R | `NEXUS_J_*`, identical |
| spring order | L hip, L knee, R hip, R knee | `SPRING_NAMES`, identical |
| `K_SPRING` | 3437.7468 Nm/rad | `ROBOT_SPRING_K_NM_PER_RAD` 3437.75 |
| `SPRING_TRAVEL` | 1.7° | `ROBOT_SPRING_MAX_DEFLECTION_RAD` 0.0297 |
| gears | `N_SEA, N_ROLL, N_ANK = 47, 26, 15` | `g_joint_gear` 47 / 26 / 15 |
| URDF | `zeus_description/urdf/zeus.urdf` | vendored, sha256 `6e7d4e74…`, identical |

## 1. The gait table — `gait_ref.c` / `.h` in this folder

Generated from `policy/z1_softland_tol05/source/gait_library.npz`, row
**v = 0.30 m/s**, stride **1.857507 s**, 200 samples, **4 columns**. Three
tables: position (turns), velocity (turns/s) and the torque the optimiser
solved for (Nm).

`ZEUS_DEPLOYMENT.md` explains why none was generated here: `export_policy.py`'s
C template has `SLOTS` fixed at five joints a leg including ankle roll. These
were written by `stm32_zeuss/tools/gen_gait_from_library.py`, which produces
deliberately the same shape — **so when `SLOTS` is made variable the two agree
and that script can be deleted.** It would be better for this to come out of
`export_policy.py`; it is one line.

One deliberate difference from the exporter's template: it emits

```c
#define GAIT_COL_ANKLE_ROLL  4
```

whatever `SLOTS` holds. With four columns that names an index past the end of
every row, so it is emitted only when there is a column for it.

Three checks refuse to write rather than warn — the library's `joint_order` is
the firmware's order name for name, the speed is a library **row** and not
between two, and the library is half-stride mirror symmetric to 1e-9 under
negated roll. That last is what lets the board play one leg's columns for both
legs, and is the same assertion `export_policy.py` makes.

### Worth raising with whoever solved this gait

**Hip roll steps 0.00833 turns — 3.0° — between row 199 and row 0.** That is
28% of that joint's entire 10.7° range, and about six times what its own peak
velocity covers in one row. The firmware does **not** smooth it, deliberately:
the policy trained on this exact table and `rl_policy_node` compares `ref_angle`
against its own copy, so a board that repaired the seam would be walking a gait
the network never learnt. But once the torque feedforward is live it is a
once-per-stride transient, and it may be worth re-solving that row.

## 2. Feedforward and pre-deflection

All three parts of the drive law `zeus_env.py` trains with are now on the wire.
The `Set_Input_Pos` frame, per drive, every 2 ms:

| bytes | field | units | firmware fills it with |
|---|---|---|---|
| 0–3 | `Input_Pos` | turns of that drive's encoder shaft | `ref_angle + residual − tau/K`, after safety and the interpolator |
| 4–5 | `Vel_FF` | turns/s | `g_gait_vel` at phase |
| 6–7 | `Torque_FF` | **motor** Nm | `g_gait_tau` at phase **÷ gear ratio** |

- **Torque** was hardcoded to 0. `Torque_FF` is motor torque and the table is
  output torque, so the wire carries `g_gait_tau / N`.
- **Velocity** was the slope of the board's own interpolated target — which
  includes the residual and the interpolator's staircase, and is not the
  quantity training used. It is now `g_gait_vel` at the current phase.
- **Pre-deflection**: hip pitch and knee are commanded to `ref − tau/K`,
  clamped to the spring's ±1.7°, because the drive holds the motor side and the
  limb sits behind the spring. `zeus_env.py` applies the same correction.

`ref_angle` in the state packet stays the **pure gait**, not the pre-deflected
value, because the policy reads it as an observation and validates it to 1°.

## 3. Commands at 500 Hz

`DT_PHYS = 0.002` in `zeus_env.py`, so that is the rate the drive law trained
at. The board's loop is still 1 kHz and **nothing on the Pi changes** — the
policy is still 50 Hz and the command stream still 250 Hz. It also halves the
largest single source of CAN traffic.

## What is still NOT done, and is not firmware

- **`ROBOT_CONFIG_CALIBRATED` is 0**, so `fused_valid` never reads OK and
  `rl_policy_node` will not engage. Every joint sign and offset and every spring
  zero is still a placeholder; they have to be measured on the robot.
- **Which shaft each drive's encoder sits on** is unconfirmed, which is what
  decides whether the `vel_gain` table above is right or out by N².
- **The gains have not been tuned on the robot**, and the ankle's predate the
  15:1 gearbox.
- **CAN has never been confirmed working** end to end with all eight drives.
- The firmware is currently built as the single-leg bench test
  (`NEXUS_MODE_LEG_CAN`); the robot needs `NEXUS_MODE_ROBOT`.
