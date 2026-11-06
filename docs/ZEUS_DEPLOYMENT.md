# Deploying the Zeus (8-DOF) walking policy

For the team bringing the robot up. It says what is in this repo, what has to
happen before the policy can run, and how to run it safely.

## KNOWN LIMITATION — read this first

**`ALL PASS: False`.** From `zeus_control_interface/policy/z1_softland_tol05/source/eval.log`,
at every tested speed (0.20 to 0.50 m/s) the policy fails the `lean<2deg` gate:

| speed (m/s) | tilt95 (deg) | roll (deg) | pitch (deg) | gates failed |
|---|---|---|---|---|
| 0.20 | 11.85 | 4.1 | 11.4 | lean, smooth |
| 0.25 | 11.58 | 4.2 | 11.4 | lean, smooth |
| 0.30 | 9.99 | 4.3 | 9.7 | lean |
| 0.35 | 8.89 | 4.0 | 8.6 | lean |
| 0.40 | 7.17 | 5.2 | 6.1 | lean, limit_cycle |
| 0.45 | 9.26 | 5.7 | 8.6 | lean |
| 0.50 | 9.00 | 5.7 | 8.2 | lean |

Gate order (from `eval.log`): `survive, track, lean<2deg, limit_cycle,
no_leg_collision, rms_ok, smooth, not_wild`. The policy survives all 20 s runs
and tracks velocity reasonably, but torso tilt is roughly 4 to 6x the 2° spec
at every speed — worse than the Mukesh policy's 4.8° p95 at its deployed
speed. This has not been fixed or retrained around. Do not treat this bundle
as closed-loop-validated; it is the best policy currently trained for this
8-DOF robot, not a policy that meets the lean spec.

## KNOWN BLOCKER — wire protocol mismatch

This workspace's wire protocol (`zeus_link/zeus_link/nexus_proto.py`) is
**hardcoded to protocol v10: 10 joints, 5 nodes per bus**, built for the
Mukesh (ankle-roll) robot. The real 8-DOF Zeus hardware
([zeus_description](../zeus_description)) has **8 joints, 4 nodes per bus**
(`index = bus*4 + (node-1)`, no ankle roll) — this is protocol **v9**, per
`firmware_handoff/FIRMWARE_CHANGES.md`'s own comparison table. The v9 Python
source no longer exists in this repo (it was overwritten when the workspace
moved to v10 for Mukesh, and this repo holds only one squashed commit with no
earlier history to recover it from).

**This means:** `zeus_link`, `zeus_msgs`, and the board-side `stm32_zeuss`
firmware as they stand in this workspace will not talk to an 8-joint board.
Byte-for-byte reconstructing v9 from the v10 code and the FIRMWARE_CHANGES.md
table risked getting packet sizes or field layout wrong without the real
source to check against, so it has **not** been attempted here. Rewriting the
protocol back to an 8-joint v9 (or writing a translation shim) is a decision
for whoever owns `zeus_link`/`stm32_zeuss`, not something this package
silently assumes.

Everything **above the wire protocol is already joint-count-generic**:
`policy_runtime.py` and `obs_adapter.py` size all internal arrays from
`policy.npz`'s own `joint_names` (8 entries here), not a hardcoded constant.
So the policy bundle itself is ready; only the transport is blocked.

No `gait_ref.c/.h` firmware files were generated for this reason — the
exporter's C template assumes 5 joints per leg including ankle roll
(`SLOTS` in `export_policy.py`), which does not apply to this 8-joint robot,
and there is no v9 firmware repo state to hand the table to.

## Status

| | |
|---|---|
| Robot | Zeus: 8 joints, no ankle roll ([zeus_description](../zeus_description)) |
| Policy | `z1_softland_tol05`, 6.01 M training steps, 50 Hz residual on a stored gait |
| Deployed speed | 0.30 m/s, stride 1.858 s |
| Simulated result at 0.30 m/s | survives 20 s, vx 0.222 m/s, tilt95 10.0° |
| Known shortfall | fails `lean<2deg` at every tested speed — see above |

0.30 m/s was picked to match the Mukesh precedent (mid-range, no leg
collisions at that speed); it is not a specially safer point here the way it
was for Mukesh, since the lean gate fails across the whole tested range.

## What is in the repo

| path | what |
|---|---|
| [zeus_description/](../zeus_description) | URDF and meshes the policy trained on (the real 8-DOF robot) |
| [zeus_control_interface/](../zeus_control_interface) | the policy node, observation adapter, guards (joint-count-generic) |
| [zeus_control_interface/policy/z1_softland_tol05/](../zeus_control_interface/policy/z1_softland_tol05) | the exported policy (`policy.npz`) and everything it came from in `source/` |
| `policy/z1_softland_tol05/source/eval.log`, `metrics.json` | the closed-loop gate results, all speeds |
| `policy/z1_softland_tol05/source/walk_v0.30.mp4` | the simulated walk at the deployed speed |
| `policy/z1_softland_tol05/source/gait_library_report.txt` | the gait library this policy's stride came from |

There is **no firmware_handoff/ equivalent** for this robot in this repo — see
the blocker above.

## What is NOT in the repo / NOT done

- No v9 (8-joint) wire protocol implementation. `zeus_link` as it stands only
  speaks v10.
- No `gait_ref.c/.h` for an STM32 board — never generated, for the reasons
  above.
- No torque-feedforward / SEA-pre-deflection spec equivalent to Mukesh's
  FIRMWARE_CHANGES item 3 has been written for this robot — it was not
  attempted since there is no firmware target it would apply to yet.
- No real-robot joint sign/direction, IMU mount position, or ODrive torque
  unit confirmation has been done — this policy has only ever run in
  MuJoCo simulation.

## Bring-up order (once the protocol blocker is resolved)

Do not attempt bring-up before someone decides how the Pi talks to an 8-joint
board. Once that exists, follow the same shape as
[docs/MUKESH_DEPLOYMENT.md](MUKESH_DEPLOYMENT.md)'s bring-up order (joint
checks with drives idle, sensor check, torque unit measurement, stand tuning,
dry run with `enable:=true` and zero residual, then live feet-off-ground,
then gantry), substituting 8 joints for 10 and this manifest's
`policy.npz` for Mukesh's. Given the lean-gate failure above, expect larger
tilt excursions during the live feet-off-ground step than Mukesh showed, and
do not be surprised if that step alone disqualifies this policy for a floor
test.

## Stopping

Same mechanism as Mukesh: `ros2 run zeus_bringup zeus stop`, killing the SSH
session, or the node's own 20°-tilt / 0.40 m height cutoffs (which latch until
restart) — all apply unchanged, since `rl_policy_node` and the guards are
shared code.

## Regenerating

On the training machine, from `biped_walk_rl/platforms/zeus` (leave
`ZEUS_VARIANT` unset — unset selects the real 8-DOF robot):

```bash
QUIET_LIMITS=1 python <repo>/zeus_control_interface/scripts/export_policy.py \
    --run z1_softland_tol05 --speed 0.30 \
    --out <repo>/zeus_control_interface/policy/z1_softland_tol05
```

Omit `--firmware`: the exporter's C template does not support an 8-joint,
no-ankle-roll layout (see blocker above).
