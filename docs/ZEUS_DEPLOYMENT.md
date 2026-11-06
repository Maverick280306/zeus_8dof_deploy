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

## The wire protocol blocker is RESOLVED

This section recorded that the workspace spoke protocol v10 (ten joints, built
for Mukesh) while the real hardware is v9 (eight joints, `index = bus*4 +
(node-1)`), that the v9 Python "no longer exists in this repo… no earlier
history to recover it from", and that reconstructing it from the v10 code was
too risky to attempt without the real source to check against.

**It did not have to be reconstructed.** The canonical copy has always lived in
the firmware repo at `stm32_zeuss/pi/nexus_proto.py`, and that repo's
`tools/check_proto.py` verifies it field by field against the C header. It was
run against the v9 `link_proto.h` to confirm agreement *before* the file was
restored here, so what this workspace carries is checked, not guessed.

Restored to v9, all from sources that already existed:

| | |
|---|---|
| `zeus_link/zeus_link/` | `nexus_proto.py`, `convert.py`, `synthetic.py`, `fake_board.py` |
| `zeus_msgs/` | `NexusState.msg`, `NexusCommand.msg`, `SetGains.srv` |
| tests | `test_proto.py`, `test_convert.py`, `test_msg_matches_proto.py` |

426-byte state, 44-byte command, 106-byte gains, **44-float policy block** —
which is exactly the observation `zeus_env.py` trains on, by its own docstring.

**And the firmware is done.** `firmware_handoff/` now carries the real 8-DOF
`gait_ref.c/.h` (0.30 m/s row, stride 1.857507 s, four columns) and a
`FIRMWARE_CHANGES.md` describing what was changed rather than what is needed:
the torque feedforward, the velocity feedforward from the gait table, the SEA
pre-deflection, and commands at the 500 Hz `DT_PHYS` trains at.

### Other things that were wrong and are fixed

Found while doing the above. None of them would have crashed; all of them would
have run the wrong robot.

- **`policy_runtime.py` defaulted to `m1_mukesh_3M`** — the ten-joint bundle —
  so the node would have loaded a policy whose action vector is two joints
  longer than the board has drives. Now `z1_softland_tol05`.
- **`setup.py` installed only `m1_mukesh_3M`**, so the 8-DOF policy never
  reached the Pi's share directory at all.
- **`robot.launch.py` loaded `mukesh_description/mukesh.urdf`**, giving
  `robot_state_publisher` a model with two joints the board never reports.
  `mukesh_description` has been removed and the dependency dropped.
- **`gains.yaml` listed ten joints** and called the ankle 9:1. It is **15:1**,
  confirmed on the robot and in the policy's own `zeus_model.py`.
- **The observation table in `zeus_control_interface/README.md`** still
  described the ten-joint block. `test_policy_readme.py` exists precisely to
  catch that and did.
- **The control-interface tests loaded the Mukesh bundle** and hardcoded `10`,
  `61` and `0.2403` (the Mukesh IMU height; this robot's is 0.1274). They now
  derive from the bundle, and one of them perturbed `q[7]` while asserting on
  `right_knee_pitch` — index 7 on the ten-joint robot, `right_ankle_pitch`
  here, so it had been passing for the wrong reason.

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
