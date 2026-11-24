# Policy deployment: changes since Vedant's handover

**Baseline:** commit `706d305`, 2026-11-06, Vedant Mudras, *"Zeus (8-DOF) policy
deployment: z1_softland_tol05 (6M steps)"*. Everything below was changed after
that, during bring-up on the robot.

This file is kept up to date: any change to `zeus_control_interface/`, or to
anything else that changes what the network sees or the PD it drives, gets an
entry here in the same commit.

To see every line yourself: `git diff 706d305 HEAD -- zeus_control_interface`.

---

## Summary

| | status |
|---|---|
| Network, weights, `policy.npz`, `golden.npz`, `manifest.json` | **unchanged** |
| `sim_check/` (closed-loop check and its results) | **unchanged** |
| Observation construction (`ObsAdapter`: frame, history, normalisation) | **unchanged** |
| Action path (`ResidualFilter`, 50 Hz decimation, scaling, residual clip) | **unchanged** |
| Fall detection (`Guards.fallen`: tilt and height thresholds) | **unchanged** |
| Default bundle | **changed**: it pointed at the 10-joint Mukesh policy |
| How the node arms, holds, releases and stops | **changed**: soft start, operator services |
| What the network sees, through firmware changes | **changed**: see the last section |

---

## Changes inside `zeus_control_interface/`

### `policy_runtime.py`

- `DEFAULT_BUNDLE`: `"m1_mukesh_3M"` → `"z1_softland_tol05"`. The default was
  the ten-joint Mukesh bundle, whose action vector is two joints longer than
  this robot has drives. (`b7679de`)

Nothing else in this file changed.

### `obs_adapter.py`: only the `Guards` class

- **`engage_blockers()` split in two.** `arm_blockers()` is everything except
  where the robot stands. `release_blockers()` is the start-phase window plus
  "every joint within 5° of the reference". The old single check could never
  pass on a standing robot, which is 20–35° from any stride pose. The robot now
  arms first, the board ramps it to the start pose, and then it releases.
  `engage_blockers()` still exists and returns both lists. (`3ce6af5`)
- **`watch_clock()` ignores a frozen clock.** During the board's ramp and hold
  the phase is held still on purpose. Measuring the stride rate across that
  read as a broken clock. (`3ce6af5`)
- **Start window tolerance, `PHASE_EPS = 1e-4`.** The board holds at phase
  0.95 and sends it as float32 (0.94999999), which the strict comparison put
  outside the window `(0.95, 0.30)`. WALK would have been refused at exactly the
  phase the board holds. The window itself is unchanged. (`15d3b07`)

Thresholds unchanged: `POSE_TOL_RAD` (5°), `START_WINDOW`, the tilt and height
limits.

### `rl_policy_node.py`

**Starting from a standing robot (soft start).** (`3ce6af5`)

1. The arm guards pass, and the node sends `enable` + `hold` with zero residual.
2. The board ramps every joint over 3 s to the gait's start pose and holds it,
   with the gait clock frozen.
3. When the board reports HOLD and the release guards pass, the node clears
   `hold`. The stride starts from the held phase and the policy engages.

New parameter `release` (default true). Set false, it stops after step 2:
armed, holding the start pose, not walking.

**Operator interface.** (`c843fe9`, `5dd1fb4`)

| service (`std_srvs/Trigger`) | does |
|---|---|
| `zeus/commander/start` | arm into the hold (not walking) |
| `zeus/commander/walk` | release into the gait; accepted only in mode `ready` |
| `zeus/commander/stop` | enable off. Also the acknowledgement that clears a fall latch |
| `zeus/commander/return` | from the ramp or hold: ramp every joint back to 0 and hold |

Plus `zeus/commander_status`: JSON at 10 Hz with mode, blockers, stop reason
and residual peak. The browser GUI uses these; it never publishes commands
itself.

**Behaviour changes worth knowing:**

| | Vedant's version | now | why |
|---|---|---|---|
| State older than 20 ms | disengage | **repeat the last command** | a 20 ms DDS gap under load dropped the robot out of its ramp; the board's own link watchdog is 200 ms |
| State older than 100 ms | (n/a) | **halt, and stay halted** until START | a tenth of a second blind is not a hiccup |
| After a fall | latched until the **node restarts** | latched until **STOP**, then START | an operator acknowledges it from the GUI |
| Any stop | `disengage()` | `halt()`: enable stays **off**, nothing re-arms on its own | the node used to re-arm itself after a stop |
| Shutdown | publish enable false | same, tolerating a dead ROS context on Ctrl-C | an exception on exit hid whether the stand-down went out |

Unchanged inside `run_policy()`: the 50 Hz decimation, `adapter.frame` →
`adapter.stack` → `policy.act` → `filter.command`, the clip to
`RESIDUAL_LIMIT_RAD`, and `dry_run` sending zero. (A released dry run therefore
walks the stored gait alone. That isn't a change, but the docstring now says
it.)

An ankle-zero pre-check was added (`814ebec`) and later removed (`1fbeb43`):
the robot can't always stand on flat feet to satisfy it. The link node now
zeroes the ankles when it starts, and on demand.

### Tests, packaging, README

- `test_obs_adapter.py`: tests for the arm/release split, the frozen clock, and
  the float32 start phase.
- `e2e_policy.py`: soft start, hold, release, and ankle cases against
  `fake_board`.
- `setup.py`, `package.xml`, `README.md`: back to protocol v9 defaults and the
  8-joint bundle (`b7679de`); `std_srvs` dependency (`c843fe9`).

---

## Changes outside the package that change what the network sees

None of these touch Vedant's code, but each one changes the observation or the
PD the policy acts through. These are the ones that matter for sim-to-real.

| # | change | effect on the policy | where |
|---|---|---|---|
| 1 | **Board holds at phase 0.95**, not 0.0 | the policy starts at 0.95. That's inside its start window; `sim_check` walks 20 s from rest there (tilt p95 5.2°, against 4.7° at phase 0) | `stm32_zeuss` `soft_start.h` |
| 2 | **Right-leg spring encoder signs flipped to −1** (measured 2026-11-23) | `spring_angle` for the right hip and knee had the wrong sign until now | `robot_config.c` `g_spring_enc` |
| 3 | **Estimator fixed**: double precision, garbled-IMU-sample rejection, contact updates on every drive frame | `quat`, `vel_hdg`, `pelvis_z` were drifting to metres per second while standing; now about 0.006 m/s rms | `fusion.c`, `inekf.c`, `lie_group.h` |
| 4 | **Estimator signed off 2026-11-23** (`ROBOT_CONFIG_CALIBRATED 1`) | `fused_valid` now reaches OK once converged, so `arm_blockers` can pass; before this, the policy could never arm | `robot_config.h` |
| 5 | **Drives forced to position control + passthrough on every arm.** Seven of eight drives had velocity or pos-filter modes saved | the trained PD now actually runs on every joint; before, four joints ignored the position target | `act_odrive.c` |
| 6 | **Drive gains = the trained PD**, no integrator: hip/knee KP 150 KD 12, hip roll KP 60 KD 4, ankle KP 30 KD 1.2 | matches training | `robot_config.c` `g_drive_gains` |
| 7 | **Measured gearbox friction:** hip pitch needs more than 12 Nm to move; the training model has 0.05 Nm on every joint | on the first walk the hips stalled 7–10° short of every peak. **Partly compensated since 2026-11-24**: see row 12 | observed 2026-11-23 |
| 8 | **The hold carries the gait's own torque at phase 0.95** (since 2026-11-24; before, it carried none) | the start pose no longer sags under the robot's weight, and release no longer steps the torque from 0 to the gait's value. The first ground walk started 6° back and 5° right, on one foot | `soft_start.c`, `app.c` |
| 9 | **Protocol v12** | same bytes as v9; the 44-float policy block is unchanged | `link_proto.h` |
| 10 | **Ankle drive `vel_limit` 10 → 40 turn/s** (saved on both ankle drives 2026-11-24) | on the first walk the drive's velocity clamp cut the ankle PD: the left ankle sat 25–31° off its target. The trained ankle PD has no velocity limit, so this moves the real ankle closer to training. Hips and knees stay at 10 | ODrive config, written by `test_can_cfg.c` `CANCFG_FIX_ANKLE_VEL` |
| 12 | **Friction feed-forward on hip pitch 9 Nm, knee 4 Nm, hip roll 1.5 Nm** (output side, in the direction of the commanded velocity, smooth sign with v0 = 0.01 turn/s) | cancels about 60% of the gearbox friction measured on the first walk (hip pitch 15.5 Nm, against 0.05 Nm in training), so the joints act closer to the model the policy trained in. KP/KD unchanged | `robot_config.c` `g_friction_ff_nm`, `app.c` |
| 11 | **Contact = foot switch only.** A pressed switch means that foot is in contact; there is no spring- or force-based contact detection | `fused_valid` (and so arm/release) needs at least one switch pressed. A foot resting on its toe or heel reads open, because the switch is at the centre of the sole | `contact.c`, `fusion.c` |

---

## Commits in `zeus_control_interface/` since the baseline

| commit | date | change |
|---|---|---|
| `b7679de` | 2026-11-06 | back to protocol v9; default bundle is the 8-joint policy |
| `3ce6af5` | 2026-11-22 | soft start: arm into a hold, ramp, release; arm/release guard split |
| `814ebec` | 2026-11-22 | ankle-zero pre-check (removed later) |
| `c843fe9` | 2026-11-22 | operator services and status topic for the GUI |
| `5dd1fb4` | 2026-11-22 | BACK TO STANDING (`zeus/commander/return`) |
| `a0461a4` | 2026-11-22 | test update for ankles zeroed at link start |
| `aa48140` | 2026-11-23 | BACK TO STANDING goes to zero (log text) |
| `1fbeb43` | 2026-11-23 | ankle-zero pre-check removed |
| `15d3b07` | 2026-11-23 | start window accepts the board's float32 0.95 |
