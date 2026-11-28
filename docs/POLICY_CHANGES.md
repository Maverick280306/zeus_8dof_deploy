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

- **Upright to release: tilt under 3°** (`RELEASE_TILT_DEG`, 2026-11-25). **Temporarily 15° since 2026-11-27** (builder's call): the IMU board is loose and its level moved 3–5° on a still robot, so 3° blocked every release. Back to 3° once the mount is fixed. Released
  leaning 7.6° onto the left foot, the gait asked that loaded foot to swing and the
  other leg to push the body the other way; both hips stalled at the trained PD's
  60–70 Nm (KP × error, the drives had 178 Nm available) and it rolled over in 1 s.
  Released at 3°, it walked 3 s. On 2026-11-26 it was released at 5.6° (the leg
  kinematics put it at 4.3°, so the lean was real, not the IMU) and it spun 128°
  and fell back in 1.15 s.
- **On the right foot to release: roll within 3° of −2.8°** (`START_ROLL_DEG`,
  2026-11-28). With no ankle roll, the robot holding the start pose rests on one
  flat sole. The gait lifts the LEFT foot first and starts (and trained) with the
  RIGHT sole flat at −2.8°; every hold 2026-11-26..28 rested on the left foot
  (+0.9…+4.4°, IMU and leg kinematics agreeing within 0.8°). The left foot stayed
  loaded, dragged, the swing hip stalled at −9° of −21°, and the robot fell
  forward before the knee lifted. The hold message says which way to tip it.
- **Both feet must be down to release** (2026-11-25). The gait's first move out of
  phase 0.95 lifts the left foot; a release on the left foot alone (the right
  switch open all hold) left nothing under the robot, and it rolled 8→16° in 0.9 s.
  The estimator check alone allowed it, since one foot is enough for OK.
- **Ankles get a 10° pose tolerance for release** (`ANKLE_POSE_TOL_RAD`, 2026-11-24);
  every other joint keeps 5°. The trained ankle gains are soft and the floor sets
  the ankle: holding the start pose, body weight pushed the right ankle 8.4° off,
  and WALK could never be released.

Thresholds otherwise unchanged: `POSE_TOL_RAD` (5°), `START_WINDOW`, the tilt and
height limits.

### `rl_policy_node.py`

- **Timing log** (2026-11-27): one CSV row per policy action while walking
  (`policy_timing_*.csv` in `~/zeus_logs`, parameter `timing_dir`, empty turns
  it off) - the board tick of the state used, when it arrived, inference start
  and end, publish time. Logging only: no change to what the policy sees or
  sends. `zeus_link latency_report` reads it with the flight recording.

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
| `zeus/commander/return` | from **anywhere** (since 2026-11-24: walking, holding, stopped, after a fall): ramp every joint back to 0 over 3 s and hold. Out of the gait it starts from the measured pose; it clears a fall latch |

Plus `zeus/commander_status`: JSON at 10 Hz with mode, blockers, stop reason
and residual peak. The browser GUI uses these; it never publishes commands
itself.

**Behaviour changes worth knowing:**

| | Vedant's version | now | why |
|---|---|---|---|
| State older than 20 ms | disengage | **repeat the last command** | a 20 ms DDS gap under load dropped the robot out of its ramp; the board's own link watchdog is 200 ms |
| State older than 100 ms | (n/a) | **halt, and stay halted** until START | a tenth of a second blind is not a hiccup |
| After a fall | latched until the **node restarts** | latched until **STOP**, then START | an operator acknowledges it from the GUI |
| A joint jams | (n/a) | **halt**: any hip or knee >20° from its command and moving <5°/s for 0.3 s (ankles excluded) | 2026-11-24: the right knee jammed mid-stride and every later run pushed 36–75 Nm into it. Worst normal error on the first walk was 18°, and only while moving. `zeus_link.precheck.JamWatch` |
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
| 1 | **Board holds at phase 0.95** (0.0 originally; 0.15 and 0.10 tried on 2026-11-24 and reverted the same evening) | the policy starts at 0.95, inside its start window. The ramp to the start pose happens under load, and the soft trained ankles cannot push the body forward: the more ankle bend a pose asks for, the further the robot sits back (0.95: −8/−6° ankles, ~6° back; 0.15: 12–19° back; 0.10: toppled). `sim_check` walks 20 s from rest at 0.95 (tilt p95 5.2°) | `stm32_zeuss` `soft_start.h` |
| 2 | **Right-leg spring encoder signs flipped to −1** (measured 2026-11-23) | `spring_angle` for the right hip and knee had the wrong sign until now | `robot_config.c` `g_spring_enc` |
| 3 | **Estimator fixed**: double precision, garbled-IMU-sample rejection, contact updates on every drive frame | `quat`, `vel_hdg`, `pelvis_z` were drifting to metres per second while standing; now about 0.006 m/s rms | `fusion.c`, `inekf.c`, `lie_group.h` |
| 4 | **Estimator signed off 2026-11-23** (`ROBOT_CONFIG_CALIBRATED 1`) | `fused_valid` now reaches OK once converged, so `arm_blockers` can pass; before this, the policy could never arm | `robot_config.h` |
| 5 | **Drives forced to position control + passthrough on every arm.** Seven of eight drives had velocity or pos-filter modes saved | the trained PD now actually runs on every joint; before, four joints ignored the position target | `act_odrive.c` |
| 6 | **Drive gains = the trained PD**, no integrator: hip/knee KP 150 KD 12, hip roll KP 60 KD 4, ankle KP 30 KD 1.2 | matches training | `robot_config.c` `g_drive_gains` |
| 7 | **Measured gearbox friction:** hip pitch needs more than 12 Nm to move; the training model has 0.05 Nm on every joint | on the first walk the hips stalled 7–10° short of every peak. **Partly compensated since 2026-11-24**: see row 12 | observed 2026-11-23 |
| 8 | **The hold has no torque feed-forward** (as on the first walk). Tried on 2026-11-24 (the gait's torque at the start phase) and switched off the same day: those are mid-stride torques for a particular stance foot, and at phase 0.10 the −13.2 Nm right hip roll shoved the unloaded right leg 9° off its target, so WALK could never release | the start pose sags under weight until release; the gait's torque arrives with the gait | `soft_start.h` `SOFT_HOLD_TAU_SCALE` 0 |
| 13 | **IMU level correction: the board is rolled 8.81° on the torso** — superseded by row 17 (measured 2026-11-24: gravity read 8.7–9.0° of roll on six occasions at the zero pose, both feet down, robot confirmed straight) | `quat`, `gyro` and the estimator now read 0° roll on a straight robot. Before, the policy would have seen a constant 9° sideways tilt (sim tolerance: ±3° of IMU mount error) and the tilt guard started every run 9° into its 20° | `robot_config.c` `g_imu_mount_R`, `g_imu_mount_q` |
| 14 | **Drive position glitch rejection** (2026-11-25): a position frame further from the last good one than its own velocity allows (+2°) is dropped and the last good value kept; 5 in a row are believed | after a rewiring the left knee's encoder read single-frame spikes of −29…+66° (10,000 in ten minutes) while the frame's velocity stayed smooth; the policy was seeing them at 50 Hz | `act_odrive.c` `pos_believable` |
| 15 | **Free ankles in the hold** (2026-11-25) — **OFF again since 2026-11-25** (`222dde6`, `SOFT_HOLD_ANKLES_FREE` 0; the ankles hold the start pose): while holding the start pose the ankles' target follows their measured angle, so they have no stiffness (damping only) and the feet settle flat; WALK drives them to the gait again | the policy starts with the ankles where the floor put them, up to 10° from the start pose (the release check's ankle tolerance), instead of fighting the floor and lifting the front foot | `soft_start.h` `SOFT_HOLD_ANKLES_FREE` |
| 16 | **Encoder `max_error_rate` 0.7 on every drive**, saved 2026-11-25 (was 0.005 / 0.2 / 0.3 / 0.9, mixed) | a burst of bad encoder reads no longer makes a drive declare its encoder lost (0x8) mid-run; bad reads that pass are caught by row 14 | drive config, `test_can_cfg.c` `CANCFG_FIX_ERROR_RATE` |
| 9 | **Protocol v12** | same bytes as v9; the 44-float policy block is unchanged | `link_proto.h` |
| 10 | **Ankle drive `vel_limit` 10 → 40 turn/s** (saved on both ankle drives 2026-11-24) | on the first walk the drive's velocity clamp cut the ankle PD: the left ankle sat 25–31° off its target. The trained ankle PD has no velocity limit, so this moves the real ankle closer to training. Hips and knees stay at 10 | ODrive config, written by `test_can_cfg.c` `CANCFG_FIX_ANKLE_VEL` |
| 12 | **Friction feed-forward on hip pitch 9 Nm, knee 4 Nm, hip roll 1.5 Nm** — **OFF again since 2026-11-25** (`stm32_zeuss` `222dde6`, back to the first walk's configuration; all `g_friction_ff_nm` 0) (output side, in the direction of the commanded velocity, smooth sign with v0 = 0.01 turn/s) | cancels about 60% of the gearbox friction measured on the first walk (hip pitch 15.5 Nm, against 0.05 Nm in training), so the joints act closer to the model the policy trained in. KP/KD unchanged | `robot_config.c` `g_friction_ff_nm`, `app.c` |
| 11 | **Contact = foot switch only.** A pressed switch means that foot is in contact; there is no spring- or force-based contact detection | `fused_valid` (and so arm/release) needs at least one switch pressed. A foot resting on its toe or heel reads open, because the switch is at the centre of the sole | `contact.c`, `fusion.c` |
| 17 | **IMU mount measured in full** (2026-11-26, `stm32_zeuss` `11cc821`): built from two positions, standing straight and tipped forward on flat feet, instead of roll-only corrections. Chip X left, Y back (as marked), tilted about 15°: 6.2° from the table the walks of 2026-11-24…26 used, mostly pitch that was never corrected | `quat`, `gyro`, `imu_accel` and the estimator now agree with the leg kinematics within about 2° held still (forward tilt: IMU +11.75°, legs +13.3/+15.1°). The walks before saw a few degrees of pitch error (sim tolerance ±3°) | `robot_config.c` `g_imu_mount_R`, `g_imu_mount_q` |
| 18 | **Right ankle drive direction reversed** (2026-11-26, `11cc821`): `g_joint_dir` −1, was +1. Measured on its load-side encoder, never re-measured after the drive moved to its motor encoder on 2026-11-22 | **every walk from 2026-11-22 to 2026-11-26 drove the right ankle as the mirror of the policy's command**: toes up when it asked for down, with the reading mirrored to match, so the observation looked like perfect tracking. Found by rocking the robot on flat feet with the ankles backdriven: the ankles read opposite signs; after the fix both read together (−13.7 / −15.1° tipped forward) | `robot_config.c` `g_joint_dir` |
| 19 | **TEMPORARY IMU offset, 9.1°** — superseded by row 20 (2026-11-27, `stm32_zeuss` after `11cc821`): the IMU board is loose on its mount (roll moved 0.3 → 7.45° in minutes with the legs still). Row 17's table turned so the builder's upright reads vertical | the policy's tilt observation is only right while the board stays where it was; it can shift with handling or impacts. **Fix the mount and re-measure (two positions) before trusting any run** | `robot_config.c` `g_imu_mount_R` |
| 20 | **IMU re-mounted rigidly and measured** (2026-11-28, `stm32_zeuss` after `0737ed9`): chip X right, Y forward, Z up, from two held positions (straight; tipped forward on flat feet) with the flashed table undone | straight reads 0.0/0.0°, a 15.9° forward tilt reads +15.9° pitch and 0.0° roll; the legs agree within 1.3°. **The previous table would have read a forward lean as backward** for this mount. The release tilt limit (row in the soft-start section, temporarily 15°) can go back to 3° once a walk confirms the mount holds | `robot_config.c` `g_imu_mount_R`, `g_imu_mount_q` |

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
