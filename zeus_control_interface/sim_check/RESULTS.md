# Closed-loop check of the deployed code

`closed_loop_check.py` runs the exact Pi code (`obs_adapter`, `policy_runtime`,
`ResidualFilter`, at 250/50 Hz) against the training MuJoCo model, with no SB3.
MuJoCo plays the board: it reports height and velocity at the IMU and an arbitrary
starting yaw, as the firmware does. All runs last 20 s at 0.30 m/s, without noise.
The original evaluation at this speed gave 20 s, tilt p95 4.8°, 0.308 m/s.

`nominal` compares the adapter's observation against the training environment's on
every frame: largest difference 3.2e-4, from MuJoCo reporting sensors one physics
step behind `qvel`.

This shows the code is faithful to training and how much error the policy
tolerates in simulation. It does not show that the simulation matches the robot.

| scenario | survived s | tilt p95 deg | speed m/s |
|---|---:|---:|---:|
| nominal: deployed stack | 20.0 | 4.8 | 0.311 |
| no policy: stored gait, zero residual | 0.7 | 18.7 | 0.401 |
| raw board values, no IMU correction | 2.4 | 11.7 | 0.251 |
| delay 8 ms | 20.0 | 4.8 | 0.308 |
| delay 20 ms | 20.0 | 4.8 | 0.309 |
| delay 40 ms | 20.0 | 4.9 | 0.316 |
| firmware as-is: no torque FF, no pre-deflect | 1.5 | 15.6 | 0.139 |
| torque FF x0.5 | 1.7 | 13.6 | 0.341 |
| torque FF x0.7 | 4.1 | 8.3 | 0.281 |
| torque FF x0.8 | 20.0 | 3.6 | 0.318 |
| torque FF x0.9 | 20.0 | 4.0 | 0.308 |
| torque FF x1.1 | 20.0 | 5.6 | 0.307 |
| torque FF x1.2 | 20.0 | 6.6 | 0.322 |
| torque FF x1.3 | 20.0 | 7.6 | 0.343 |
| gains x0.7 | 20.0 | 4.7 | 0.296 |
| gains x1.5 | 20.0 | 5.4 | 0.319 |
| gains x0.33 | 2.5 | 8.6 | 0.196 |
| gains x0.5 | 6.8 | 11.0 | 0.268 |
| gains x2.0 | 20.0 | 5.4 | 0.314 |
| joint zero error +-0.5 deg | 20.0 | 4.4 | 0.297 |
| joint zero error +-1.0 deg | 20.0 | 4.3 | 0.284 |
| joint zero error +-2.0 deg | 20.0 | 6.0 | 0.245 |
| joint zero error +-3.0 deg | 20.0 | 8.5 | 0.211 |
| joint zero error +-5.0 deg | 1.7 | 17.5 | 0.059 |
| IMU mount pitch error +1.0 deg | 20.0 | 5.2 | 0.311 |
| IMU mount pitch error +2.0 deg | 20.0 | 5.7 | 0.315 |
| IMU mount pitch error +3.0 deg | 20.0 | 6.2 | 0.318 |
| IMU mount pitch error -1.0 deg | 20.0 | 4.5 | 0.308 |
| IMU mount pitch error -2.0 deg | 20.0 | 4.0 | 0.302 |
| IMU mount pitch error -3.0 deg | 20.0 | 3.9 | 0.295 |
| start AT REST, phase 0.00 | 20.0 | 4.7 | 0.301 |
| start AT REST, phase 0.05 | 20.0 | 4.7 | 0.304 |
| start AT REST, phase 0.10 | 20.0 | 5.0 | 0.306 |
| start AT REST, phase 0.15 | 20.0 | 4.9 | 0.311 |
| start AT REST, phase 0.20 | 20.0 | 5.0 | 0.311 |
| start AT REST, phase 0.25 | 20.0 | 4.9 | 0.309 |
| start AT REST, phase 0.30 | 20.0 | 5.0 | 0.311 |
| start AT REST, phase 0.40 | 1.0 | 15.8 | 0.301 |
| start AT REST, phase 0.50 | 1.2 | 16.6 | -0.023 |
| start AT REST, phase 0.60 | 1.2 | 16.5 | 0.100 |
| start AT REST, phase 0.75 | 1.4 | 14.8 | 0.226 |
| start AT REST, phase 0.90 | 1.0 | 14.9 | 0.183 |
| start AT REST, phase 0.95 | 20.0 | 5.2 | 0.306 |
