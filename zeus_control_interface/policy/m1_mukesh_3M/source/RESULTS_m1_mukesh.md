# Mukesh: gait library and policy results

Run `m1_mukesh`, library `variants/mukesh/libs/M1`. Gate thresholds are the `eval_zeus.py` ones; `no_leg_collision` counts real MuJoCo contacts.

## Library

| speed m/s | cycle s | step m | duty | lean deg | RMS/rating | peak/drive | spring deg | CoT | repeat residual |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 2.077 | 0.156 | 0.61 | 0.96 | 0.56 | 0.45 | 0.10 | 0.177 | 5.0e-03 |
| 0.20 | 1.697 | 0.170 | 0.53 | 1.16 | 0.59 | 0.44 | 0.10 | 0.175 | 5.0e-03 |
| 0.25 | 1.640 | 0.205 | 0.51 | 1.51 | 0.61 | 0.49 | 0.10 | 0.176 | 5.0e-03 |
| 0.30 | 1.565 | 0.235 | 0.51 | 2.00 | 0.63 | 0.56 | 0.10 | 0.197 | 5.0e-03 |
| 0.35 | 1.412 | 0.247 | 0.51 | 2.00 | 0.63 | 0.60 | 0.11 | 0.244 | 5.0e-03 |
| 0.40 | 1.393 | 0.279 | 0.51 | 2.00 | 0.62 | 0.60 | 0.15 | 0.336 | 5.0e-03 |
| 0.45 | 1.312 | 0.295 | 0.52 | 2.00 | 0.65 | 0.65 | 0.15 | 0.423 | 5.0e-03 |
| 0.50 | 1.363 | 0.341 | 0.51 | 2.00 | 0.66 | 0.64 | 0.24 | 0.619 | 5.0e-03 |
| 0.55 | 1.318 | 0.362 | 0.52 | 2.00 | 0.72 | 0.67 | 0.22 | 0.856 | 5.0e-03 |
| 0.60 | 1.158 | 0.347 | 0.52 | 2.00 | 0.82 | 0.90 | 0.31 | 1.175 | 5.0e-03 |

## Torque per joint in the library (worst over all speeds)

| joint | peak Nm | RMS Nm | peak rad/s | range used deg | drive peak / cont. Nm |
|---|---|---|---|---|---|
| hip_pitch | 74.1 | 22.4 | 3.04 | -28.6 .. +16.6 | 188.0 / 61.1 |
| hip_roll | 39.3 | 12.3 | 2.40 | -11.3 .. +13.1 | 43.7 / 15.1 |
| knee_pitch | 132.8 | 45.7 | 5.44 | +12.2 .. +43.0 | 188.0 / 61.1 |
| ankle_pitch | 8.6 | 3.6 | 6.06 | -40.1 .. +40.1 | 25.2 / 8.7 |
| ankle_roll | 9.1 | 4.3 | 2.05 | -20.1 .. +20.1 | 25.2 / 8.7 |

## Torque per joint with the policy running (worst over evaluated speeds)

| joint | peak Nm | RMS Nm | peak / drive | RMS / rating |
|---|---|---|---|---|
| hip_pitch | 39.8 | 15.0 | 0.21 | 0.25 |
| hip_roll | 25.4 | 9.1 | 0.58 | 0.60 |
| knee_pitch | 65.9 | 25.5 | 0.35 | 0.42 |
| ankle_pitch | 11.6 | 4.1 | 0.46 | 0.47 |
| ankle_roll | 9.3 | 3.5 | 0.37 | 0.41 |

## Training progress

```
steps,v,alive_s,vx,tilt95,da_rms
1000000,0.2500,2.7000,0.3234,17.1481,0.0569
1000000,0.3500,7.8600,0.3723,11.2238,0.0647
1000000,0.4500,3.1400,0.5106,13.4675,0.0709
2000000,0.2500,10.0000,0.2819,6.6858,0.0555
2000000,0.3500,10.0000,0.3757,9.1544,0.0740
2000000,0.4500,10.0000,0.4989,11.8745,0.0883
3000000,0.2500,10.0000,0.2669,4.1422,0.0581
3000000,0.3500,10.0000,0.3732,5.5418,0.0790
3000000,0.4500,10.0000,0.5299,8.8260,0.0984
```

## Gates per speed (policy, mean over seeds)

| command m/s | survive | track | lean<2deg | limit_cycle | no_leg_collision | rms_ok | smooth | not_wild |
|---|---|---|---|---|---|---|---|---|
| 0.20 | PASS | PASS | FAIL | PASS | FAIL | PASS | PASS | PASS |
| 0.25 | PASS | PASS | FAIL | PASS | FAIL | PASS | PASS | PASS |
| 0.30 | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS |
| 0.35 | PASS | PASS | FAIL | FAIL | PASS | PASS | FAIL | PASS |
| 0.40 | PASS | PASS | FAIL | FAIL | PASS | PASS | FAIL | PASS |
| 0.45 | PASS | FAIL | FAIL | PASS | PASS | PASS | FAIL | PASS |
| 0.50 | PASS | PASS | FAIL | PASS | FAIL | PASS | FAIL | PASS |
| **speeds passed of 7** | 7 | 6 | 0 | 5 | 4 | 7 | 3 | 7 |

## Measured values

| command m/s | survive s | speed m/s (% of cmd) | tilt p95 deg | roll / pitch p95 | stride s | stride-time CV | step-length CV | rho | leg contacts | RMS/rating | peak/drive | spring max deg | time over 1 deg | da p95 | HF | sat | residual max deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 20.0 | 0.196 (98%) | 2.74 | 2.7 / 1.5 | 1.69 | 1% | 1% | 0.86 | 1 | 0.57 | 0.41 | 0.24 | 0.0% | 0.111 | 0.038 | 0.0% | 6.2 |
| 0.25 | 20.0 | 0.268 (107%) | 4.12 | 4.1 / 1.6 | 1.64 | 2% | 2% | 0.68 | 2 | 0.59 | 0.45 | 0.45 | 0.0% | 0.119 | 0.025 | 0.0% | 6.7 |
| 0.30 | 20.0 | 0.308 (103%) | 4.79 | 4.4 / 2.2 | 1.56 | 1% | 1% | 0.84 | 0 | 0.59 | 0.48 | 0.24 | 0.0% | 0.136 | 0.039 | 0.0% | 6.8 |
| 0.35 | 20.0 | 0.372 (106%) | 5.57 | 5.1 / 2.4 | 1.41 | 1% | 1% | 0.97 | 0 | 0.58 | 0.53 | 0.32 | 0.0% | 0.161 | 0.034 | 0.0% | 7.2 |
| 0.40 | 20.0 | 0.431 (108%) | 6.89 | 4.5 / 5.1 | 1.39 | 1% | 1% | 1.00 | 0 | 0.57 | 0.51 | 0.37 | 0.0% | 0.184 | 0.038 | 0.1% | 8.5 |
| 0.45 | 20.0 | 0.534 (119%) | 8.88 | 6.3 / 6.2 | 1.31 | 1% | 1% | 0.85 | 0 | 0.60 | 0.54 | 0.35 | 0.0% | 0.209 | 0.038 | 0.6% | 9.7 |
| 0.50 | 20.0 | 0.537 (107%) | 12.42 | 5.1 / 12.3 | 1.36 | 1% | 3% | 0.88 | 2 | 0.60 | 0.58 | 0.67 | 0.0% | 0.194 | 0.054 | 0.2% | 13.8 |

## Baseline: `z1_softland_tol05` (no ankle roll)

| command m/s | survive | track | lean<2deg | limit_cycle | no_leg_collision | rms_ok | smooth | not_wild |
|---|---|---|---|---|---|---|---|---|
| 0.20 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL |
| 0.25 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL |
| 0.30 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL |
| 0.35 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL |
| 0.40 | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| 0.45 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL |
| 0.50 | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL |
| **speeds passed of 7** | 7 | 0 | 0 | 0 | 6 | 7 | 0 | 0 |

| command m/s | survive s | speed m/s (% of cmd) | tilt p95 deg | roll / pitch p95 | stride s | stride-time CV | step-length CV | rho | leg contacts | RMS/rating | peak/drive | spring max deg | time over 1 deg | da p95 | HF | sat | residual max deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 20.0 | 0.133 (66%) | 11.85 | 4.1 / 11.4 | 1.14 | 38% | 76% | 0.82 | 0 | 0.35 | 0.36 | 0.61 | 0.0% | 0.180 | 0.022 | 6.9% | 13.6 |
| 0.25 | 20.0 | 0.175 (70%) | 11.58 | 4.2 / 11.4 | 0.84 | 59% | 98% | 0.48 | 0 | 0.41 | 0.38 | 0.57 | 0.0% | 0.200 | 0.024 | 4.3% | 13.4 |
| 0.30 | 20.0 | 0.222 (74%) | 9.99 | 4.3 / 9.7 | 0.79 | 54% | 50% | 0.71 | 0 | 0.39 | 0.36 | 0.57 | 0.0% | 0.202 | 0.022 | 5.9% | 14.1 |
| 0.35 | 20.0 | 0.249 (71%) | 8.89 | 4.0 / 8.6 | 0.79 | 45% | 55% | 0.74 | 0 | 0.41 | 0.39 | 0.54 | 0.0% | 0.212 | 0.030 | 6.5% | 14.0 |
| 0.40 | 20.0 | 0.291 (73%) | 7.17 | 5.2 / 6.1 | 1.37 | 37% | 56% | 0.67 | 5 | 0.40 | 0.48 | 0.84 | 0.0% | 0.211 | 0.024 | 6.4% | 14.1 |
| 0.45 | 20.0 | 0.303 (67%) | 9.26 | 5.7 / 8.6 | 0.56 | 25% | 111% | 0.90 | 0 | 0.40 | 0.44 | 0.83 | 0.0% | 0.234 | 0.039 | 2.9% | 14.0 |
| 0.50 | 20.0 | 0.324 (65%) | 9.00 | 5.7 / 8.2 | 0.56 | 21% | 119% | 0.95 | 0 | 0.43 | 0.53 | 0.96 | 0.0% | 0.249 | 0.044 | 2.3% | 13.8 |
