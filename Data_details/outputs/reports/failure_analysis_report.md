# Failure Analysis Report — Raw Dead Reckoning Baseline

## Sequence: S1

## Blackout Configuration

- **Start**: 150.0s
- **Duration**: 60s
- **Samples**: 600
- **Distance Travelled**: 472.9m

## Error Summary

| Metric | Value |
|--------|-------|
| Endpoint Error | 284.9 m |
| RMSE | 184.1 m |
| Max Error | 291.1 m |
| Drift % | 60.2% |
| Max Error Time | 205.0s |

## Error Growth Rate

- At 25% of blackout: 112.0m
- At 50% of blackout: 142.0m
- At 75% of blackout: 248.9m
- At end of blackout: 284.9m
- Average error growth rate: 4.76 m/s

## Dynamic Events During Blackout

| Event Type | Samples | % of Window | Peak Value | Unit |
|------------|---------|-------------|------------|------|
| High Angular Velocity (Turning) | 7 | 1.2% | 0.444 | rad/s |
| Hard Braking | 5 | 0.8% | -5.22 | m/s² |
| High Lateral Acceleration (Sharp Turn) | 11 | 1.8% | 4.29 | m/s² |

## Likely Root Causes of Large Drift

1. **Accelerometer Bias Leakage**: Even small constant bias (~0.01 m/s²) causes quadratic position growth: error ≈ ½ × bias × t², leading to ~18m error in 60s.

2. **Gravity Removal Imperfection**: The Android gravity sensor is itself a filtered estimate. Any residual gravity component directly integrates into position error.

3. **Phone-to-Vehicle Frame Misalignment**: The simplified yaw-only rotation ignores the full 3D orientation matrix. Pitch/roll contributions leak acceleration between axes.

4. **Sensor Noise Double-Integration**: White noise in acceleration becomes a random walk in velocity and Brownian motion in position (error grows as t^1.5 for white noise).

5. **No Velocity Constraints**: A real vehicle cannot accelerate sideways or exceed physical speed limits, but raw INS has no such knowledge.


## Recommendations for Phase 2+

1. Implement proper phone-to-vehicle calibration (full rotation matrix)
2. Apply ZUPT (Zero Velocity Update) corrections during detected stops
3. Use vehicle non-holonomic constraints (no sideslip)
4. Apply low-pass filtering to remove vibration/bumps before integration
5. Train AI/ML model to predict velocity directly from IMU windows
6. Implement EKF/ESKF state estimator for proper sensor fusion