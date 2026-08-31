# Phase 1 Implementation Audit for Phase 2 Integration

## 1. Existing Relevant Modules in `Data_details/src/`

- **[`dataset_loader.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/dataset_loader.py)**:
  - `DatasetLoader`: Resolves LFS pointers / local cache (`Data_details/data/raw`), parses CSVs with `latin1` encoding.
  - `standardize_smartphone_df`: Renames 24 raw channels to clean snake_case:
    - Accelerometer: `acc_x`, `acc_y`, `acc_z` ($m/s^2$)
    - Gravity: `grav_x`, `grav_y`, `grav_z` ($m/s^2$)
    - Gyroscope: `gyro_x`, `gyro_y`, `gyro_z` ($rad/s$)
    - Magnetometer: `mag_x`, `mag_y`, `mag_z` ($\mu T$)
    - Orientation: `ori_yaw_deg`, `ori_pitch_deg`, `ori_roll_deg` ($^\circ$)
    - GPS: `gps_lat`, `gps_lon`, `gps_alt`, `gps_speed_kmh`, `gps_speed_mps`, `gps_acc_m`, `gps_heading_deg`, `gps_sats`
    - Timing: `time_ms`, `datetime_raw`, `time_s` (normalized to $t_0 = 0$), `dt` (actual sampling period)
  - `standardize_vehicle_df`: Standardizes 29 ECU/CAN-bus channels including 4 wheel speeds, steering angle, engine RPM, brake pressure, yaw rate.

- **[`coordinate_transform.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/coordinate_transform.py)**:
  - `geodetic_to_ecef`, `ecef_to_enu`, `geodetic_to_enu`: High-precision WGS84 Geodetic $\to$ local metric East-North-Up (ENU) tangent plane conversion.
  - `haversine_distance`, `cumulative_distance_m`, `enu_cumulative_distance`: Metric distance calculation utilities.

- **[`stationary_analysis.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/stationary_analysis.py)**:
  - `detect_stationary_periods`: Detects stops using GPS speed ($v < 0.5\text{ m/s}$) with minimum duration ($T \ge 5\text{ s}$).
  - `analyze_stationary_bias`: Computes per-segment mean/std across sensor channels.

- **[`blackout_simulator.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/blackout_simulator.py)**:
  - `create_blackout_mask`, `simulate_blackout`, `simulate_multiple_blackouts`: Non-destructive GPS field masking during defined time windows.

- **[`inertial_baseline.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/inertial_baseline.py)**:
  - `gravity_removal_simple`: Subtracts Android gravity sensor in body frame: $\vec{a}_{\text{lin}} = \vec{a}_{\text{raw}} - \vec{g}_{\text{android}}$.
  - `phone_to_enu_simple`: Simplified 2D yaw-only projection to East-North.
  - `integrate_dead_reckoning`: Trapezoidal double numerical integration.
  - `run_inertial_baseline`: Runs dead reckoning during blackout window and compares with reference trajectory.

- **[`metrics.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/metrics.py)**:
  - `compute_trajectory_metrics`: Computes endpoint error ($m$), RMSE ($m$), MAE ($m$), max error ($m$), and drift percentage ($\%$).
  - `compute_error_timeseries`: Timestep-by-timestep position error vector.

---

## 2. Existing Reusable APIs & Functions for Phase 2

1. `DatasetLoader.load_sequence(smartphone_file, vehicle_file)`: Provides standardized sequence `S1` and `Vw1`.
2. `geodetic_to_enu(lat, lon, alt)`: Establishes local ENU reference coordinates and distance ground truth.
3. `create_blackout_mask(time_s, start_s, duration_s)`: Generates identical blackout windows for benchmarking.
4. `compute_trajectory_metrics(ref_east, ref_north, est_east, est_north, time_s, distance_m)`: Standardizes error calculations.

---

## 3. Existing Assumptions in Phase 1

1. **Sampling Frequency**: $f_s = 10.00\text{ Hz}$, mean $\Delta t = 100.0\text{ ms}$ (verified empirically).
2. **Benchmark Windows**: Primary sequence `S1`, blackout start $t_0 = 150.0\text{ s}$, durations $10\text{ s}, 30\text{ s}, 60\text{ s}, 120\text{ s}$.
3. **Reference Origin**: First valid GPS coordinate of sequence `S1` ($(52.401660^\circ\text{ N}, -1.505290^\circ\text{ E})$).
4. **Primary Coordinate Frames**:
   - Navigation frame: Local Cartesian ENU ($X=\text{East}, Y=\text{North}, Z=\text{Up}$).
   - Vehicle frame: Standard SAE ($X=\text{Forward}, Y=\text{Right}, Z=\text{Up}$).
   - Phone body frame: Android sensor frame ($X=\text{Right}, Y=\text{Top}, Z=\text{Screen-Out}$).

---

## 4. Phase 1 Limitations Addressed by Phase 2

1. **Gravity Leakage**:
   - *Phase 1 limitation*: Subtracted Android gravity output directly from body acceleration without 3D world-frame rotation. Any small tilt misestimation projected residual gravity ($9.8\text{ m/s}^2$) directly into forward acceleration.
   - *Phase 2 solution*: Attitude propagation via Quaternions and world-frame gravity subtraction: $\vec{a}_{\text{nav}} = R(q)\vec{a}_{\text{body}} - [0, 0, g]^T$.
2. **Uncalibrated Sensor Biases**:
   - *Phase 1 limitation*: Gyroscope and accelerometer electronic zero-offsets were not subtracted prior to integration.
   - *Phase 2 solution*: Multi-sensor stationary detection and baseline bias subtraction ($\vec{b}_g, \vec{b}_a$).
3. **Phone-to-Vehicle Misalignment**:
   - *Phase 1 limitation*: Assumed phone $Y$-axis roughly aligned with vehicle forward direction.
   - *Phase 2 solution*: Rigorous 3D rotation matrix estimation $R_{\text{phone}\to\text{vehicle}} \in SO(3)$ combining gravity leveling and forward acceleration vectors during straight driving.
4. **Lack of Ablation Benchmarking**:
   - *Phase 2 solution*: Incremental ablation study (V0 to V6) isolating the individual performance impact of each calibration step.

---

## 5. Phase 2 Scope & Extension Strategy

Phase 2 **extends** rather than replaces Phase 1:
- All Phase 1 modules, tests, plots, and tables remain 100% intact.
- Phase 2 outputs are isolated under `Data_details/outputs/phase2/` and `Data_details/tests/phase2/`.
- Phase 2 benchmark evaluation directly re-uses the exact Phase 1 blackout configurations on sequence `S1` to ensure direct, uncompromised comparability.
