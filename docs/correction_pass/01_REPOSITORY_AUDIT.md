# Pre-Phase-8 Audit: Complete Repository Architecture & Call-Graph Trace

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/01_REPOSITORY_AUDIT.md`  

---

## 1. Executive Summary & Purpose

This audit establishes the definitive source-of-truth mapping for the SIH26168 navigation stack from raw smartphone sensor logs to the final dead-reckoning state. It traces the full data-flow graph, identifies all duplicated preprocessing paths, and isolates where evaluation code diverges from streaming runtime implementation.

---

## 2. End-to-End Source-of-Truth Pipeline Table

| Stage | Actual Code Module | Primary Inputs | Outputs Produced | Physical Units | Coordinate Frame | Nominal Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Raw Ingestion** | `Data_details/src/dataset_loader.py` | `S-Dataset/S-S1.csv`, `V-Dataset/V-S1.csv` | Synchronized sensor DataFrames | Acc: $\text{m/s}^2$, Gyro: $\text{rad/s}$, Speed: $\text{m/s}$ | Phone Body ($b_{\text{phone}}$) | 100 Hz (IMU), 10 Hz (GPS/CAN) |
| **Phase 2 Calibration** | `Data_details/src/coordinate_transform.py` | Raw accel, gyro, gravity vector | Misalignment matrix $R_{vp}$, calibrated IMU | Acc: $\text{m/s}^2$, Gyro: $\text{rad/s}$ | Vehicle Frame ($b_{\text{veh}}$) | 100 Hz |
| **Phase 3 Feature Conditioning** | `Data_details/src/filter_design.py`, `Data_details/outputs/phase3/...` | Calibrated IMU signals | 12 conditioned feature channels | Acc: $\text{m/s}^2$, Gyro: $\text{rad/s}$, Jerk: $\text{m/s}^3$ | Vehicle Frame ($b_{\text{veh}}$) | 10 Hz (downsampled/averaged) |
| **Phase 4 AI Training** | `Data_details/src/phase4/dataset.py`, `trainer.py` | `s1_filtered_causal_imu.csv`, CAN speed | Trained checkpoint `best_model.pt`, `normalization.json` | Normalized feature windows $30\times 12$ | Dimensionless | 10 Hz (stride 5) |
| **Phase 4 Runtime Inference** | `Data_details/src/phase4/streaming.py` | 12-channel IMU vector, timestamp | `MotionEstimate` ($\hat{v}_{\text{fwd}}, \sigma_v, \text{state}, c_m$) | Speed: $\text{m/s}$, Sigma: $\text{m/s}$ | Vehicle Longitudinal ($x_b$) | 10 Hz (causal rolling buffer $W=30$) |
| **Phase 5 INS Propagation** | `Data_details/src/phase5/core/propagation.py` | Specific force $\mathbf{f}_b$, angular rate $\boldsymbol{\omega}_b$, $dt$ | Propagated nominal state $(\mathbf{p}, \mathbf{v}, \mathbf{q})$ and covariance $P$ | Pos: $\text{m}$, Vel: $\text{m/s}$, Att: Quat | Local ENU ($n$) | 100 Hz or 10 Hz steps |
| **Phase 5 AI Speed Update** | `Data_details/src/phase5/core/measurement.py` | $\hat{v}_{\text{fwd}}$, $\sigma_v$ from Phase 4 | Indirect error state correction $\delta\hat{\mathbf{x}}$, updated $P$ | Residual: $\text{m/s}$, NIS: dimensionless | Local ENU ($n$) & Body ($b$) | 10 Hz |
| **Phase 6 Vehicle Physics** | `Data_details/src/phase6/constraints/` (`nhc.py`, `zupt.py`, `zaru.py`) | $\mathbf{v}_b$, $\boldsymbol{\omega}_b$, stationary & disturbance indices | Clamped lateral/up velocity, zero-velocity updates | Residuals: $\text{m/s}$, $\text{rad/s}$ | Body ($b$) & ENU ($n$) | 10 Hz |
| **Phase 7 Road Matching** | `Data_details/src/phase7/matching/` (`candidate_generator.py`, `temporal_matcher.py`) | $\mathbf{p}_{\text{pre}}$, $\psi_{\text{pre}}$, $P_{\text{pos}}$, road graph | `MapMatchResult` (segment ID, $d_{\perp}$, $\psi_{\text{road}}$, $c_{\text{map}}$) | Distance: $\text{m}$, Heading: $\text{rad}$ | Local ENU ($n$) | 10 Hz |
| **Phase 7 Soft Map ESKF** | `Data_details/src/phase7/matching/map_measurement.py`, `map_update.py` | $d_{\perp}$, $\psi_{\text{road}}$, $c_{\text{map}}$ | Final corrected state $(\mathbf{p}^+, \mathbf{v}^+, \mathbf{q}^+, P^+)$ | Cross-track: $\text{m}$, Heading: $\text{rad}$ | Local ENU ($n$) | 10 Hz |

---

## 3. Discovered Duplicated Preprocessing Paths

### 1. Feature Channel Extraction (CRITICAL DEFECT)
- **Observed Fact**:
  - In `Data_details/src/phase4/dataset.py` (Training), lines 21–34:
    ```python
    FEATURE_COLUMNS = [
        "acc_fwd_veh", "acc_lat_veh", "acc_up_veh",
        "gyro_roll_veh", "gyro_pitch_veh", "gyro_yaw_veh",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    ```
  - In `Data_details/src/phase7/evaluation/blackout_benchmarks.py` (Runtime Benchmark), lines 102–109:
    ```python
    feature_cols = [
        "acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered",
        "gyro_roll_veh_filtered", "gyro_pitch_veh_filtered", "gyro_yaw_veh_filtered",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    ```
- **Derived Result**:
  The evaluation pipeline substituted low-pass filtered acceleration and gyro signals for the raw vehicle-frame kinematic channels. The low-pass filter stripped high-frequency vibrations that the neural network was explicitly trained to detect. Correlation between the training feature and runtime input on vertical acceleration dropped to $r = 0.2044$.
- **Fix**: Align `blackout_benchmarks.py` and `Phase7StreamingEngine` to use the exact `FEATURE_COLUMNS` defined in `dataset.py`.

---

## 4. Discovered Evaluation Bypasses of Runtime Code

### 1. Cold-Start Buffer Reset at Outage Onset
- **Observed Fact**:
  In `Data_details/src/phase7/pipeline.py` (lines 159 & 173):
  ```python
  ai_engine.reset()
  res_p6 = run_phase7_trajectory(...)
  ```
  Immediately prior to running each blackout slice ($t_0 = 4,600.0\text{ s}$), `ai_engine.reset()` was called. This filled the 30-sample rolling FIFO buffer with zeros.
- **Derived Result**:
  At $t_0$, the vehicle was traveling at $11.0\text{ m/s}$. The zero-padded buffer caused Model F to predict $2.38\text{ m/s}$ at step 1, introducing an artificial $8.62\text{ m/s}$ speed deficit that took 30 frames ($3.0\text{ s}$) to wash out. This single evaluation bypass artificially generated $>50\text{ m}$ of longitudinal error in short blackouts.
- **Fix**: Formulate two separate benchmark modes:
  - `WARM_START`: Pre-buffer 30 samples using causal historical data strictly before the blackout ($t < t_0$). This represents true operational navigation during sudden GNSS loss.
  - `COLD_START`: Evaluate zero-buffer behavior as an explicit edge-case stress test.

### 2. Standalone Duplicate Metrics Computation
- **Observed Fact**:
  `compute_phase7_metrics` in `map_matching_metrics.py` re-implemented traveled distance, endpoint error, and drift percentage using separate scalar formulas from `phase6/evaluation/constraint_metrics.py`.
- **Derived Result**:
  Metric inconsistencies arose (e.g. cumulative geodesic distance vs. integrated CAN speed).
- **Fix**: Unify metric calculation into a single canonical module used across all phases.

---

## 5. Architectural Verdict

The core algorithmic design (Phase 3 conditioning $\to$ Phase 4 Model F $\to$ Phase 5 ESKF $\to$ Phase 6 NHC $\to$ Phase 7 Map Matching) is conceptually sound. However, the evaluation scripts introduced critical synthetic errors:
1. Feeding filtered features into a model trained on vibration features.
2. Artificially resetting the temporal buffer at blackout onset.

When these evaluation bugs are corrected, the underlying algorithms function as originally designed.
