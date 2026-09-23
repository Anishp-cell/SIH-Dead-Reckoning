# Pre-Phase-8 Audit: Data & Signal Provenance Verification

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/02_DATA_AND_PROVENANCE_AUDIT.md`  

---

## 1. Executive Summary & Purpose

Signal provenance tracking guarantees that no artificial, future, or leaked data enters the navigation pipeline. Every input consumed across Phases 3 through 7 is documented below with its exact producer, consumer, mathematical transformation, coordinate frame, physical units, and filtering status.

---

## 2. Complete Signal Provenance Matrix

| Signal Name | Producer Module | Consumer Modules | Mathematical Transformation | Coordinate Frame | Physical Units | Filtering & Normalization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`acc_fwd_veh`** | Phase 2 Calibration (`coordinate_transform.py`) | Phase 4 Training & Streaming Engine | $R_{vp} \mathbf{a}_{\text{phone}}$ (longitudinal body axis projection) | Vehicle Body ($x_b$) | $\text{m/s}^2$ | Raw calibrated; normalized via $\mu=0.0998, \sigma=1.0996$. |
| **`acc_lat_veh`** | Phase 2 Calibration (`coordinate_transform.py`) | Phase 4 Training & Streaming Engine | $R_{vp} \mathbf{a}_{\text{phone}}$ (lateral body axis projection) | Vehicle Body ($y_b$) | $\text{m/s}^2$ | Raw calibrated; normalized via $\mu=0.0345, \sigma=1.0959$. |
| **`acc_up_veh`** | Phase 2 Calibration (`coordinate_transform.py`) | Phase 4 Training & Streaming Engine | $R_{vp} \mathbf{a}_{\text{phone}}$ (vertical body axis projection) | Vehicle Body ($z_b$, inverted mount) | $\text{m/s}^2$ | Raw calibrated; normalized via $\mu=-9.8390, \sigma=0.5317$. |
| **`gyro_roll_veh`** | Phase 2 Calibration (`coordinate_transform.py`) | Phase 4 Training & Streaming Engine | $R_{vp} \boldsymbol{\omega}_{\text{phone}}$ (roll angular rate) | Vehicle Body ($x_b$) | $\text{rad/s}$ | Raw calibrated; normalized via $\mu=-0.0022, \sigma=0.1405$. |
| **`gyro_pitch_veh`** | Phase 2 Calibration (`coordinate_transform.py`) | Phase 4 Training & Streaming Engine | $R_{vp} \boldsymbol{\omega}_{\text{phone}}$ (pitch angular rate) | Vehicle Body ($y_b$) | $\text{rad/s}$ | Raw calibrated; normalized via $\mu=0.0028, \sigma=0.1049$. |
| **`gyro_yaw_veh`** | Phase 2 Calibration (`coordinate_transform.py`) | Phase 4 Training & Streaming Engine | $R_{vp} \boldsymbol{\omega}_{\text{phone}}$ (yaw angular rate) | Vehicle Body ($z_b$) | $\text{rad/s}$ | Raw calibrated; normalized via $\mu=-0.0016, \sigma=0.0532$. |
| **`jerk_fwd`** | Phase 3 Conditioning | Phase 4 Training & Streaming Engine | Discrete derivative: $(a_{\text{fwd}}[k] - a_{\text{fwd}}[k-1]) / \Delta t$ | Vehicle Body ($x_b$) | $\text{m/s}^3$ | Backward finite difference; normalized via $\mu=0.0010, \sigma=10.4093$. |
| **`acc_horiz_norm`** | Phase 3 Conditioning | Phase 4 Training & Streaming Engine | Vector magnitude: $\sqrt{a_{\text{fwd}}^2 + a_{\text{lat}}^2}$ | Vehicle Horizontal Plane | $\text{m/s}^2$ | Euclidean norm; normalized via $\mu=1.3114, \sigma=0.8376$. |
| **`gyro_norm`** | Phase 3 Conditioning | Phase 4 Training & Streaming Engine | Total angular speed: $\sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ | Vehicle Body | $\text{rad/s}$ | Euclidean norm; normalized via $\mu=0.1392, \sigma=0.1192$. |
| **`ori_pitch_deg`** | Phase 2 Calibration | Phase 4 Training & Streaming Engine | Gravitational tilt angle from accelerometer | Gravity vs. Body | Degrees ($^\circ$) | Geometric tilt; normalized via $\mu=-80.935, \sigma=2.9089$. |
| **`vibration_energy`** | Phase 3 Conditioning | Phase 4, Phase 6 Disturbance Detector | High-pass power: $(a_{z, \text{raw}} - a_{z, \text{filtered}})^2$ | Vehicle Vertical Axis | $\text{m}^2/\text{s}^4$ | High-frequency variance; normalized via $\mu=388.37, \sigma=41.99$. |
| **`is_stationary`** | Phase 3 Stationary Detector | Phase 4 Training & Streaming Engine | Acceleration variance threshold detector | Scalar flag | $\{0.0, 1.0\}$ | Binary indicator; normalized via $\mu=0.0664, \sigma=0.2493$. |
| **`f_meas` (ESKF Specific Force)** | Phase 2 Calibration | Phase 5 INS Propagation (`propagation.py`) | $[a_{\text{fwd}}, a_{\text{lat}}, -a_{\text{up}}]^T$ (Inverted channel 2 reconciled) | Vehicle Body ($b$) | $\text{m/s}^2$ | Unnormalized; bias compensated dynamically: $\mathbf{f}_b - \hat{\mathbf{b}}_a$. |
| **`omega_meas` (ESKF Angular Rate)**| Phase 2 Calibration | Phase 5 INS Propagation (`propagation.py`) | $[\omega_{\text{roll}}, \omega_{\text{pitch}}, \omega_{\text{yaw}}]^T$ | Vehicle Body ($b$) | $\text{rad/s}$ | Unnormalized; bias compensated dynamically: $\boldsymbol{\omega}_b - \hat{\mathbf{b}}_g$. |
| **`v_speed_mps` (Reference Target)**| Vehicle CAN Bus (`V-S1.csv`) | Phase 4 Training loss only; Benchmark evaluation only | Synchronized wheel-encoder velocity | Vehicle Longitudinal Axis | $\text{m/s}$ | **STRICTLY ISOLATED**: Zero CAN signals enter the runtime estimator. |
| **`gps_lat`, `gps_lon`, `gps_alt`** | Survey-Grade RTK GNSS / VBOX | Local ENU Origin Setup, Benchmark Evaluation only | WGS84 Geodetic to Cartesian ENU Projection | Local ENU ($n$) | Meters ($\text{m}$) | **STRICTLY ISOLATED**: Zero GPS signals enter during simulated outages. |

---

## 3. Data Integrity & Leakage Verification

### 1. Training vs. Evaluation Isolation
- **Training Interval**: $t \in [0.0\text{ s}, 3,622.2\text{ s}]$ (Sequence S1 first 70%).
- **Validation Interval**: $t \in [3,627.2\text{ s}, 4,398.4\text{ s}]$ (Separated by a 50-sample / 5-second temporal purge gap).
- **Test / Benchmark Interval**: $t \ge 4,600.0\text{ s}$ (Blackout evaluated at $4,600 - 4,720\text{ s}$, separated by an extensive 200-second purge gap).
- **Leakage Audit**: The normalization parameters in `normalization.json` were strictly computed on the training interval ($N=7,239$ samples). Zero statistics or information from the test interval were used in computing mean and standard deviation.

### 2. Causality Verification
At each navigation step $k$ at time $t_k$:
- IMU propagation uses specific force and angular velocity measured at $t_k$.
- Phase 4 AI inference uses a FIFO buffer containing samples $[t_{k-29}, t_{k-28}, \dots, t_k]$.
- Max input timestamp satisfies: $\max(t_{\text{in}}) = t_k \le t_k$.
- **Zero Future Samples**: No forward-backward filtering, non-causal smoothing, or future lookahead is permitted in the streaming navigation pipeline.
