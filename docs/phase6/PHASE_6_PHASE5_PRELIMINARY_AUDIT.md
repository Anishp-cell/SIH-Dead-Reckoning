# Phase 6 Preliminary Audit: Phase 5 Navigation Core Pre-Flight Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Audit Document**: `docs/phase6/PHASE_6_PHASE5_PRELIMINARY_AUDIT.md`  

---

## 1. Executive Summary

Phase 5 successfully implemented a 15-state indirect Error-State Kalman Filter (ESKF) running at $10\text{ Hz}$ that propagates calibrated smartphone inertial dynamics and incorporates the Phase 4 forward velocity prediction with heteroscedastic uncertainty.

Before extending this filter with vehicle physical constraints (Non-Holonomic Constraints, Zero-Velocity Updates, Zero-Angular-Rate Updates, and disturbance-aware weighting) in Phase 6, this preliminary audit evaluates:
1. State definitions and reference points.
2. Coordinate frames and quaternion error conventions.
3. Measurement update abstractions.
4. Upstream Phase 3/4 detector interfaces.
5. Benchmark generation reproducibility and physical consistency.
6. Required fixes and terminology updates.
7. Proposed Phase 6 software architecture.

---

## 2. State & Reference Point Conventions

### 2.1 Nominal Navigation State (16 Dimensions)
Defined in [`state.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/phase5/core/state.py):
$$\mathbf{x} = \begin{bmatrix} \mathbf{p}_n \\ \mathbf{v}_n \\ \mathbf{q} \\ \mathbf{b}_a \\ \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^3 \times \mathbb{R}^3 \times \mathbb{S}^3 \times \mathbb{R}^3 \times \mathbb{R}^3$$
- $\mathbf{p}_n = [p_{\text{east}}, p_{\text{north}}, p_{\text{up}}]^T \in \mathbb{R}^3$: 3D position in the local East-North-Up (ENU) Cartesian frame [m].
- $\mathbf{v}_n = [v_{\text{east}}, v_{\text{north}}, v_{\text{up}}]^T \in \mathbb{R}^3$: 3D velocity in the local ENU Cartesian frame [m/s].
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{S}^3$: Attitude quaternion mapping Body frame to Navigation frame ($R_{nb}$).
- $\mathbf{b}_a = [b_{a,x}, b_{a,y}, b_{a,z}]^T \in \mathbb{R}^3$: Accelerometer sensor bias in the Body frame [$\text{m/s}^2$].
- $\mathbf{b}_g = [b_{g,x}, b_{g,y}, b_{g,z}]^T \in \mathbb{R}^3$: Gyroscope sensor bias in the Body frame [$\text{rad/s}$].

### 2.2 Indirect Error State (15 Dimensions)
Defined in [`state.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/phase5/core/state.py):
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_n \\ \delta\mathbf{v}_n \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$
- $\delta\mathbf{p}_n \in \mathbb{R}^3$: Position error in ENU navigation frame [m].
- $\delta\mathbf{v}_n \in \mathbb{R}^3$: Velocity error in ENU navigation frame [m/s].
- $\delta\boldsymbol{\theta} \in \mathbb{R}^3$: Small-angle orientation error defined in the **Body frame** [rad].
- $\delta\mathbf{b}_a \in \mathbb{R}^3$: Accelerometer bias error in the Body frame [$\text{m/s}^2$].
- $\delta\mathbf{b}_g \in \mathbb{R}^3$: Gyroscope bias error in the Body frame [$\text{rad/s}$].

### 2.3 Reference Point & Lever-Arm Audit
- In Phase 5, the state represents the **smartphone IMU location** mounted on the vehicle dashboard.
- If the vehicle rotates at angular rate $\boldsymbol{\omega}_b$, the velocity at the vehicle center/rear axle reference point $\mathbf{v}_{\text{ref}}^b$ relates to the IMU velocity $\mathbf{v}_{\text{imu}}^b$ via the lever arm $\mathbf{r}_{ib}^b$:
  $$\mathbf{v}_{\text{imu}}^b = \mathbf{v}_{\text{ref}}^b + \boldsymbol{\omega}_b \times \mathbf{r}_{ib}^b$$
- *Dataset Reality*: The IO-VNBD dataset does not provide sub-centimeter caliper survey measurements for the exact lever arm from the phone cradle to the vehicle center of gravity.
- *Phase 6 Specification*: Lever arm $\mathbf{r}_{ib}^b$ must be exposed as an explicit configurable parameter in Phase 6 (defaulting to $\mathbf{0}$), and NHC confidence must causally down-weight during high angular rates $\|\boldsymbol{\omega}_b\|$ when lever-arm velocity induced by turning could introduce non-zero lateral velocity.

---

## 3. Coordinate Frames & Quaternion Conventions

### 3.1 Navigation Frame: Local ENU
- **+X**: East ($\mathbf{e}_E$)
- **+Y**: North ($\mathbf{e}_N$)
- **+Z**: Up ($\mathbf{e}_U$)
- **Gravity Vector**: $\mathbf{g}_n = [0, 0, -9.80665]^T\text{ m/s}^2$.

### 3.2 Vehicle Body Frame: Adapted ISO 8855
- **+X**: Forward ($x_{\text{fwd}}$)
- **+Y**: Lateral Right ($y_{\text{lat}}$)
- **+Z**: Vertical Up ($z_{\text{up}}$)
- *Axis Reconciliation from Phase 2*: In Phase 2 ([`phone_vehicle_alignment.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/phone_vehicle_alignment.py)), the vertical mount alignment was computed using $-\mathbf{g} / \|\mathbf{g}\|$, defining downwards as positive in the logged channel `acc_up_veh` (reporting $-9.84\text{ m/s}^2$ when stationary). Phase 5 established the body specific force vector as:
  $$\mathbf{f}_b = \begin{bmatrix} a_{\text{fwd}} \\ a_{\text{lat}} \\ -a_{\text{up\_veh}} \end{bmatrix}$$
  ensuring stationary specific force is $[0, 0, +9.80665]^T\text{ m/s}^2$ upwards.

### 3.3 Rotation Matrix $R_{nb}$
Converts vector from Body frame to Navigation frame:
$$\mathbf{v}_n = R_{nb} \mathbf{v}_b, \quad \mathbf{v}_b = R_{nb}^T \mathbf{v}_n$$
The column vectors of $R_{nb}$ are the Body unit vectors expressed in ENU:
$$R_{nb} = \begin{bmatrix} \mathbf{x}_b^n & \mathbf{y}_b^n & \mathbf{z}_b^n \end{bmatrix}$$

### 3.4 Quaternion Parameterization & Error Injection
- Parameterization: Unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ with $q_w$ scalar-first.
- Multiplication: Standard Hamiltonian quaternion product $\mathbf{q}_1 \otimes \mathbf{q}_2$.
- Attitude Error Definition: Defined as a **Body-frame right multiplication**:
  $$\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nominal}} \otimes \delta\mathbf{q}$$
  where for small angle $\delta\boldsymbol{\theta} \in \mathbb{R}^3$:
  $$\delta\mathbf{q} \approx \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix}$$
- State Update & Reset:
  $$\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\hat{\boldsymbol{\theta}} \end{bmatrix}, \quad \hat{\mathbf{q}}^+ \leftarrow \frac{\hat{\mathbf{q}}^+}{\|\hat{\mathbf{q}}^+\|}, \quad \delta\hat{\mathbf{x}} \leftarrow \mathbf{0}$$

---

## 4. Measurement Update Abstraction Audit

In Phase 5 ([`measurement.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/phase5/core/measurement.py)):
- The measurement update was hardcoded specifically for scalar forward speed:
  $$h(\mathbf{x}) = \mathbf{e}_1^T R_{nb}^T \mathbf{v}_n = v_b[0]$$
  $$H = \begin{bmatrix} \mathbf{0}_{1\times 3} & \mathbf{e}_1^T R_{nb}^T & [0, -v_{\text{up}}, v_{\text{lat}}] & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} \end{bmatrix}$$
- It utilized Joseph-form covariance update:
  $$P^+ = (I - KH)P^-(I - KH)^T + KRK^T$$
- **Phase 6 Refactoring Need**:
  Phase 6 introduces multiple vector and scalar constraints:
  - NHC (2D vector: $v_y^b \approx 0, v_z^b \approx 0$)
  - ZUPT (3D vector: $\mathbf{v}_n \approx \mathbf{0}$)
  - ZARU (3D vector: $\boldsymbol{\omega}_m \approx \mathbf{b}_g$)
  - Phase 4 AI Speed (1D scalar: $v_x^b \approx \hat{v}_{\text{fwd}}$)
  A generic, vectorized measurement engine `RobustMeasurementUpdater` must be implemented to support arbitrary measurement dimension $m$, handling innovation $\boldsymbol{\nu} \in \mathbb{R}^m$, Jacobian $H \in \mathbb{R}^{m\times 15}$, noise $R \in \mathbb{R}^{m\times m}$, Chi-square gating $\text{NIS} = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu} \le \chi^2_{\alpha}(m)$, and Joseph covariance updates without duplicating Kalman update code.

---

## 5. Upstream Phase 3/4 Interfaces

### 5.1 Phase 3 Stationary Detection
- Located in [`zupt_detector.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/zupt_detector.py) and column `is_stationary` in the 12-channel feature dataset.
- Parameters:
  - 10-sample rolling window (1.0 s at 10 Hz)
  - Accelerometer variance threshold: $< 0.04\text{ (m/s}^2)^2$
  - Gyroscope norm threshold: $< 0.045\text{ rad/s}$ ($2.58^\circ/\text{s}$)
  - Gravity difference threshold: $< 0.40\text{ m/s}^2$ ($|\|\mathbf{f}\| - 9.80665|$)
  - Dwell persistence: $\ge 3$ consecutive samples ($0.3\text{ s}$)
- Outputs: `(is_stationary: bool, stationary_duration_s: float)`

### 5.2 Phase 4 Motion Intelligence
- Located in [`streaming.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/phase4/streaming.py).
- Output: `MotionEstimate`:
  - `timestamp: float`
  - `forward_speed_mps: float` (Predicted body forward velocity component $v_{\text{fwd}}$)
  - `speed_uncertainty: float` ($\sigma_v$ from heteroscedastic loss)
  - `motion_state: str` ("STANDSTILL", "CRUISING", "ACCELERATING", "BRAKING", "TURNING")
  - `motion_confidence: float` (Softmax probability $\in [0, 1]$)

---

## 6. Audit of Phase 5 Benchmark Consistency

### 6.1 Investigation of Blackout Distances
A critical check was mandated regarding the reported traveled distances across blackout durations:
- At blackout start $t_0 = 4,600.0\text{ s}$ on sequence S1:
  - **10 s** ($t \in [4600.0, 4610.0\text{ s}]$): Traveled distance = **98.96 m** (Vehicle moving at $11.0 \to 6.4\text{ m/s}$).
  - **30 s** ($t \in [4600.0, 4630.0\text{ s}]$): Traveled distance = **191.78 m** (Vehicle decelerated and stopped at $t \approx 4618.0\text{ s}$).
  - **60 s** ($t \in [4600.0, 4660.0\text{ s}]$): Traveled distance = **191.78 m** (Vehicle remained completely stationary at a traffic stop from $t=4618.0$ to $4668.0\text{ s}$).
  - **120 s** ($t \in [4600.0, 4720.0\text{ s}]$): Traveled distance in raw CSV = **661.44 m** (Vehicle resumed driving at $t \approx 4670.0\text{ s}$ and accelerated up to $14.0\text{ m/s}$).

### 6.2 Identified Discrepancies & Corrections
1. **Walkthrough Markdown Typo**: In the Phase 5 walkthrough text, the 120s traveled distance was mistakenly transcribed as `191.8 m` in the summary table, whereas the actual generated file [`e0_e1_e2_comparison.csv`](file:///d:/python/SIH%2026%20ISRO/Data_details/outputs/phase5/tables/e0_e1_e2_comparison.csv) correctly recorded `661.44 m`.
2. **Physical Insight on 30s vs 60s**: The identical distance (191.78 m) for 30s and 60s is **physically authentic**—the vehicle was held at a red light/traffic standstill. However, because Phase 5 lacked ZUPT, the ESKF integrated sensor noise and residual bias during this 50-second standstill, accumulating over $500\text{ m}$ of artificial position creep!
3. **Multi-Regime Benchmarking Required in Phase 6**:
   Evaluating only one blackout start point ($t_0=4600$) mixes stop-and-go with cruising. Phase 6 benchmarks must evaluate both:
   - Regime A (Pure Cruising & Maneuvering): e.g., $t_0 = 4750.0\text{ s}$ or $4450.0\text{ s}$.
   - Regime B (Stop-and-Go Traffic with Standstill): $t_0 = 4600.0\text{ s}$ (to directly test ZUPT/ZARU effectiveness).

---

## 7. Terminology Correction

### Phase 4 Speed Terminology
- **Correction**: Phase 4 predicts the **forward vehicle velocity component** ($v_{\text{fwd}} = \mathbf{e}_1^T R_{nb}^T \mathbf{v}_n$), NOT total 3D speed magnitude ($\|\mathbf{v}_n\|$).
- In all Phase 6 derivations and documentation, $z_v$ is strictly defined as forward body velocity $v_x^b$.

---

## 8. Proposed Phase 6 Software Architecture

To avoid code duplication and cleanly extend Phase 5, Phase 6 will be organized as follows:

```text
Data_details/src/phase6/
├── __init__.py
├── constraints/
│   ├── __init__.py
│   ├── nhc.py                   # Non-Holonomic Constraints (v_y^b approx 0, v_z^b approx 0)
│   ├── zupt.py                  # Zero Velocity Updates (v_n approx 0)
│   └── zaru.py                  # Zero Angular Rate Updates (omega_m approx b_g)
├── detection/
│   ├── __init__.py
│   ├── stationary_detector.py   # Multi-signal stationary detector with hysteresis
│   └── disturbance_detector.py  # Causal motion/vibration disturbance index for adaptive R_NHC
├── measurement/
│   ├── __init__.py
│   ├── robust_update.py         # Generalized vector/scalar Joseph-form EKF updater
│   └── gating.py                # Chi-square NIS innovation gating for 1D, 2D, 3D measurements
├── models/
│   ├── __init__.py
│   └── constraint_noise.py      # Adaptive covariance generators for NHC, ZUPT, and ZARU
├── core/
│   ├── __init__.py
│   └── phase6_eskf.py           # Phase 6 ESKF subclass/wrapper inheriting Phase 5 core
├── evaluation/
│   ├── __init__.py
│   ├── constraint_metrics.py    # Lateral/vertical velocity, stationary creep, residual metrics
│   ├── phase5_vs_phase6.py      # E2 vs E3 vs E4 vs E5 vs E6 ablation suite
│   └── blackout_benchmarks.py   # Multi-regime held-out blackout benchmarks
├── streaming.py                 # Real-time streaming Phase 6 navigation engine
└── pipeline.py                  # Master Phase 6 evaluation pipeline
```

---

## 9. Pre-Flight Verification Checklist

- [x] Phase 5 nominal state (16D) and error state (15D) verified.
- [x] Navigation frame (ENU) and Body frame (Forward-Right-Up) confirmed.
- [x] Quaternion convention (scalar-first, Body-frame right-multiplication error) confirmed.
- [x] Specific force axis inversion resolved.
- [x] Reason for identical 30s and 60s traveled distance verified (real vehicle standstill).
- [x] Terminology corrected from "speed magnitude" to "forward velocity component".
- [x] All 110 Phase 1–5 tests passing.
- [x] Zero Phase 7–10 code present.
