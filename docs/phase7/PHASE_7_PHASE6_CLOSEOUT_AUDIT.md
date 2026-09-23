# Phase 7 Pre-Flight: Phase 6 Closeout Audit & Road-Constrained Foundation

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Audit Document**: `docs/phase7/PHASE_7_PHASE6_CLOSEOUT_AUDIT.md`  

---

## 1. Executive Summary

Phase 6 established a vehicle-physics augmented 15-state indirect Error-State Kalman Filter (ESKF) by formulating:
- 2D Non-Holonomic Constraints (NHC) ($v_y^b \approx 0, v_z^b \approx 0$).
- 3D Zero-Velocity Updates (ZUPT) ($\mathbf{v}_n \approx \mathbf{0}$) with multi-signal hysteresis detection.
- 3D Zero-Angular-Rate Updates (ZARU) ($\boldsymbol{\omega}_m \approx \mathbf{b}_g$).
- Causal disturbance-aware adaptive covariance weighting.

This closeout audit verifies the exact state of the system, addresses required terminology and metric corrections, analyzes remaining error sources, and establishes the formal handoff to Phase 7 OpenStreetMap (OSM) map matching.

---

## 2. Phase 6 State & Frame Verification

### 2.1 State Vector Definitions
1. **Nominal State $\mathbf{x} \in \mathbb{R}^{16}$**:
   $$\mathbf{x} = \begin{bmatrix} \mathbf{p}_n \\ \mathbf{v}_n \\ \mathbf{q} \\ \mathbf{b}_a \\ \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^3 \times \mathbb{R}^3 \times \mathbb{S}^3 \times \mathbb{R}^3 \times \mathbb{R}^3$$
2. **Indirect Error State $\delta\mathbf{x} \in \mathbb{R}^{15}$**:
   $$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_n \\ \delta\mathbf{v}_n \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$
   where $\delta\boldsymbol{\theta} \in \mathbb{R}^3$ is the small-angle orientation error defined in the **Body frame** via right-multiplication:
   $$\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nominal}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix}$$

### 2.2 Coordinate Frames
- **Navigation Frame**: Local Cartesian East-North-Up (ENU).
  - $+X$: East, $+Y$: North, $+Z$: Up.
  - Gravity: $\mathbf{g}_n = [0, 0, -9.80665]^T\text{ m/s}^2$.
- **Vehicle Body Frame**: Adapted ISO 8855.
  - $+X$: Forward ($v_{\text{fwd}}$), $+Y$: Lateral Right ($v_{\text{lat}}$), $+Z$: Vertical Up ($v_{\text{up}}$).
  - Specific force reconciliation: $\mathbf{f}_b = [a_{\text{fwd}}, a_{\text{lat}}, -a_{\text{up\_veh}}]^T$.

### 2.3 Measurement Execution Ordering
At each $10\text{ Hz}$ sampling interval:
1. **INS Prediction**: Nominal propagation + covariance propagation ($F_k, Q_k$).
2. **AI Speed Update**: Scalar forward velocity update ($v_x^b = \hat{v}_{\text{fwd}}$).
3. **Causal Disturbance & Stationary Detection**: Hysteresis state machine evaluation.
4. **Stationary Updates**: If verified standstill, apply ZUPT ($\mathbf{v}_n = \mathbf{0}$) then ZARU ($\boldsymbol{\omega}_m = \mathbf{b}_g$).
5. **NHC Update**: 2D body constraint ($v_y^b = 0, v_z^b = 0$) with adaptive covariance $\mathbf{R}_{NHC}(t)$.

---

## 3. Required Phase 6 Corrections & Status

### 3.1 Terminology Correction
- **Enforced**: Phase 4 AI model output is strictly designated as the **forward vehicle velocity component** ($v_{\text{fwd}} = \mathbf{e}_1^T R_{nb}^T \mathbf{v}_n$), not generic 3D speed magnitude.

### 3.2 Stationary Creep Terminology
- **Enforced**: Any claim that ZUPT "eliminated" creep is formally replaced with:
  > **"ZUPT substantially reduced stationary position creep."**
- **Measured Value**: Standstill position creep was reduced from $426.19\text{ m}$ (Phase 5) down to **$6.26\text{ m} - 7.23\text{ m}$** (Phase 6), an authentic **59x reduction**.

### 3.3 Adaptive NHC Claims
- **Distinction**:
  - *Fixed NHC* ($R_{NHC} = \text{diag}(0.04, 0.04)$): Optimal on straight/smooth segments, achieving $109.51\text{ m}$ endpoint error on 60s blackout.
  - *Adaptive NHC* ($\sigma_{\text{lat}}(t) = 0.20 \cdot (1 + 3 D_{\text{lat}})$): Achieved $113.32\text{ m}$ endpoint error ($1.8\text{ m}$ difference), but crucially **prevents NIS gating rejection and filter instability during sharp cornering maneuvers and road shocks**.

### 3.4 Explicit Chi-Square Gating Formalism
Ambiguous "3-sigma" phrasing is replaced with formal Chi-square confidence limits:
$$\text{NIS} = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu} \le \gamma_m = \chi^2_m(1 - \alpha)$$
- $m=1$ (AI Speed): $\gamma_1 = 9.00$ ($\alpha = 0.0027$)
- $m=2$ (NHC): $\gamma_2 = 9.21$ ($\alpha = 0.0100$)
- $m=3$ (ZUPT / ZARU): $\gamma_3 = 11.34$ ($\alpha = 0.0100$)

### 3.5 Refactored Segment-by-Segment Creep Evaluator
- `compute_constraint_metrics` in [`constraint_metrics.py`](file:///d:/python/SIH%2026%20ISRO/Data_details/src/phase6/evaluation/constraint_metrics.py) was refactored to evaluate each continuous stationary segment independently:
  $$C_i = \max_{t \in [t_{\text{start}, i}, t_{\text{end}, i}]} \|\mathbf{p}(t) - \mathbf{p}(t_{\text{start}, i})\|$$
  Reporting maximum, mean, and median segment creep alongside the number of stationary segments.

---

## 4. Remaining Error Analysis: The Need for Road Constraints

### 4.1 Residual Drift in Phase 6
While Phase 6 reduced 60s drift by **4.55x (from 269.1% to 59.1%)**, long-duration outages (120 s) still exhibit significant error:
- Traveled distance: $661.44\text{ m}$
- 120s Endpoint error: $2,003.25\text{ m}$ (Drift: $302.86\%$)
- 120s Yaw error: $63.06^\circ$

### 4.2 Why Vehicle Physics Reaches Its Limit
1. **Unobservable Yaw during Straight Driving**: As proven in [`PHASE_6_OBSERVABILITY.md`](file:///d:/python/SIH%2026%20ISRO/docs/phase6/PHASE_6_OBSERVABILITY.md), the forward velocity model and NHC only align velocity with the currently estimated heading. During straight constant-speed highway driving, unobserved gyroscope bias slowly drifts the estimated heading. NHC faithfully forces the velocity to follow that drifted heading!
2. **Missing Absolute Geometric Reference**: No internal inertial sensor can observe whether a road is oriented at $45^\circ$ or $55^\circ$ azimuth.
3. **The Solution**: An external spatial prior—the **OpenStreetMap road network**—which provides road link azimuths and centerline geometries.

---

## 5. Phase 7 Architecture & Handoff

Phase 7 introduces offline OpenStreetMap (OSM) road-constrained navigation:
1. **Offline Road Network Extraction**: Extracting ways, nodes, geometries, and topological connectivity within the vehicle's operational bounding box.
2. **Local ENU Map Representation**: Transforming WGS84 polylines into the exact local ENU frame used by the ESKF.
3. **Spatial Indexing**: $O(\log N)$ candidate segment lookup via `scipy.spatial.cKDTree` / uniform spatial grid.
4. **Probabilistic Candidate Scoring**: Fusing cross-track distance likelihood, heading alignment compatibility, and road topological transition probabilities.
5. **Soft Kalman Map Constraints**: Formulating cross-track distance and road heading as measurement updates in the ESKF with NIS gating, avoiding hard nearest-road snapping.
6. **Zero Runtime Internet Dependency**: All map matching executes 100% offline from locally cached road geometries.

> **PRE-FLIGHT AUDIT VERDICT**: Phase 6 is frozen, verified, and passing 100% of automated unit tests (123/123 passed). Ready to proceed to Phase 7 implementation.
