# Phase 8 Architecture Audit & System Integration Specification

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 8 — GNSS + INS Fusion, Outage Detection, and Seamless Recovery  
**Document**: `docs/phase8/00_PHASE8_ARCHITECTURE_AUDIT.md`  

---

## 1. Executive Summary

This architecture audit establishes the exact execution graph, state representation, measurement sequencing, data structures, and coordinate standards for Phase 8. Phase 8 integrates GNSS as a probabilistic, covariance-aware absolute position and velocity reference into the frozen Phase 7 navigation stack.

### Core Architectural Principles:
1. **Preservation of the Frozen Baseline**: Phase 5 (15-state ESKF), Phase 6 (Vehicle-physics NHC/ZUPT/ZARU), and Phase 7 (Offline OSM map constraints) remain active and functionally uninhibited.
2. **Unified Navigation State**: GNSS does not maintain an independent parallel estimator. Instead, GNSS position and velocity updates directly correct the unified 15-state error vector $\delta\mathbf{x} = [\delta\mathbf{p}, \delta\mathbf{v}, \delta\boldsymbol{\theta}, \delta\mathbf{b}_a, \delta\mathbf{b}_g]^T$ via Joseph-form stabilized indirect Kalman updates.
3. **Causal Outage State Machine**: An explicit 4-state automaton (`GNSS_HEALTHY`, `GNSS_SUSPECT`, `GNSS_OUTAGE`, `GNSS_RECOVERING`) governs measurement gating with strict temporal hysteresis.
4. **Seamless Covariance-Aware Recovery**: When GNSS returns after an outage, the estimator transitions smoothly without state teleportation, position snaps, or covariance collapse.

---

## 2. Phase 7 to Phase 8 Execution Graph & Call Flow

```
                                [ Raw Telemetry Stream ]
                                            │
               ┌────────────────────────────┴───────────────────────────┐
               ▼ (100 Hz / 10 Hz)                                       ▼ (1-10 Hz)
     [ IMU Accelerometer & Gyro ]                              [ GNSS Receiver Telemetry ]
               │                                                        │
               ▼                                                        ▼
┌───────────────────────────────┐                             ┌───────────────────────────┐
│ Phase 5 Nominal Propagation   │                             │ Phase 8 GNSS Ingestion    │
│ - Shepperd Quaternion DCM     │                             │ - WGS84 Geodetic -> ENU   │
│ - Gravity Vector Compensation │                             │ - Velocity Vector E/N/U   │
│ - Error Covariance Prop (Q_k) │                             │ - Quality Parsing         │
└──────────────┬────────────────┘                             └─────────────┬─────────────┘
               │                                                            │
               ▼                                                            ▼
┌───────────────────────────────┐                             ┌───────────────────────────┐
│ Phase 4 AI Speed Estimation   │                             │ Phase 8 Quality & Gating  │
│ - Causal FIFO Buffer (W=30)   │                             │ - Statistical Jump Filter │
│ - Heteroscedastic Speed & Var │                             │ - 3-DOF Chi-Square NIS    │
│ - Kalman Update (H_v, R_v)    │                             │ - ACCEPT / DOWNWEIGHT     │
└──────────────┬────────────────┘                             └─────────────┬─────────────┘
               │                                                            │
               ▼                                                            ▼
┌───────────────────────────────┐                             ┌───────────────────────────┐
│ Phase 6 Vehicle Constraints   │                             │ Phase 8 Outage Automaton  │
│ - Disturbance Detector        │                             │ - HEALTHY / SUSPECT       │
│ - NHC (v_lat = 0, v_up = 0)   │                             │ - OUTAGE / RECOVERING     │
│ - ZUPT (v = 0) & ZARU (w = 0) │                             │ - Hysteresis Persistence  │
└──────────────┬────────────────┘                             └─────────────┬─────────────┘
               │                                                            │
               ▼                                                            │
┌───────────────────────────────┐                                           │
│ Phase 7 Offline OSM Matcher   │                                           │
│ - 1st-Order Markov Topology   │                                           │
│ - Soft Cross-Track Normal Upd │                                           │
│ - Road Tangent Yaw Update     │                                           │
└──────────────┬────────────────┘                                           │
               │                                                            │
               ▼                                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 8 GNSS/INS Fusion & Seamless Recovery Engine                                      │
│ - Active when State in {HEALTHY, RECOVERING}                                            │
│ - Position Measurement: z_p = p + R_nb*l_b (H_p, R_p)                                   │
│ - Velocity Measurement: z_v = v + R_nb*(omega_b x l_b) (H_v, R_v)                       │
│ - Soft Recovery Innovation Shaping & Continuity Monitoring                              │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Output Navigation State   │
                               │ [p, v, q, ba, bg] & P_15  │
                               └───────────────────────────┘
```

---

## 3. Pipeline Sequencing & Update Precedence

The execution order within each discrete epoch $t_k$ is mathematically derived to ensure causal, non-conflicting constraint enforcement:

1. **Step 1: High-Rate IMU Propagation ($100\text{ Hz}$)**:
   - State propagation: $\mathbf{p}_{k|k-1}, \mathbf{v}_{k|k-1}, \mathbf{q}_{k|k-1}$ using matrix exponential.
   - Covariance propagation: $\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k$.
2. **Step 2: Phase 4 AI Forward Speed Update ($10\text{ Hz}$)**:
   - Updates forward body velocity: $\mathbf{z}_{\text{speed}} = v_{\text{AI}}$.
   - Constrains longitudinal velocity before physical constraints.
3. **Step 3: Phase 6 Physical Constraints**:
   - Disturbance evaluation: computes lateral and vertical variance inflation $\mathbf{R}_{\text{NHC}}(c_{\text{NHC}})$.
   - NHC update: lateral and vertical velocity zero-updates ($v_{\text{lat}} = 0, v_{\text{up}} = 0$).
   - Stationary detection: if standstill verified, executes ZUPT ($\mathbf{v} = \mathbf{0}$) and ZARU ($\boldsymbol{\omega} = \mathbf{0}$).
4. **Step 4: Phase 7 Offline OSM Map Updates**:
   - Cross-track normal position constraint: $\mathbf{z}_{\perp} = 0, h_{\perp}(\mathbf{x}) = d_{\text{cross}}$.
   - Road tangent heading constraint: $\nu_\psi = \text{wrap}(\psi_{\text{road}} - \psi_{\text{veh}})$.
   - Modulated by topological belief confidence $c_{\text{map}}$.
5. **Step 5: Phase 8 GNSS Quality & State Machine Update**:
   - Parses GNSS packet; converts geodetic coordinates to local ENU.
   - Evaluates quality indicators (satellite count, horizontal accuracy, velocity rate-of-change).
   - Computes 3-DOF position and velocity NIS against prior covariance $\mathbf{P}$.
   - Advances outage state machine (`HEALTHY`, `SUSPECT`, `OUTAGE`, `RECOVERING`).
6. **Step 6: Phase 8 GNSS Fusion Update (Conditional)**:
   - If state is `HEALTHY` or `RECOVERING`:
     - Executes 3D position Kalman update ($\mathbf{H}_p, \mathbf{R}_p$).
     - Executes 3D velocity Kalman update ($\mathbf{H}_v, \mathbf{R}_v$).
     - Applies Joseph-form stabilized covariance update.
     - Logs recovery continuity metrics.
   - If state is `OUTAGE` or `SUSPECT` (rejected):
     - GNSS measurements are strictly suppressed; state and covariance continue undisturbed under Phase 7 dead reckoning.

---

## 4. State Vector & Mathematical Standards

### Nominal State Vector (16D):
$$\mathbf{x} = \begin{bmatrix} \mathbf{p}^n \\ \mathbf{v}^n \\ \mathbf{q}_{nb} \\ \mathbf{b}_a \\ \mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{16}$$
- $\mathbf{p}^n = [p_E, p_N, p_U]^T$: Cartesian position in local ENU frame (meters).
- $\mathbf{v}^n = [v_E, v_N, v_U]^T$: Cartesian velocity in local ENU frame (m/s).
- $\mathbf{q}_{nb} = [q_w, q_x, q_y, q_z]^T$: Unit quaternion representing orientation from body frame to navigation frame. Scalar-first convention ($\|\mathbf{q}\| = 1$).
- $\mathbf{b}_a = [b_{ax}, b_{ay}, b_{az}]^T$: Accelerometer bias in body frame (m/s$^2$).
- $\mathbf{b}_g = [b_{gx}, b_{gy}, b_{gz}]^T$: Gyroscope bias in body frame (rad/s).

### Error State Vector (15D):
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$
- $\delta\boldsymbol{\theta} = [\delta\theta_x, \delta\theta_y, \delta\theta_z]^T$: Small-angle rotation error vector under right-multiplicative body-frame attitude error convention:
  $$\mathbf{R}_{nb}(\delta\boldsymbol{\theta}) \approx \hat{\mathbf{R}}_{nb} (\mathbf{I}_3 + [\delta\boldsymbol{\theta}]_\times)$$
- Error injection into quaternion:
  $$\mathbf{q}^+ = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix}$$

---

## 5. Coordinate Conventions & Provenance Standard

1. **Local Navigation Frame (ENU)**:
   - Origin: Tangent plane origin $(52.401660^\circ\text{ N}, -1.505290^\circ\text{ E}, 147.50\text{ m})$.
   - X-axis: East (+E)
   - Y-axis: North (+N)
   - Z-axis: Up (+U)
2. **Heading & Yaw Standards**:
   - Geographic Azimuth ($\psi_{\text{GPS}}$): Clockwise from True North $[0, 360^\circ)$.
   - Cartesian ENU Yaw ($\psi_{\text{ENU}}$): Counter-clockwise from East $(-\pi, +\pi]$.
   - Authoritative conversion:
     $$\psi_{\text{ENU}} = \frac{\pi}{2} - \psi_{\text{GPS}}$$
3. **Data Provenance Categories**:
   - `REAL_GNSS`: Actual smartphone receiver telemetry from `s1_filtered_causal_imu.csv`.
   - `SYNTHETIC_GNSS`: Controlled synthetic GNSS generated from reference truth with explicit Gaussian noise, multipath, and outage intervals.
   - `REFERENCE_ONLY`: High-precision VBOX/RTK truth from `V-S1.csv` used solely for evaluation and ground-truth metrics.

---

## 6. Files Requiring Phase 8 Implementation

| Path | Module | Responsibility |
|---|---|---|
| `Data_details/src/phase8/gnss/coordinate.py` | Geodetics | Authoritative WGS84 $\to$ ECEF $\to$ ENU and heading conversions. |
| `Data_details/src/phase8/gnss/quality.py` | Quality Engine | Signal validation, jump detection, structured `GNSSQualityReport`. |
| `Data_details/src/phase8/gnss/gating.py` | Innovation Gating | 3-DOF Chi-square NIS gating on unclipped raw innovation. |
| `Data_details/src/phase8/gnss/state_machine.py` | Outage Automaton | 4-state causal state machine with temporal hysteresis. |
| `Data_details/src/phase8/gnss/synthetic.py` | Outage Generator | Reproducible synthetic GNSS generation with controlled noise and outages. |
| `Data_details/src/phase8/fusion/gnss_measurement.py` | Observation Model | Position & velocity observation models, analytical Jacobians ($H_p, H_v$, lever-arm). |
| `Data_details/src/phase8/fusion/gnss_update.py` | Kalman Updates | Joseph-form stabilized indirect Kalman updates for position and velocity. |
| `Data_details/src/phase8/fusion/recovery.py` | Seamless Recovery | Soft covariance-aware recovery, zero-teleportation enforcement, continuity metrics. |
| `Data_details/src/phase8/core/phase8_eskf.py` | Core Estimator | Extends `Phase6ESKF` with GNSS update methods and diagnostic logging. |
| `Data_details/src/phase8/streaming.py` | Streaming Pipeline | Full real-time pipeline orchestrating IMU, AI speed, NHC, Map, and GNSS. |
| `Data_details/src/phase8/evaluation/blackout_benchmarks.py` | Benchmarks | Multi-duration and multi-scenario benchmark evaluation. |
| `Data_details/src/phase8/evaluation/recovery_metrics.py` | Metrics | Quantitative calculation of position/velocity/heading continuity and settling time. |
| `Data_details/src/phase8/visualization/research_replay.py` | Visualizer | Interactive HTML5 replay with offline cartographic OSM basemap. |
| `Data_details/tests/phase8/` | Test Suite | Comprehensive unit, numerical Jacobian, quality, state machine, and recovery tests. |

---

## 7. Architecture Audit Certification

The Phase 8 architecture respects all frozen mathematical conventions, guarantees causality ($t \le t_k$), introduces zero hard trajectory resets, and provides full empirical observability for the previously unconstrained along-track longitudinal error.
