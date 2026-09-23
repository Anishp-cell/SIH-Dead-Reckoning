# Pre-Phase-8 Audit: Root-Cause Hierarchy & Scientific Evidence Tree

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/16_ROOT_CAUSE_TREE.md`  

---

## 1. Executive Summary

This document structures all observed phenomena, anomalies, and performance bounds into a formal **Root-Cause Hierarchy Tree**. Each branch is assigned an explicit scientific verdict based on mathematical derivation and experimental evidence:
- **`[CONFIRMED]`**: Formally derived and empirically demonstrated through controlled testing.
- **`[LIKELY]`**: Physically plausible and supported by observed correlations, but lacking direct measurement.
- **`[NOT SUPPORTED]`**: Evaluated and disproven by empirical experiment.
- **`[UNRESOLVED]`**: Outside current phase scope; reserved for Phase 8+ investigation.

---

## 2. Root-Cause Hierarchy Tree

```text
ROOT: Navigation Performance & Anomaly Investigation (Phase 4–7)
│
├── 1.0 SYMPTOM: Longitudinal Position Error Dominates Total Drift at 60s (78.8m total vs 0.48m lateral)
│   ├── 1.1 Unobservability of Along-Track Position by 1D Road Centerline Maps [CONFIRMED]
│   │   ├── Proof: Jacobian H_p * t_road = 0 identically. Along-track position is in the null space of H_p.
│   │   └── Evidence: Cross-track error drops to 0.48m while along-track error remains 73.0m.
│   │
│   ├── 1.2 Phase 4 Speed Model Underestimation During Crawl & Deceleration [CONFIRMED]
│   │   ├── Proof: AI speed under-predicts crawl speeds (< 3 m/s) by ~1.2 m/s average.
│   │   └── Evidence: Mode B 60s distance integral deficit matches along-track error within 1.01m.
│   │
│   ├── 1.3 Feature Column Low-Pass Filtering Mismatch (Pre-Correction) [CONFIRMED]
│   │   ├── Proof: Evaluating on _filtered channels removed 41% of vertical vibration variance.
│   │   └── Evidence: Raw feature priority restored speed estimation fidelity and reduced variance loss.
│   │
│   └── 1.4 Gyroscope Pitch/Roll Tilt Coupling into Longitudinal Acceleration [LIKELY]
│       └── Theory: Unobserved vehicle pitch dynamics during braking bleed gravity into forward axis.
│
├── 2.0 SYMPTOM: Standstill Velocity and Position Explosion in Phase 5 (1557m drift in 60s)
│   ├── 2.1 Double Integration of Unconstrained Accelerometer Bias [CONFIRMED]
│   │   ├── Proof: Double integral of residual 0.04 m/s^2 bias produces quadratic creep during stops.
│   │   └── Evidence: Phase 5 drift exploded to 811.9% during the 35s traffic standstill.
│   │
│   ├── 2.2 Inability of Pure AI Speed to Constrain Standstill Without State Zeros [CONFIRMED]
│   │   ├── Proof: Neural network outputs small residual speeds (0.1 - 0.3 m/s) due to sensor noise floor.
│   │   └── Evidence: Phase 6 ZUPT/ZARU eliminated 95% of this error, dropping drift to 41.1%.
│   │
│   └── 2.3 Hypothesis: Phase 4 AI Speed Outputted Massive Negative Speeds [NOT SUPPORTED]
│       └── Disproof: AI speed output is bounded below by zero (ReLU/clamp in model architecture).
│
├── 3.0 SYMPTOM: 4.85m Endpoint Error Gap at 10s Outage (Cold Start vs Warm Start)
│   ├── 3.1 Rolling Buffer Cold-Start Transient (3-second fill lag) [CONFIRMED]
│   │   ├── Proof: CNN receptive field requires 30 continuous frames. Zero buffer yields ~0 m/s prediction.
│   │   └── Evidence: Mode B (Warm) achieved 16.57m endpoint vs Mode C (Cold) 21.42m (delta = 4.85m).
│   │
│   └── 3.2 Hypothesis: Initial ESKF Covariance Misalignment Caused the 10s Gap [NOT SUPPORTED]
│       └── Disproof: Both Mode B and Mode C used identical initial covariance matrices P_0.
│
├── 4.0 SYMPTOM: Heading Error Injected into Non-Yaw Axes Under Vehicle Tilt
│   ├── 4.1 Heading Jacobian Assumed Flat Level Vehicle (H[0, 8] = 1.0) [CONFIRMED]
│   │   ├── Proof: DCM perturbation shows non-zero pitch coupling H_theta_y = 0.154 at realistic tilt.
│   │   └── Evidence: Exact analytical Jacobian derived and validated vs finite differences (< 1e-9).
│   │
│   └── 4.2 In-Run Z-Gyro Bias Drift Over Extended Outages [CONFIRMED]
│       ├── Proof: Earth rotation rate + MEMS thermal drift accumulate ~0.15 deg/s unconstrained.
│       └── Evidence: Phase 7 road tangent updates confined yaw RMSE from 5.71 deg down to 1.07 deg.
│
├── 5.0 SYMPTOM: Potential Filter Corruption by Off-Road Outliers
│   ├── 5.1 Innovation Clamping Prior to Chi-Square Test (Pre-Correction) [CONFIRMED]
│   │   ├── Proof: Clamping nu to 15m compressed NIS (nu^2/S), allowing 50m outliers to pass.
│   │   └── Evidence: Reordering sequence so NIS is computed on raw nu rejects all gross outliers.
│   │
│   └── 5.2 Decoupling of Map Updates During High-Drift Cruising & Cornering [CONFIRMED]
│       ├── Proof: Adaptive measurement covariance R = R_0 / c_map^2 scales to infinity as c_map -> 0.
│       └── Evidence: Scenario 2 & 3 de-weighted map updates, preventing false road snapping.
│
└── 6.0 REMAINING SYSTEMIC BOUNDARIES (Phase 8+ Scope)
    ├── 6.1 Along-Track Position Resets upon GNSS Signal Reacquisition [UNRESOLVED - PHASE 8 SCOPE]
    │   └── Plan: Phase 8 GNSS fusion will provide absolute position fixes to eliminate accumulated along-track error.
    │
    └── 6.2 Adaptive Heading Drift Under Mismatched Road Curvature [UNRESOLVED - PHASE 8 SCOPE]
        └── Plan: Phase 8 innovation monitoring will dynamically detect multi-lane highway splits and road deviations.
```

---

## 3. Disproven Hypotheses Summary

1. **"Lateral dead-reckoning drift cannot be contained by offline OpenStreetMap"**:
   - **Verdict**: **`[NOT SUPPORTED]`**.
   - **Fact**: Cross-track error was reduced by **$91.3\%$** ($5.49\text{ m} \to 0.48\text{ m}$), demonstrating that offline 2D road centerlines provide near-perfect lateral confinement.

2. **"Exporting the model to ONNX / TorchScript destroys uncertainty predictions"**:
   - **Verdict**: **`[NOT SUPPORTED]`** (after CA-01 remediation).
   - **Fact**: `ExportWrapperDual` preserves both speed and uncertainty with bitwise identical outputs ($0.00\text{ m/s}$ deviation).

3. **"Cold-start buffer lag cannot be causally solved"**:
   - **Verdict**: **`[NOT SUPPORTED]`**.
   - **Fact**: Feeding pre-blackout telemetry ($t \in [t_0 - 30\Delta t, t_0)$) into the rolling buffer is strictly causal ($t \le t_0$) and completely eliminates the $4.85\text{ m}$ startup deficit.

4. **"Map matching alone solves all localization error in GPS blackouts"**:
   - **Verdict**: **`[NOT SUPPORTED]`**.
   - **Fact**: As proven by both mathematical null space derivation and empirical data, 1D road centerlines cannot observe longitudinal (along-track) position. Phase 8 GNSS fusion is required for complete recovery.
