# Phase 4 Feature Leakage & Causality Audit

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document Code**: `SIH26168-AUDIT-LEAKAGE-01`  
**Classification**: RESEARCH COMPLIANCE AUDIT  

---

## 1. Audit Purpose & Scope
This audit rigorously examines all 12 input features consumed by the Phase 4 AI Motion Intelligence models to guarantee that:
1. **Zero Future Sample Leakage**: Every filter and computation uses strictly backward-looking or causal formulas ($t \le t_k$). No future samples ($t > t_k$), centered windowing, or two-pass filters (such as `filtfilt`) are permitted.
2. **Zero Ground-Truth Contamination**: No ground truth sensors (CAN wheel speed, vehicle ECU, Racelogic VBOX DGPS, smartphone GPS position/velocity) enter the model input features.
3. **Hardware & Edge Deployability**: Every input feature can be computed in real-time from a streaming smartphone or external IMU at $10\text{ Hz}$.

---

## 2. Comprehensive 12-Feature Provenance Matrix

| # | Feature Name | Source Sensor | Mathematical Formulation | Uses Future ($t > t_k$)? | Uses GPS? | Uses CAN? | Uses VBOX? | Deployable at Edge? | Audit Verdict |
|---|--------------|---------------|--------------------------|--------------------------|-----------|-----------|------------|---------------------|---------------|
| 1 | `acc_fwd_veh` | Smartphone Accel | Causal Butterworth low-pass ($1.5\text{ Hz}$) on $\mathbf{R}_v^p \mathbf{a}_{\text{phone}}$ | **NO** (IIR Direct Form II) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 2 | `acc_lat_veh` | Smartphone Accel | Causal Butterworth low-pass ($1.5\text{ Hz}$) on $\mathbf{R}_v^p \mathbf{a}_{\text{phone}}$ | **NO** (IIR Direct Form II) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 3 | `acc_up_veh` | Smartphone Accel | Causal Butterworth low-pass ($1.5\text{ Hz}$) on $\mathbf{R}_v^p \mathbf{a}_{\text{phone}}$ | **NO** (IIR Direct Form II) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 4 | `gyro_roll_veh` | Smartphone Gyro | Causal Butterworth low-pass ($2.0\text{ Hz}$) on $\mathbf{R}_v^p (\boldsymbol{\omega}_{\text{phone}} - \mathbf{b}_g)$ | **NO** (IIR Direct Form II) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 5 | `gyro_pitch_veh` | Smartphone Gyro | Causal Butterworth low-pass ($2.0\text{ Hz}$) on $\mathbf{R}_v^p (\boldsymbol{\omega}_{\text{phone}} - \mathbf{b}_g)$ | **NO** (IIR Direct Form II) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 6 | `gyro_yaw_veh` | Smartphone Gyro | Causal Butterworth low-pass ($2.0\text{ Hz}$) on $\mathbf{R}_v^p (\boldsymbol{\omega}_{\text{phone}} - \mathbf{b}_g)$ | **NO** (IIR Direct Form II) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 7 | `jerk_fwd` | Derived Accel | Backward difference: $\frac{a_{\text{fwd}}[k] - a_{\text{fwd}}[k-1]}{\Delta t_k}$ | **NO** (Backward step only) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 8 | `acc_horiz_norm` | Derived Accel | Magnitude: $\sqrt{a_{\text{fwd}}[k]^2 + a_{\text{lat}}[k]^2}$ | **NO** (Instantaneous) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 9 | `gyro_norm` | Derived Gyro | Magnitude: $\sqrt{\omega_{\text{roll}}[k]^2 + \omega_{\text{pitch}}[k]^2 + \omega_{\text{yaw}}[k]^2}$ | **NO** (Instantaneous) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 10 | `ori_pitch_deg` | Phone-Veh Frame | Vehicle body pitch angle relative to horizontal | **NO** (Complementary filter) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 11 | `vibration_energy` | Derived IMU | High-frequency power: $(a_{z, \text{raw}}[k] - a_{z, \text{up}}[k])^2$ | **NO** (Instantaneous) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |
| 12 | `is_stationary` | Streaming ZUPT | Likelihood test on past $0.5\text{ s}$ variance | **NO** (Causal rolling window) | **NO** | **NO** | **NO** | **YES** | **CLEAN** |

---

## 3. Detailed Verification of Critical Features

### Feature 7: Longitudinal Jerk (`jerk_fwd`)
- **Code implementation**:
  ```python
  jerk_fwd = (acc_fwd[k] - acc_fwd[k - 1]) / dt
  ```
- **Audit**: Uses only sample $k$ and historical sample $k-1$. No forward differentiation is used.

### Feature 11: Vibration Energy (`vibration_energy`)
- **Code implementation**:
  ```python
  vibration_energy = (raw_acc_z[k] - filtered_acc_up[k]) ** 2
  ```
- **Audit**: Computed instantaneously at sample $k$ by subtracting the causally filtered acceleration from the raw measurement. Quantifies high-frequency chassis vibration without future context.

### Feature 12: Stationary Indicator (`is_stationary`)
- **Code implementation**:
  ```python
  # Causal Generalized Likelihood Ratio / variance test over past window [k-W+1 : k]
  var_a = np.var(acc_mag[max(0, k - 5) : k + 1])
  is_stat = 1.0 if (var_a < threshold and acc_mag[k] < static_thresh) else 0.0
  ```
- **Audit**: Computed using strictly historical samples ($[k-5 : k]$). Does NOT use vehicle speedometer, wheel sensors, or GPS speed.

---

## 4. Normalization Statistics & Data Partition Isolation

### Audit Check: Preprocessing Pipeline
1. **Mean ($\boldsymbol{\mu}$) and Standard Deviation ($\boldsymbol{\sigma}$)**:
   - Must be computed **strictly from the training partition** ($N_{\text{train}}$ windows).
   - Validation and test sets must be transformed using the frozen training statistics:
     $$\mathbf{X}_{\text{val}}^{\text{norm}} = \frac{\mathbf{X}_{\text{val}} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
   - **Audit Result**: Verified. Preprocessing code does not re-compute scaler statistics on test or validation splits.

2. **Purge Gap Between Splits**:
   - Training: Windows spanning samples $[0, K_1]$.
   - Purge Gap: Samples $[K_1 + 1, K_1 + 50]$ completely discarded.
   - Validation: Windows spanning samples $[K_1 + 51, K_2]$.
   - Purge Gap: Samples $[K_2 + 1, K_2 + 50]$ completely discarded.
   - Test: Windows spanning samples $[K_2 + 51, N]$.
   - **Audit Result**: Zero overlapping raw timesteps exist between Train, Val, and Test.

---

## 5. Formal Certification
We certify that all 12 input features in Phase 4 are:
1. **100% Causal** (dependent only on $t \le t_k$).
2. **100% Free of Ground-Truth Contamination** (no GPS, CAN, or VBOX data in inputs).
3. **100% Free of Data Leakage across Temporal Partitions**.
4. Fully deployable on an edge device (smartphone or embedded processor) receiving an online IMU stream.
