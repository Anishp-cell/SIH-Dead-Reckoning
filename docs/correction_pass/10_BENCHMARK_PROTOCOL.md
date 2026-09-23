# Pre-Phase-8 Audit: Standardized Multi-Mode Benchmark Protocol

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/10_BENCHMARK_PROTOCOL.md`  

---

## 1. Executive Summary & Protocol Separation

To eliminate the ambiguities and artificial transients identified during the audit, this protocol formally defines **three standardized benchmark modes**. All future evaluations across Phases 4 through 8 must report results labeled by these explicit modes.

---

## 2. Benchmark Mode Definitions

### Benchmark Mode A: `WARM_START_OPERATIONAL` (Primary Operational Benchmark)
- **Causal History**: The AI inference rolling buffer is pre-warmed using $W=30$ historical IMU frames strictly before the outage onset: $t \in [t_0 - 30\Delta t, t_0)$.
- **Causality Constraint**: $\max(t_{\text{pre}}) < t_0$. No future or contemporaneous samples are accessed.
- **Initialization State**: State is initialized using realistic consumer smartphone GNSS noise:
  - Position uncertainty: $\sigma_p = 5.0\text{ m}$.
  - Velocity uncertainty: $\sigma_v = 0.5\text{ m/s}$.
  - Yaw heading uncertainty: $\sigma_\psi = 10.0^\circ$.
- **Significance**: Represents the true operational environment where an active navigation app suddenly experiences a GNSS outage (e.g. entering an underground tunnel).

### Benchmark Mode B: `WARM_START_IDEAL` (Controlled Estimator Benchmark)
- **Causal History**: Identical warm pre-buffering as Mode A ($W=30$ historical samples).
- **Initialization State**: Exact survey-grade RTK GNSS position, velocity, and dual-antenna heading reference at $t_0$.
- **Significance**: Isolates dead-reckoning estimator and constraint drift from initialization errors.

### Benchmark Mode C: `COLD_START` (Robustness Stress Test)
- **Causal History**: The AI rolling buffer is cleared to all zeros at $t_0$.
- **Initialization State**: Exact survey-grade reference at $t_0$.
- **Significance**: Stress-tests recovery dynamics when an app is launched cold inside a tunnel with zero prior sensor history.

---

## 3. Mandatory Metric Specifications

For every benchmark test run, the following 15 metrics must be logged and reported:

| Metric Category | Metric Symbol / Name | Calculation Definition | Physical Units |
| :--- | :--- | :--- | :---: |
| **Endpoint Error** | $e_{\text{end}}$ | $\|\hat{\mathbf{p}}(T) - \mathbf{p}_{\text{truth}}(T)\|_2$ | Meters ($\text{m}$) |
| **Trajectory Drift %** | $\text{Drift } \%$ | $(e_{\text{end}} / \int_0^T v_{\text{CAN}}(t) dt) \times 100\%$ | Percentage ($\%$) |
| **2D Position RMSE** | $\text{RMSE}_{\text{pos}}$ | $\sqrt{\frac{1}{N}\sum_{k=1}^N \|\hat{\mathbf{p}}_{2D, k} - \mathbf{p}_{\text{truth}, 2D, k}\|^2}$ | Meters ($\text{m}$) |
| **Cross-Track RMSE** | $\text{RMSE}_{\text{cross}}$ | $\sqrt{\frac{1}{N}\sum_{k=1}^N d_{\perp, k}^2}$ relative to true road/trajectory centerline | Meters ($\text{m}$) |
| **Along-Track RMSE** | $\text{RMSE}_{\text{along}}$ | $\sqrt{\frac{1}{N}\sum_{k=1}^N s_{\parallel, k}^2}$ longitudinal error along travel heading | Meters ($\text{m}$) |
| **Velocity RMSE** | $\text{RMSE}_{\text{vel}}$ | $\sqrt{\frac{1}{N}\sum_{k=1}^N \|\hat{\mathbf{v}}_k - \mathbf{v}_{\text{truth}, k}\|^2}$ | $\text{m/s}$ |
| **Forward Velocity RMSE** | $\text{RMSE}_{v_{\text{fwd}}}$ | $\sqrt{\frac{1}{N}\sum_{k=1}^N (\hat{v}_{\text{fwd}, k} - v_{\text{CAN}, k})^2}$ | $\text{m/s}$ |
| **Forward Velocity Bias** | $\text{Bias}_{v_{\text{fwd}}}$ | $\frac{1}{N}\sum_{k=1}^N (\hat{v}_{\text{fwd}, k} - v_{\text{CAN}, k})$ | $\text{m/s}$ |
| **Final Yaw Error** | $|\Delta\psi_{\text{end}}|$ | $|\text{wrap}_\pi(\hat{\psi}(T) - \psi_{\text{truth}}(T))|$ | Degrees ($^\circ$) |
| **Yaw RMSE** | $\text{RMSE}_\psi$ | $\sqrt{\frac{1}{N}\sum_{k=1}^N (\text{wrap}_\pi(\hat{\psi}_k - \psi_{\text{truth}, k}))^2}$ | Degrees ($^\circ$) |
| **Stationary Creep** | $\Delta p_{\text{stat}}$ | $\|\hat{\mathbf{p}}(t_{\text{stop\_end}}) - \hat{\mathbf{p}}(t_{\text{stop\_start}})\|_2$ during validated standstills | Meters ($\text{m}$) |
| **Normalized Innovation** | $\overline{\text{NIS}}$ | Mean test statistic $d_M^2 = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu}$ across accepted steps | Dimensionless |
| **Rejection Rate** | $\text{Rej } \%$ | $(N_{\text{rejected}} / N_{\text{total}}) \times 100\%$ | Percentage ($\%$) |
| **Mean Map Confidence** | $\overline{c_{\text{map}}}$ | Average continuous confidence score $[0, 1]$ | Dimensionless |
| **Step CPU Latency** | $\overline{T_{\text{step}}}$ | Mean end-to-end execution time per $10\text{ Hz}$ step | Milliseconds ($\text{ms}$) |

---

## 4. Evaluated Durations & Driving Scenarios

### Durations:
- **10 Seconds**: Quick underpass / overpass outage.
- **30 Seconds**: Moderate urban canyon / bridge outage.
- **60 Seconds**: Standard highway underpass / urban tunnel blackout (includes red light stop).
- **120 Seconds**: Extended multi-kilometer tunnel blackout.

### Operational Scenarios:
1. **Stop-and-Go Traffic**: Standstill followed by moderate acceleration.
2. **Steady Cruising**: High-speed arterial road cruising ($35 - 50\text{ km/h}$).
3. **Sharp Cornering Turn**: $90^\circ$ turn across a multi-branch urban intersection.
4. **Long Straight Highway**: Extended straight corridor testing unobserved yaw drift.
