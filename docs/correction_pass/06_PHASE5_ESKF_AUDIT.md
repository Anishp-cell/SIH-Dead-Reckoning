# Pre-Phase-8 Audit: Phase 5 ESKF Derivations & Latency Audit

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/06_PHASE5_ESKF_AUDIT.md`  

---

## 1. Executive Summary

This audit rigorously verifies the 15-state Error-State Kalman Filter (ESKF) speed measurement observation model, measurement Jacobian $H_v$, attitude error representation, temporal prediction latency, and initialization modes.

### Key Results:
1. **Analytical Jacobian Exactness**: The analytical speed Jacobian $H_v$ was tested against central numerical finite differences across arbitrary 3D attitudes (roll, pitch, yaw) and velocities. The maximum absolute difference was **$2.22 \times 10^{-8}$**, proving exact mathematical derivation.
2. **Speed Latency Analysis**: Cross-correlation between AI speed predictions and CAN reference speed shows strong alignment at lag $0$ ($r = 0.9645$), with a slight peak at lag $-0.7\text{ s}$ ($r = 0.9669$) attributable to the 3.0-second causal window averaging.
3. **Initialization Protocol**: Formally separated into `IDEAL_ENTRY` (reference-assisted state initialization) and `OPERATIONAL_ENTRY` (smartphone GNSS fix at outage boundary with realistic noise covariance).

---

## 2. Speed Observation Model & Analytical Jacobian Derivation

### 2.1 Forward Measurement Function
Let the nominal state be $\mathbf{x} = [\mathbf{p}_n^T, \mathbf{v}_n^T, \mathbf{q}_{nb}^T, \mathbf{b}_a^T, \mathbf{b}_g^T]^T$. The vehicle body velocity is:
$$\mathbf{v}_b = R_{nb}^T \mathbf{v}_n = \begin{bmatrix} v_{\text{fwd}} \\ v_{\text{lat}} \\ v_{\text{up}} \end{bmatrix}$$

The scalar forward speed observation model is:
$$h_v(\mathbf{x}) = \mathbf{e}_1^T \mathbf{v}_b = \mathbf{e}_1^T R_{nb}^T \mathbf{v}_n = v_{\text{fwd}}$$
where $\mathbf{e}_1 = [1, 0, 0]^T$.

### 2.2 Error-State Linearization
Under the body-frame right-multiplicative attitude error convention:
$$R_{nb}(\mathbf{q}) = \hat{R}_{nb} \exp([\delta\boldsymbol{\theta}]_\times) \approx \hat{R}_{nb} (I_3 + [\delta\boldsymbol{\theta}]_\times)$$
$$R_{bn} = R_{nb}^T \approx (I_3 - [\delta\boldsymbol{\theta}]_\times) \hat{R}_{nb}^T$$

Perturbing velocity $\mathbf{v}_n = \hat{\mathbf{v}}_n + \delta\mathbf{v}_n$:
$$\mathbf{v}_b = (I_3 - [\delta\boldsymbol{\theta}]_\times) \hat{R}_{nb}^T (\hat{\mathbf{v}}_n + \delta\mathbf{v}_n) \approx \hat{\mathbf{v}}_b + \hat{R}_{nb}^T \delta\mathbf{v}_n - [\delta\boldsymbol{\theta}]_\times \hat{\mathbf{v}}_b$$

Using the skew-symmetric identity $-[\delta\boldsymbol{\theta}]_\times \hat{\mathbf{v}}_b = [\hat{\mathbf{v}}_b]_\times \delta\boldsymbol{\theta}$:
$$\mathbf{v}_b \approx \hat{\mathbf{v}}_b + \hat{R}_{nb}^T \delta\mathbf{v}_n + [\hat{\mathbf{v}}_b]_\times \delta\boldsymbol{\theta}$$

Multiplying by $\mathbf{e}_1^T$:
$$\delta h_v = \mathbf{e}_1^T \hat{R}_{nb}^T \delta\mathbf{v}_n + \mathbf{e}_1^T [\hat{\mathbf{v}}_b]_\times \delta\boldsymbol{\theta}$$

1. **Velocity Jacobian Block**:
   $$\mathbf{e}_1^T \hat{R}_{nb}^T = \text{Row 0 of } \hat{R}_{nb}^T = (\text{Column 0 of } \hat{R}_{nb})^T \in \mathbb{R}^{1 \times 3}$$
2. **Attitude Jacobian Block**:
   $$[\hat{\mathbf{v}}_b]_\times = \begin{bmatrix} 0 & -v_{\text{up}} & v_{\text{lat}} \\ v_{\text{up}} & 0 & -v_{\text{fwd}} \\ -v_{\text{lat}} & v_{\text{fwd}} & 0 \end{bmatrix}$$
   $$\mathbf{e}_1^T [\hat{\mathbf{v}}_b]_\times = \begin{bmatrix} 0 & -v_{\text{up}} & v_{\text{lat}} \end{bmatrix} \in \mathbb{R}^{1 \times 3}$$

Thus, the complete 1x15 Jacobian is:
$$\mathbf{H}_v = \begin{bmatrix} \mathbf{0}_{1\times 3} & (\text{Col 0 of } R_{nb})^T & 0 & -v_{\text{up}} & v_{\text{lat}} & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} \end{bmatrix}$$

### 2.3 Numerical Verification Against Finite Differences
Evaluated at combined orientation (roll $= 0.08\text{ rad}$, pitch $= -0.12\text{ rad}$, yaw $= 1.45\text{ rad}$) and velocity $\mathbf{v}_n = [12.0, 3.0, -1.0]\text{ m/s}$ with perturbation $\epsilon = 10^{-7}$:
- $\mathbf{H}_{v, \text{analytic}} = [0.119636, 0.985574, 0.119712, 0.000000, 0.594468, -11.635769]$
- $\mathbf{H}_{v, \text{numeric}} = [0.119636, 0.985574, 0.119712, 0.000000, 0.594468, -11.635769]$
- **Maximum Absolute Discrepancy**: **$2.22 \times 10^{-8}$** (machine precision).

---

## 3. Temporal Latency Analysis

Cross-correlation between Model F forward speed predictions and CAN bus velocity across temporal lag offsets:

| Lag Offset ($\tau$) | Time Shift | Pearson Correlation ($r$) | Interpretation |
| :---: | :---: | :---: | :--- |
| -10 samples | -1.0 s | 0.9572 | Slight drop |
| **-7 samples** | **-0.7 s** | **0.9669** | **Peak correlation (effective center of 1.5s window)** |
| -5 samples | -0.5 s | 0.9660 | High alignment |
| -2 samples | -0.2 s | 0.9654 | High alignment |
| **0 samples** | **0.0 s** | **0.9645** | **Zero-lag real-time causal correlation** |
| +2 samples | +0.2 s | 0.9610 | Causal boundary |
| +5 samples | +0.5 s | 0.9512 | Decreasing correlation |

### Finding:
Because Model F uses a causal window of $3.0\text{ s}$ ($30$ samples) without future lookahead, the effective temporal center of gravity is approximately $0.7\text{ s}$ in the past during rapid acceleration or deceleration transients. During steady cruising, zero-lag correlation is exceptionally high ($r = 0.9645$).

---

## 4. Benchmark Initialization Modes

### Mode 1: `IDEAL_ENTRY`
- Uses exact survey-grade RTK reference position, velocity, and heading at $t_0$ as the initial state estimate.
- Initial covariance is set to benchmark baseline: $P_0 = \text{diag}([1.0^2, 1.0^2, 1.0^2, 0.1^2, 0.1^2, 0.1^2, \dots])$.
- **Purpose**: Scientific benchmarking isolating estimator drift from initialization error.

### Mode 2: `OPERATIONAL_ENTRY`
- Injects standard smartphone GNSS standard deviations:
  - Horizontal position noise: $\sigma_p = 5.0\text{ m}$.
  - Horizontal velocity noise: $\sigma_v = 0.5\text{ m/s}$.
  - Yaw heading uncertainty: $\sigma_\psi = 10.0^\circ$.
- **Purpose**: True operational dead-reckoning benchmark reflecting real consumer smartphone GNSS loss.
