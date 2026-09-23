# Phase 6 Noise Model, Covariance Propagation, & Tuning Parameters

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_NOISE_AND_COVARIANCE.md`  

---

## 1. Process Noise Model (Retained from Phase 5)

The 15-state indirect error state propagation utilizes the validated continuous process noise covariance $Q_c \in \mathbb{R}^{12\times 12}$:
$$Q_c = \text{diag}\left(\sigma_a^2 I_3, \sigma_g^2 I_3, \sigma_{ba}^2 I_3, \sigma_{bg}^2 I_3\right)$$

Empirical sensor noise parameters from Allan variance characterization:
- **Accelerometer White Noise**: $\sigma_a = 0.08\text{ m/s}^2/\sqrt{\text{Hz}}$
- **Gyroscope White Noise**: $\sigma_g = 0.005\text{ rad/s}/\sqrt{\text{Hz}}$
- **Accelerometer Bias Random Walk**: $\sigma_{ba} = 0.0005\text{ m/s}^3/\sqrt{\text{Hz}}$
- **Gyroscope Bias Random Walk**: $\sigma_{bg} = 0.00005\text{ rad/s}^2/\sqrt{\text{Hz}}$

Discrete-time propagation over $\Delta t = 0.1\text{ s}$ via Van Loan / first-order mapping:
$$Q_k \approx G_k Q_c G_k^T \Delta t$$
where:
$$G_k = \begin{bmatrix}
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \\
-R_{nb} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & -I_3 & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & I_3 & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & I_3
\end{bmatrix} \in \mathbb{R}^{15\times 12}$$

---

## 2. Phase 6 Measurement Covariance Matrix Suite

| Constraint | State Dimension $m$ | Nominal Covariance Formula | Nominal Parameter Values | Physical Justification |
|:-----------|:-------------------:|:---------------------------|:-------------------------|:-----------------------|
| **Phase 4 Speed** | 1 | $R_v = \max(\sigma_v^2, 10^{-4})$ | Predicted by Heteroscedastic TCN ($\approx 0.3 - 1.2\text{ m/s}$) | Matches dynamic AI prediction confidence |
| **NHC** | 2 | $\text{diag}(\sigma_{\text{lat}}^2(t), \sigma_{\text{up}}^2(t))$ | Baseline: $\sigma_{\text{lat},0} = 0.20\text{ m/s}, \sigma_{\text{up},0} = 0.20\text{ m/s}$ | Reflects tire compliance and road grade variations |
| **ZUPT** | 3 | $R_{\text{zupt}} = \sigma_{\text{zupt}}^2 I_3$ | $\sigma_{\text{zupt}} = 0.05\text{ m/s}$ | Clamps velocity tightly to zero during verified stop |
| **ZARU** | 3 | $R_{\text{zaru}} = \sigma_{\text{zaru}}^2 I_3$ | $\sigma_{\text{zaru}} = 0.005\text{ rad/s}$ | Reflects sensor noise floor during stationary idling |

---

## 3. Innovation Gating (Chi-Square Test)

To prevent spurious measurements from corrupting state estimates, every measurement is subjected to a statistical Normalized Innovation Squared (NIS) gating test:
$$\text{NIS} = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu} \le \gamma_m$$
where $S = H P^- H^T + R \in \mathbb{R}^{m\times m}$.

Critical thresholds:
- $m=1$ (AI Speed): $\gamma_1 = 9.0$ ($\approx 3\sigma$, 99.73% confidence)
- $m=2$ (NHC): $\gamma_2 = 9.21$ ($\chi^2_{0.99}(2)$ 99.0% confidence)
- $m=3$ (ZUPT / ZARU): $\gamma_3 = 11.34$ ($\chi^2_{0.99}(3)$ 99.0% confidence)
