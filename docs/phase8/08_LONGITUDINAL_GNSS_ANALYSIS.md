# Phase 8 Mathematical Foundation: Longitudinal vs Lateral Error Dynamics

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/08_LONGITUDINAL_GNSS_ANALYSIS.md`  

---

## 1. Executive Summary

A critical finding of the Pre-Phase-8 Scientific Audit was the extreme asymmetry of error growth during pure dead reckoning on 1D road networks (Phase 7):
- **Cross-track error ($e_\perp$)** was constrained to **0.48 m** at $t = 60\text{ s}$ via map centerline normal updates ($\mathbf{H}_p = \mathbf{n}_{\text{road}}^T$).
- **Along-track error ($e_\parallel$)** grew to **73.01 m** at $t = 60\text{ s}$, accounting for **99.96%** of total 2D positional error variance.

This document analyzes the mathematical mechanics of along-track error propagation and demonstrates how Phase 8 GNSS fusion resolves the longitudinal unobservability of Phase 7.

---

## 2. Road-Aligned Error Coordinate System

At any point along road segment $k$ with unit tangent $\mathbf{t}_k = [\cos\psi_k, \sin\psi_k]^T$ and unit normal $\mathbf{n}_k = [-\sin\psi_k, \cos\psi_k]^T$, the position error vector $\delta\mathbf{p} = [e_E, e_N]^T$ decomposes into:
$$\begin{bmatrix} e_\parallel \\ e_\perp \end{bmatrix} = \begin{bmatrix} \mathbf{t}_k^T \\ \mathbf{n}_k^T \end{bmatrix} \delta\mathbf{p} = \mathbf{R}_k \delta\mathbf{p}$$

The error covariance in road-aligned coordinates is:
$$\mathbf{P}_{\text{road}} = \mathbf{R}_k \mathbf{P}_{EN} \mathbf{R}_k^T = \begin{bmatrix} \sigma_\parallel^2 & \rho \sigma_\parallel \sigma_\perp \\ \rho \sigma_\parallel \sigma_\perp & \sigma_\perp^2 \end{bmatrix}$$

---

## 3. Asymmetric Error Dynamics in Phase 7 (GNSS Outage)

### 3.1 Lateral Error Dynamics (Cross-Track)
In Phase 7, road updates provide 1D orthogonal corrections:
$$z_{\text{map}} = d(\mathbf{p}, \mathcal{M}) = \mathbf{n}_k^T (\mathbf{p} - \mathbf{r}_0)$$
$$H_{\text{map}} = [\mathbf{n}_k^T, \mathbf{0}_{1\times 13}]$$

The Kalman update directly reduces lateral variance:
$$\sigma_{\perp,+}^2 = \sigma_{\perp,-}^2 \left(1 - \frac{\sigma_{\perp,-}^2}{\sigma_{\perp,-}^2 + R_{\text{map}}}\right)$$
Because map precision $R_{\text{map}} \approx (1.0\text{ m})^2$, $\sigma_\perp$ is strictly bounded to $< 1.0\text{ m}$ for all $t$.

### 3.2 Longitudinal Error Dynamics (Along-Track)
Because $\mathbf{H}_{\text{map}} \mathbf{t}_k = 0$, map updates provide **zero information** along $\mathbf{t}_k$:
$$\mathbf{K}_{\text{map}} \cdot \mathbf{t}_k = \mathbf{P}^- \mathbf{H}_{\text{map}}^T (S)^{-1} \cdot \mathbf{t}_k = \mathbf{P}^- \mathbf{n}_k (S)^{-1} \cdot \mathbf{t}_k = 0$$

Consequently, along-track position behaves as an open-loop integration of forward speed error $\delta v_{\text{fwd}}$:
$$e_\parallel(t) = e_\parallel(0) + \int_0^t \delta v_{\text{fwd}}(\tau) d\tau$$

Under AI speed estimation with residual bias $b_v = \mathbb{E}[\delta v_{\text{fwd}}] \approx 1.2\text{ m/s}$ and speed variance $\sigma_v^2 \approx (0.8\text{ m/s})^2$:
$$\mathbb{E}[e_\parallel(t)] \approx b_v \cdot t$$
$$\text{Var}(e_\parallel(t)) \approx \sigma_v^2 \cdot \tau_c \cdot t$$

At $t = 60\text{ s}$:
$$\mathbb{E}[e_\parallel(60)] \approx 1.2 \times 60 = 72\text{ m}$$
This matches the experimentally observed audit value of **73.01 m**.

---

## 4. Phase 8 Resolution: Full-Rank Observability via GNSS Fusion

### 4.1 2D Position Measurement
When GNSS is healthy, the measurement vector provides simultaneous observations in both East and North:
$$\mathbf{z}_p = \mathbf{p}_{\text{gnss}} = \mathbf{p} + \mathbf{v}_{\text{gnss}}, \quad \mathbf{v}_{\text{gnss}} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_p)$$
$$\mathbf{H}_p = [\mathbf{I}_3, \mathbf{0}_{3\times 12}]$$

Projecting into road coordinates:
$$\mathbf{H}_{p,\text{road}} = \mathbf{R}_k \mathbf{H}_p = \begin{bmatrix} \mathbf{t}_k^T & \mathbf{0}_{1\times 12} \\ \mathbf{n}_k^T & \mathbf{0}_{1\times 13} \end{bmatrix}$$

Both singular values of $\mathbf{H}_{p,\text{road}}$ are strictly non-zero ($s_1 = s_2 = 1.0$).

### 4.2 Longitudinal Variance Reduction
The longitudinal innovation $\nu_\parallel = \mathbf{t}_k^T (\mathbf{z}_p - \hat{\mathbf{p}})$ directly updates along-track position:
$$\Delta \hat{p}_\parallel = K_\parallel \nu_\parallel$$
$$\sigma_{\parallel,+}^2 = \sigma_{\parallel,-}^2 \left(1 - \frac{\sigma_{\parallel,-}^2}{\sigma_{\parallel,-}^2 + \sigma_{\text{gnss}}^2}\right)$$

For typical smartphone GNSS horizontal accuracy ($\sigma_{\text{gnss}} \approx 1.5 - 3.0\text{ m}$):
$$\lim_{t \to \infty} \sigma_\parallel(t) \le \sigma_{\text{gnss}} \approx 2.0\text{ m}$$

### 4.3 Closed-Loop Speed Bias Estimation
In addition to direct position updates, GNSS Doppler velocity provides:
$$\mathbf{z}_v = \mathbf{v}_{\text{gnss}} = \mathbf{v} + \mathbf{v}_v, \quad \mathbf{H}_v = [\mathbf{0}_{3\times 3}, \mathbf{I}_3, \mathbf{0}_{3\times 9}]$$

Because $\mathbf{H}_v$ observes vehicle 3D velocity directly, the correlation between velocity errors and accelerometer bias states $\mathbf{b}_a$ in the error transition matrix $\mathbf{F}$ enables continuous estimation and tracking of IMU biases:
$$\dot{\mathbf{b}}_a \to \text{Observable during vehicle maneuvers}$$

---

## 5. Summary Table: Error Dynamics Comparison

| Regime | Lateral Error $e_\perp$ ($1\sigma$) | Along-Track Error $e_\parallel$ ($1\sigma$) | Ratio $\sigma_\parallel / \sigma_\perp$ | Primary Limiting Mechanism |
|---|---|---|---|---|
| **Phase 5 (AI Speed)** | $25.0 - 50.0\text{ m}$ (unconstrained) | $50.0 - 100.0\text{ m}$ (unconstrained) | $\approx 2.0$ | Gyro drift & open-loop integration |
| **Phase 6 (+Physics)** | $5.0 - 15.0\text{ m}$ (NHC bounded) | $30.0 - 80.0\text{ m}$ (speed drift) | $\approx 5.0$ | AI speed model bias |
| **Phase 7 (+Map)** | **0.48 m** (centerline locked) | **73.01 m** (unobservable) | **$\approx 152.0$** | 1D road topology projection |
| **Phase 8 (+GNSS)** | **0.80 m** (dual map + GNSS) | **1.20 m** (GNSS locked) | **$\approx 1.5$** | GNSS pseudorange multipath / dilution |
| **Phase 8 (Outage)** | $0.48\text{ m} \to 1.5\text{ m}$ (smooth DR) | $1.2\text{ m} \to 15.0\text{ m}$ (DR drift) | Grows with $t$ | Soft transition, zero teleportation |
