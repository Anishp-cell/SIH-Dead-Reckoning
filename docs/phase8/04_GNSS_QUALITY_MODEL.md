# Phase 8 Mathematical Foundation: GNSS Quality Assessment & Statistical Anomaly Detection

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/04_GNSS_QUALITY_MODEL.md`  

---

## 1. Executive Summary

This document establishes the multi-layer GNSS quality assessment engine. Rather than relying on naive single-threshold binary flags, the engine combines geometric signal metrics, kinematic consistency checks, and 3-DOF Chi-square Normalized Innovation Squared (NIS) hypothesis testing.

---

## 2. Multi-Signal Quality Indicators

The structured `GNSSQualityReport` evaluates seven physical and statistical indicators:

1. **Fix Validity & Timestamp Continuity**:
   - Time elapsed since last fix: $\Delta t_{\text{gnss}} = t_k - t_{k-1}$.
   - Flagged stale if $\Delta t_{\text{gnss}} > 1.5\text{ s}$.
   - Non-positive or backward timestamps ($\Delta t \le 0$) immediately rejected.

2. **Satellite Constellation Geometry**:
   - Number of tracked satellites: $N_{\text{sats}} \ge 4$ mandatory for 3D fix; $N_{\text{sats}} \ge 6$ nominal.
   - If $N_{\text{sats}} < 4$: Fix declared invalid.

3. **Receiver Horizontal Accuracy Floor**:
   - Reported horizontal accuracy radius: $\sigma_{\text{acc}} \le 10.0\text{ m}$ nominal.
   - If $\sigma_{\text{acc}} > 15.0\text{ m}$: Flagged degraded.

4. **Kinematic Step Rate-of-Change Checks**:
   - Instantaneous pseudo-speed between successive GNSS fixes:
     $$v_{\text{step}} = \frac{\|\mathbf{p}_{\text{gnss}, k} - \mathbf{p}_{\text{gnss}, k-1}\|}{\Delta t_{\text{gnss}}}$$
   - If $v_{\text{step}} > 45.0\text{ m/s}$ ($162\text{ km/h}$ for ground vehicle): Flagged as an isolated multipath position jump.
   - Velocity step acceleration:
     $$a_{\text{step}} = \frac{\|\mathbf{v}_{\text{gnss}, k} - \mathbf{v}_{\text{gnss}, k-1}\|}{\Delta t_{\text{gnss}}}$$
   - If $a_{\text{step}} > 12.0\text{ m/s}^2$ ($> 1.22\text{ g}$): Flagged as a Doppler velocity spike.

---

## 3. Statistical Normalized Innovation Squared (NIS) Hypothesis Testing

Every GNSS measurement is validated against the filter's prior state covariance before error injection.

### Position NIS (3-DOF):
$$\boldsymbol{\nu}_p = \mathbf{z}_p - \hat{\mathbf{p}}^n \in \mathbb{R}^3$$
$$\mathbf{S}_p = \mathbf{H}_p \mathbf{P}^- \mathbf{H}_p^T + \mathbf{R}_p \in \mathbb{R}^{3 \times 3}$$
$$\text{NIS}_p = \boldsymbol{\nu}_p^T \mathbf{S}_p^{-1} \boldsymbol{\nu}_p$$

### Velocity NIS (3-DOF):
$$\boldsymbol{\nu}_v = \mathbf{z}_v - \hat{\mathbf{v}}^n \in \mathbb{R}^3$$
$$\mathbf{S}_v = \mathbf{H}_v \mathbf{P}^- \mathbf{H}_v^T + \mathbf{R}_v \in \mathbb{R}^{3 \times 3}$$
$$\text{NIS}_v = \boldsymbol{\nu}_v^T \mathbf{S}_v^{-1} \boldsymbol{\nu}_v$$

### Critical Design Principle (Preserving Pre-Phase-8 Freeze):
> **Raw unclipped innovation must be evaluated in NIS calculation before any optional soft clamping is applied.**

---

## 4. Chi-Square Gating Thresholds & Classification Actions

Under the null hypothesis $H_0$ that measurements are consistent with prior state uncertainty and zero-mean Gaussian noise, the NIS follows a central Chi-square distribution with $m = 3$ degrees of freedom:
$$\text{NIS} \sim \chi_3^2$$

At significance level $\alpha = 0.01$ (99% confidence region):
$$\gamma_{99\%} = \chi_3^2(0.99) = 11.345$$
At significance level $\alpha = 0.001$ (99.9% confidence region):
$$\gamma_{99.9\%} = \chi_3^2(0.999) = 16.266$$

### Decision Mapping:
1. **`ACCEPT`**: $\text{NIS} \le 11.345$ and all physical checks pass. Standard Kalman update applied.
2. **`DOWNWEIGHT`**: $11.345 < \text{NIS} \le 16.266$ or single isolated anomaly. Measurement covariance scaled by $\kappa = \frac{\text{NIS}}{11.345} \in [1.5, 4.0]$.
3. **`REJECT`**: $\text{NIS} > 16.266$ or kinematic jump detected. Measurement discarded; logged to outage counter.
