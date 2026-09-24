# Phase 8 Mathematical Foundation: GNSS Measurement Models & Covariance Formulations

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/02_GNSS_MEASUREMENT_MODEL.md`  

---

## 1. Executive Summary

This document derives the probabilistic measurement equations for GNSS position, velocity, and antenna lever-arm compensation within the 15-state Error-State Kalman Filter (ESKF).

---

## 2. GNSS Position Observation Model

The true GNSS antenna position $\mathbf{p}_{\text{ant}}^n$ in the local navigation (ENU) frame is related to the vehicle IMU reference center $\mathbf{p}^n$ by:
$$\mathbf{p}_{\text{ant}}^n = \mathbf{p}^n + \mathbf{R}_{nb} \mathbf{l}_b$$
where:
- $\mathbf{p}^n = [p_E, p_N, p_U]^T \in \mathbb{R}^3$: IMU center position.
- $\mathbf{R}_{nb} \in \text{SO}(3)$: Direction cosine matrix rotating from body frame to navigation frame.
- $\mathbf{l}_b = [l_x, l_y, l_z]^T \in \mathbb{R}^3$: Antenna lever-arm offset expressed in body frame.

The GNSS position measurement $\mathbf{z}_p \in \mathbb{R}^3$ is corrupted by additive zero-mean Gaussian noise $\mathbf{n}_p \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_p)$:
$$\mathbf{z}_p = \mathbf{p}_{\text{ant}}^n + \mathbf{n}_p = \mathbf{p}^n + \mathbf{R}_{nb} \mathbf{l}_b + \mathbf{n}_p$$

### Predicted Measurement:
$$\hat{\mathbf{z}}_p = \hat{\mathbf{p}}^n + \hat{\mathbf{R}}_{nb} \mathbf{l}_b$$

### Position Innovation Vector:
$$\boldsymbol{\nu}_p = \mathbf{z}_p - \hat{\mathbf{z}}_p \in \mathbb{R}^3$$

When $\mathbf{l}_b = \mathbf{0}$ (default zero lever arm):
$$\hat{\mathbf{z}}_p = \hat{\mathbf{p}}^n \implies \boldsymbol{\nu}_p = \mathbf{z}_p - \hat{\mathbf{p}}^n$$

---

## 3. GNSS Velocity Observation Model

The velocity of the antenna in the navigation frame includes the tangential velocity induced by vehicle angular rotation:
$$\mathbf{v}_{\text{ant}}^n = \mathbf{v}^n + \mathbf{R}_{nb} (\boldsymbol{\omega}_b \times \mathbf{l}_b)$$
where:
- $\mathbf{v}^n = [v_E, v_N, v_U]^T \in \mathbb{R}^3$: IMU center linear velocity in ENU.
- $\boldsymbol{\omega}_b \in \mathbb{R}^3$: Unbiased vehicle angular velocity in body frame ($\boldsymbol{\omega}_b = \boldsymbol{\omega}_{\text{raw}} - \mathbf{b}_g$).

The GNSS velocity measurement $\mathbf{z}_v \in \mathbb{R}^3$ with noise $\mathbf{n}_v \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_v)$ is:
$$\mathbf{z}_v = \mathbf{v}^n + \mathbf{R}_{nb} (\boldsymbol{\omega}_b \times \mathbf{l}_b) + \mathbf{n}_v$$

### Predicted Measurement:
$$\hat{\mathbf{z}}_v = \hat{\mathbf{v}}^n + \hat{\mathbf{R}}_{nb} (\boldsymbol{\omega}_b \times \mathbf{l}_b)$$

### Velocity Innovation Vector:
$$\boldsymbol{\nu}_v = \mathbf{z}_v - \hat{\mathbf{z}}_v \in \mathbb{R}^3$$

When $\mathbf{l}_b = \mathbf{0}$:
$$\hat{\mathbf{z}}_v = \hat{\mathbf{v}}^n \implies \boldsymbol{\nu}_v = \mathbf{z}_v - \hat{\mathbf{v}}^n$$

---

## 4. Adaptive Measurement Covariance Formulations

Measurement noise covariances $\mathbf{R}_p$ and $\mathbf{R}_v$ are dynamically parameterized from receiver accuracy estimates:

### 1. Position Covariance $\mathbf{R}_p$:
When reported horizontal accuracy $\sigma_h$ (from `gps_acc_m`) and vertical accuracy $\sigma_v$ are available:
$$\mathbf{R}_p = \begin{bmatrix}
\sigma_h^2 & 0 & 0 \\
0 & \sigma_h^2 & 0 \\
0 & 0 & \sigma_v^2
\end{bmatrix}$$
where vertical uncertainty defaults to $\sigma_v = 1.5 \cdot \sigma_h$ in standard satellite geometry.

If receiver metadata is absent, defaults are configured:
$$\sigma_{p,\text{base}} = 2.5\text{ m} \implies \mathbf{R}_p = \text{diag}([2.5^2, 2.5^2, 4.0^2])$$

### 2. Velocity Covariance $\mathbf{R}_v$:
GNSS Doppler velocity uncertainty $\sigma_{v,\text{base}}$:
$$\mathbf{R}_v = \begin{bmatrix}
\sigma_{v,\text{horiz}}^2 & 0 & 0 \\
0 & \sigma_{v,\text{horiz}}^2 & 0 \\
0 & 0 & \sigma_{v,\text{vert}}^2
\end{bmatrix}$$
where baseline horizontal velocity uncertainty is $\sigma_{v,\text{horiz}} = 0.15\text{ m/s}$ and vertical velocity uncertainty is $\sigma_{v,\text{vert}} = 0.30\text{ m/s}$.

### 3. Covariance Inflation During Degraded State:
When GNSS quality is classified as `GNSS_SUSPECT` or recovering from outage:
$$\mathbf{R}_p \leftarrow \kappa_p \cdot \mathbf{R}_p, \quad \mathbf{R}_v \leftarrow \kappa_v \cdot \mathbf{R}_v$$
where inflation factors $\kappa_p, \kappa_v \in [2.0, 10.0]$ prevent state shocks while preserving Kalman filter convergence.
