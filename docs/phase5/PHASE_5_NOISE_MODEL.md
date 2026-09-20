# Phase 5 Stochastic Noise Model & Process Covariance

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-NOISE-PHASE5-01`  
**Classification**: STOCHASTIC MODELING SPECIFICATION  

---

## 1. Continuous Stochastic Noise Process Formulation

### Technical Derivation
The 15-state error dynamics driven by continuous white noise vectors is:
$$\delta \dot{\mathbf{x}}(t) = \mathbf{F}_c(t) \delta \mathbf{x}(t) + \mathbf{G}_c(t) \mathbf{w}(t)$$

where the driving noise vector $\mathbf{w}(t) \in \mathbb{R}^{12}$ is composed of four 3-dimensional uncorrelated Gaussian white noise processes:
$$\mathbf{w}(t) = \begin{bmatrix} \mathbf{n}_a(t) \\ \mathbf{n}_g(t) \\ \mathbf{n}_{ba}(t) \\ \mathbf{n}_{bg}(t) \end{bmatrix}, \quad \mathbb{E}[\mathbf{w}(t)] = \mathbf{0}, \quad \mathbb{E}[\mathbf{w}(t) \mathbf{w}^T(\tau)] = \mathbf{Q}_c \delta(t - \tau)$$

The continuous power spectral density matrix $\mathbf{Q}_c \in \mathbb{R}^{12 \times 12}$ is block diagonal:
$$\mathbf{Q}_c = \begin{bmatrix}
\sigma_a^2 \mathbf{I}_{3 \times 3} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \sigma_g^2 \mathbf{I}_{3 \times 3} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \sigma_{ba}^2 \mathbf{I}_{3 \times 3} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \sigma_{bg}^2 \mathbf{I}_{3 \times 3}
\end{bmatrix}$$

---

## 2. Empirical Parameter Estimation from Sequence Vw1

Rather than inventing arbitrary noise figures, noise densities were calculated from the stationary calibration sequence Vw1 ($3,524\text{ samples}$ = $352.4\text{ s}$ at $10\text{ Hz}$):

| Parameter | Physical Interpretation | Continuous PSD Density | Discrete Equivalent at 10 Hz ($\Delta t = 0.1\text{ s}$) | Empirical Source |
|:----------|:------------------------|:----------------------:|:--------------------------------------------------------:|:-----------------|
| $\sigma_a$ | Accelerometer Velocity Random Walk (VRW) | $0.080\text{ m/s}^2/\sqrt{\text{Hz}}$ | $\sigma_a \sqrt{\Delta t} = 0.0253\text{ m/s}^2$ | Allan Variance / Vw1 static standard deviation |
| $\sigma_g$ | Gyroscope Angular Random Walk (ARW) | $0.005\text{ rad/s}/\sqrt{\text{Hz}}$ | $\sigma_g \sqrt{\Delta t} = 0.00158\text{ rad/s}$ | Allan Variance / Vw1 static standard deviation |
| $\sigma_{ba}$| Accelerometer Bias Random Walk | $1.0 \times 10^{-4}\text{ m/s}^3/\sqrt{\text{Hz}}$ | $3.16 \times 10^{-5}\text{ m/s}^2$ | Empirical thermal drift bounds |
| $\sigma_{bg}$| Gyroscope Bias Random Walk | $1.0 \times 10^{-5}\text{ rad/s}^2/\sqrt{\text{Hz}}$ | $3.16 \times 10^{-6}\text{ rad/s}$ | Empirical thermal drift bounds |

---

## 3. Discretization & Trapezoidal Van Loan Integration

### Technical Derivation
The discrete process noise covariance $\mathbf{Q}_k \in \mathbb{R}^{15 \times 15}$ over integration interval $\Delta t$ is:
$$\mathbf{Q}_k = \int_0^{\Delta t} \exp(\mathbf{F}_c \tau) \mathbf{G}_c \mathbf{Q}_c \mathbf{G}_c^T \exp(\mathbf{F}_c^T \tau) d\tau$$

Retaining terms up to $\mathcal{O}(\Delta t^3)$, the block diagonal elements are:
$$\mathbf{Q}_{\delta p} = \frac{1}{3} \sigma_a^2 \Delta t^3 \mathbf{I}_{3 \times 3}$$
$$\mathbf{Q}_{\delta v} = \sigma_a^2 \Delta t \mathbf{I}_{3 \times 3}$$
$$\mathbf{Q}_{\delta p, \delta v} = \mathbf{Q}_{\delta v, \delta p}^T = \frac{1}{2} \sigma_a^2 \Delta t^2 \mathbf{I}_{3 \times 3}$$
$$\mathbf{Q}_{\delta \theta} = \sigma_g^2 \Delta t \mathbf{I}_{3 \times 3}$$
$$\mathbf{Q}_{\delta ba} = \sigma_{ba}^2 \Delta t \mathbf{I}_{3 \times 3}$$
$$\mathbf{Q}_{\delta bg} = \sigma_{bg}^2 \Delta t \mathbf{I}_{3 \times 3}$$

### Simple Explanation
Think of process noise like adding a pinch of salt to a recipe at every step.
Because every sensor reading has tiny tremors, your position, speed, and heading become slightly less certain every millisecond.
$\mathbf{Q}_k$ tells the Kalman filter: *"Every 0.1 seconds, expand your position doubt by a few millimeters and your angle doubt by a fraction of a degree."*

### Why This Matters for Our Project
If process noise $\mathbf{Q}$ is set too low, the filter becomes "smug" (over-confident) and ignores new AI speed measurements. If $\mathbf{Q}$ is set too high, the filter becomes "jittery" and chases every vibration spike. The empirically derived noise figures ensure the filter remains balanced, responsive, and mathematically consistent.
