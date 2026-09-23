# Pre-Phase-8 Audit: Longitudinal Velocity Error Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/05_LONGITUDINAL_ERROR_ANALYSIS.md`  

---

## 1. Executive Summary

This document addresses one of the primary paradoxes revealed by the debug telemetry:
> **Why is the final Phase 6 forward velocity error substantially larger than the raw Phase 4 AI forward speed error over extended blackouts?**

On the 120-second blackout benchmark:
- Raw Phase 4 AI speed integrated distance error: **$-60.50\text{ m}$**.
- Phase 6 ESKF forward velocity integrated distance error: **$-280.04\text{ m}$** ($4.6\times$ larger!).

Through systematic state tracing, we isolate the two distinct physical and mathematical mechanisms responsible for this divergence:
1. **Initial Cold-Start Buffer Transient**: Injects an artificial $-24.25\text{ m}$ longitudinal deficit during the first 3 seconds when the rolling FIFO buffer is initialized to zero.
2. **Attitude Heading Error Cross-Projection**: As unobserved gyroscope yaw drift accumulates over long outages ($\delta\psi \to 63.06^\circ$ at $t = 120\text{ s}$ in Phase 6), the velocity vector in the navigation frame is projected into the estimated heading with a degradation factor of $\cos(\delta\psi)$. At $\delta\psi = 60^\circ$, $\cos(60^\circ) = 0.50$, causing the estimated forward speed in the vehicle body frame to collapse by $50\%$.

---

## 2. Quantitative Speed Error Breakdown Across Outage Durations

Evaluated on the $4,600.0 - 4,720.0\text{ s}$ benchmark slice against survey-grade CAN reference speed:

| Duration | Signal Source | Instantaneous MAE | Instantaneous RMSE | Bias | Integrated Distance Deficit $E_d(T)$ | Primary Error Driver |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **10 s** | Phase 4 AI Speed (Cold) | 2.465 m/s | 3.131 m/s | -2.425 m/s | -24.25 m | Cold-Start Zero Buffer |
| | **Phase 4 AI Speed (Warm)** | **0.428 m/s** | **0.612 m/s** | **-0.120 m/s** | **-1.20 m** | **Normal Sensor Noise** |
| | Phase 6 Fwd Velocity (Cold) | 3.675 m/s | 3.961 m/s | -3.675 m/s | -36.75 m | Cold Start + Prior Filter Inertia |
| **30 s** | Phase 4 AI Speed (Cold) | 1.346 m/s | 1.997 m/s | -0.291 m/s | -8.72 m | Buffer Washout (27s warm) |
| | Phase 6 Fwd Velocity (Cold) | 1.698 m/s | 2.476 m/s | -1.212 m/s | -36.36 m | Initial Velocity Drag Persisting |
| **60 s** | Phase 4 AI Speed (Cold) | 0.673 m/s | 1.412 m/s | -0.145 m/s | -8.72 m | Red Light Standstill Clamping |
| | Phase 6 Fwd Velocity (Cold) | 0.852 m/s | 1.751 m/s | -0.608 m/s | -36.50 m | Standstill Clamping (ZUPT Active) |
| **120 s** | Phase 4 AI Speed (Cold) | 1.010 m/s | 1.676 m/s | -0.505 m/s | -60.50 m | AI Model Tracking Active Cruising |
| | **Phase 6 Fwd Velocity** | **2.902 m/s** | **4.822 m/s** | **-2.336 m/s** | **-280.04 m** | **Yaw Drift Projection Collapse ($\cos\delta\psi$)** |

---

## 3. Operating Regime Error Analysis

Breaking down Phase 4 speed inference by vehicle dynamic state:

| Operating Regime | Sample Count | Duration | CAN Speed Range | AI Speed MAE | AI Speed Bias | Findings |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **STANDSTILL** | 436 samples | 43.6 s | $0.00\text{ m/s}$ | **0.002 m/s** | **0.000 m/s** | Standstill detection perfectly locks speed to zero. |
| **CRUISING** | 763 samples | 76.3 s | $10.5 - 12.0\text{ m/s}$ | 1.585 m/s | -0.793 m/s | Includes the initial 3-second cold start transient. |

---

## 4. Mathematical Derivation of Yaw-Induced Longitudinal Velocity Collapse

Let the true vehicle velocity in the local ENU navigation frame be $\mathbf{v}_n^{\text{true}} = v \begin{bmatrix} \cos\psi \\ \sin\psi \\ 0 \end{bmatrix}$.

Let the filter estimated attitude quaternion have yaw error $\delta\psi = \hat{\psi} - \psi$. The estimated forward direction unit vector is:
$$\hat{\mathbf{u}}_{\text{fwd}} = \begin{bmatrix} \cos\hat{\psi} \\ \sin\hat{\psi} \\ 0 \end{bmatrix}$$

The estimated body forward velocity extracted from the navigation frame state is:
$$\hat{v}_{\text{fwd}} = \mathbf{v}_n^T \hat{\mathbf{u}}_{\text{fwd}} = v (\cos\psi \cos\hat{\psi} + \sin\psi \sin\hat{\psi}) = v \cos(\hat{\psi} - \psi) = v \cos(\delta\psi)$$

### Consequence:
1. When $\delta\psi = 0^\circ \implies \cos(0^\circ) = 1.00 \implies \hat{v}_{\text{fwd}} = v$.
2. When $\delta\psi = 20^\circ \implies \cos(20^\circ) = 0.940 \implies 6\%$ speed reduction.
3. When $\delta\psi = 45^\circ \implies \cos(45^\circ) = 0.707 \implies 29.3\%$ speed reduction.
4. When $\delta\psi = 63.06^\circ$ (Phase 6 at 120s) $\implies \cos(63.06^\circ) = \mathbf{0.453} \implies \mathbf{54.7\%}$ speed collapse!

This proves analytically that in Phase 6, **large forward velocity errors over 120s are not caused by Phase 4 AI failures, but by unobserved attitude yaw drift rotating the navigation velocity vector away from the body longitudinal axis**.

This directly validates the Phase 7 architecture: road heading updates $H_\psi$ bound yaw drift ($\delta\psi \le 2.79^\circ$), preserving $\cos(\delta\psi) \ge 0.998$ and preventing longitudinal velocity collapse.
