# Phase 8 Mathematical Foundation: Smooth GNSS Recovery Strategy & Continuity Bounds

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/06_RECOVERY_STRATEGY.md`  

---

## 1. Executive Summary

This document formalizes the seamless recovery strategy employed when GNSS signals return after an extended outage. The system strictly avoids "hard position snapping" or "teleportation", instead utilizing covariance-aware soft Kalman gain injection with bounded correction rates.

---

## 2. The Danger of Hard Trajectory Snapping

In naive navigation systems, GNSS re-acquisition triggers a hard state overwrite:
$$\mathbf{p}_{\text{est}}(t_r^+) \leftarrow \mathbf{z}_{\text{gnss}}(t_r)$$
If dead reckoning has accumulated $50\text{ m}$ of longitudinal drift during a 60s blackout, a hard overwrite causes:
1. **Severe Trajectory Teleportation**: Instantaneous spatial jump $\|\Delta\mathbf{p}\| = 50\text{ m}$.
2. **Infinite Acceleration Spike**: $a = \frac{\Delta v}{\Delta t} \to \infty$, corrupting vehicle dynamics.
3. **Attitude Distortion**: If velocity direction is snapped directly to GNSS course without roll/pitch decoupling.
4. **Covariance Inconsistency**: Zeroing or shrinking $\mathbf{P}$ without proper measurement update ruins filter stability.

---

## 3. Covariance-Aware Soft Recovery Formulation

In Phase 8, recovery is executed via recursive Kalman innovation updates, where the correction is strictly governed by the Kalman gain $\mathbf{K}$:
$$\delta\mathbf{x} = \mathbf{K} \boldsymbol{\nu}$$
$$\mathbf{p}^+ = \hat{\mathbf{p}}^- + \delta\mathbf{p} = \hat{\mathbf{p}}^- + \mathbf{K}_p \boldsymbol{\nu}_p$$
where:
$$\mathbf{K}_p = \mathbf{P}_{pp}^- (\mathbf{P}_{pp}^- + \mathbf{R}_{p,\text{rec}})^{-1}$$

### Soft Recovery Covariance Inflation:
During the first $M = 5$ frames of recovery ($t \in [t_r, t_r + 0.5\text{ s}]$), the measurement covariance is smoothly faded:
$$\mathbf{R}_{p,\text{rec}}(k) = \mathbf{R}_p \cdot \left(1.0 + \kappa_{\text{fade}} \cdot \exp\left(-\frac{k - k_r}{\tau_{\text{rec}}}\right)\right)$$
where:
- $\kappa_{\text{fade}} = 4.0$: Initial covariance inflation factor.
- $\tau_{\text{rec}} = 10\text{ frames} = 1.0\text{ s}$: Exponential convergence time constant.

### Maximum Single-Step Correction Clamping:
To guarantee physical plausibility, the maximum single-step position correction is bounded by maximum vehicle kinematic limits:
$$\|\delta\mathbf{p}_k\| \le v_{\text{max}} \cdot \Delta t_k + \frac{1}{2} a_{\text{max}} \Delta t_k^2 \approx 3.5\text{ m per 100ms step}$$
If $\|\mathbf{K}_p \boldsymbol{\nu}_p\| > 3.5\text{ m}$, the correction vector is scaled:
$$\delta\mathbf{p}_k \leftarrow 3.5 \cdot \frac{\mathbf{K}_p \boldsymbol{\nu}_p}{\|\mathbf{K}_p \boldsymbol{\nu}_p\|}$$
The remaining innovation is absorbed smoothly across subsequent epochs, ensuring zero visual teleportation.

---

## 4. Quantitative Continuity Metrics

At recovery time $t_r$ and during the settling interval $[t_r, t_r + 3\text{ s}]$, four metrics are formally evaluated:

1. **Instantaneous Position Discontinuity**:
   $$\Delta p_{\text{jump}} = \|\mathbf{p}(t_r^+) - \mathbf{p}(t_r^-)\|$$
   - Target Acceptance Bound: $\Delta p_{\text{jump}} \le 3.5\text{ m}$ (zero teleportation).

2. **Instantaneous Velocity Discontinuity**:
   $$\Delta v_{\text{jump}} = \|\mathbf{v}(t_r^+) - \mathbf{v}(t_r^-)\|$$
   - Target Acceptance Bound: $\Delta v_{\text{jump}} \le 1.0\text{ m/s}$.

3. **Heading Jump**:
   $$\Delta\psi_{\text{jump}} = |\text{wrap}(\psi(t_r^+) - \psi(t_r^-))|$$
   - Target Acceptance Bound: $\Delta\psi_{\text{jump}} \le 2.0^\circ$.

4. **Correction Settling Time ($T_{\text{settle}}$)**:
   The elapsed time from recovery onset $t_r$ until the position error residual drops and stabilizes within $2\sigma$ of the steady-state GNSS covariance envelope:
   $$T_{\text{settle}} = \min \{ \tau \ge 0 \mid \|\mathbf{p}(t_r + \tau) - \mathbf{p}_{\text{ref}}(t_r + \tau)\| \le 2\sqrt{\text{tr}(\mathbf{P}_{pp})} \}$$
   - Nominal Settling Time: $T_{\text{settle}} \in [1.2\text{ s}, 2.5\text{ s}]$.
