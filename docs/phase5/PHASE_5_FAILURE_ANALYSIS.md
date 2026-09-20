# Phase 5 Failure & Drift Mechanism Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 5 — 15-State Error-State Kalman Filter Navigation Core  
**Document Code**: `SIH26168-FAIL-PHASE5-01`  
**Classification**: DIAGNOSTIC & ROOT CAUSE ANALYSIS  

---

## 1. Executive Summary of Failure Modes

While Phase 5 successfully demonstrates a **2.14x drift reduction** on 60s blackouts and exact NIS innovation consistency ($\text{NIS} = 0.943$), long-duration outages ($>60\text{ s}$) and multi-turn maneuvers reveal specific failure modes. In accordance with the project's **No-False-Claims Policy**, these failure modes are rigorously investigated below.

---

## 2. Failure Mode 1: Unconstrained Horizontal Yaw Drift

### Mechanism:
As derived in [`PHASE_5_OBSERVABILITY.md`](file:///d:/python/SIH%2026%20ISRO/docs/phase5/PHASE_5_OBSERVABILITY.md), the forward speed measurement $z_v = \mathbf{e}_1^T \mathbf{R}_{nb}^T \mathbf{v}_n$ only constrains the **magnitude** of the velocity vector; it cannot directly observe the absolute horizontal rotation angle around the vertical $+z_n$ axis.

During the 120s blackout ($661.4\text{ m}$ traveled), uncompensated gyroscope bias random walk caused the estimated heading to drift by **$61.95^\circ$** (E1) and **$76.22^\circ$** (E2).
When heading drifts by $\Delta \psi$, projecting forward speed $\hat{v}$ produces a cross-track velocity error:
$$v_{\text{cross-track}} = \hat{v} \sin(\Delta \psi)$$
At $\hat{v} = 15\text{ m/s}$ and $\Delta \psi = 30^\circ$, cross-track velocity error is $7.5\text{ m/s}$, producing an endpoint error of $450\text{ meters}$ in 60 seconds.

### Root Cause:
Consumer smartphone gyroscopes suffer from thermal bias instability ($\sim 0.05 - 0.2^\circ/\text{s}$). Without an absolute heading reference (magnetometer or dual-antenna GNSS) or vehicle kinematic constraints, gyro integration will inevitably wander.

---

## 3. Failure Mode 2: Acceleration-Tilt Cross-Coupling

### Mechanism:
In the error dynamics matrix:
$$\mathbf{F}[3:6, 6:9] = - \mathbf{R}_{nb} [\hat{\mathbf{f}}_b]_\times \Delta t$$
When the vehicle executes a prolonged curve, the lateral specific force is non-zero:
$$f_{\text{lat}} \approx v \cdot \omega_{\text{yaw}} + g \sin\phi$$
In the absence of lateral velocity zero-constraints ($v_{\text{lat}} \approx 0$), the filter cannot distinguish whether horizontal acceleration is caused by true vehicle turning or by phone roll tilt $\phi$. This ambiguity induces slight pitch/roll estimation errors that bleed gravity into the forward axis.

---

## 4. Failure Mode 3: Creep Drift during Standstill

### Mechanism:
In Phase 5, standstill zero-velocity updates (ZUPT) are **intentionally not implemented** (scoped strictly for Phase 6).
When the vehicle halts at a red light:
- The AI speed correctly outputs $\hat{v} \approx 0.1\text{ m/s}$.
- However, residual accelerometer noise ($0.08\text{ m/s}^2$) continues to be integrated by the nominal INS mechanization, causing estimated position to slowly "creep" forward at $0.1 - 0.3\text{ m/s}$ during extended stops.

---

## 5. Architectural Prescriptions for Phase 6

To solve these three root-cause mechanisms, Phase 6 must introduce:
1. **Non-Holonomic Constraints (NHC)**:
   Land vehicles cannot drive sideways or fly:
   $$v_{\text{lateral}} \approx 0, \quad v_{\text{vertical}} \approx 0$$
   Applying these two virtual measurement updates pins the velocity vector strictly to the vehicle's longitudinal centerline, eliminating sideways sliding drift.
2. **Zero-Velocity Updates (ZUPT)**:
   When the Phase 3 standstill detector flags vehicle standstill, the filter must apply $\mathbf{v}_n = \mathbf{0}$, resetting velocity variance and halting position creep.
3. **Turn-Rate Constrained Yaw Updates**:
   Using NHC cross-coupling to bound heading drift during maneuvers.
