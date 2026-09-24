# Phase 8 Mathematical Foundation: Causal Outage State Machine & Hysteresis Logic

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/05_OUTAGE_STATE_MACHINE.md`  

---

## 1. Executive Summary

This document formalizes the causal 4-state automaton governing GNSS fusion modes: `GNSS_HEALTHY`, `GNSS_SUSPECT`, `GNSS_OUTAGE`, and `GNSS_RECOVERING`. Strict temporal hysteresis prevents mode thrashing under noisy edge conditions, and state transitions are strictly causal ($t \le t_k$).

---

## 2. Formal Automaton State Definitions

```text
       ┌───────────────────────────────┐
       │         GNSS_HEALTHY          │◀─────────────────────────┐
       │ (Active Fusion, Nominal Cov)  │                          │
       └──────────────┬────────────────┘                          │
                      │                                           │
         NIS Reject or│                                           │
         Degraded Fix │ (1-2 consecutive)                         │
                      ▼                                           │
       ┌───────────────────────────────┐                          │
       │         GNSS_SUSPECT          │                          │
       │ (Downweighted, Cov Inflation) │                          │
       └──────────────┬────────────────┘                          │
                      │                                           │
         Consecutive  │ (N >= 3 or                                │ M >= 3 valid
         Rejections   │  dt > 1.0s)                               │ consecutive
                      ▼                                           │ fixes with
       ┌───────────────────────────────┐                          │ bounded NIS
       │          GNSS_OUTAGE          │                          │
       │ (Suppression, Pure Phase 7 DR)│                          │
       └──────────────┬────────────────┘                          │
                      │                                           │
         Valid Fix    │                                           │
         Re-Acquired  │ (1st valid fix)                           │
                      ▼                                           │
       ┌───────────────────────────────┐                          │
       │        GNSS_RECOVERING        │                          │
       │ (Soft Convergence, No Snaps)  ├──────────────────────────┘
       └───────────────────────────────┘
```

### 1. `GNSS_HEALTHY`:
- **Preconditions**: Continuous valid fixes, $N_{\text{sats}} \ge 4$, $\sigma_h \le 10\text{ m}$, $\text{NIS}_p \le 11.345$, $\text{NIS}_v \le 11.345$.
- **Action**: Both position and velocity Kalman updates executed at nominal covariance $\mathbf{R}_p, \mathbf{R}_v$.

### 2. `GNSS_SUSPECT`:
- **Preconditions**: Isolated NIS rejection ($11.345 < \text{NIS} \le 16.266$) or single missing frame ($0.2\text{ s} \le \Delta t \le 0.5\text{ s}$).
- **Action**: Downweighted Kalman update with inflated covariance ($\mathbf{R} \leftarrow \kappa \mathbf{R}$) or single-frame freeze. Transition back to `HEALTHY` if next frame is valid.

### 3. `GNSS_OUTAGE`:
- **Preconditions**: $N_{\text{fail}} \ge 3$ consecutive rejected frames, or timestamp gap $\Delta t > 1.0\text{ s}$, or receiver reporting zero satellites / no fix.
- **Action**: Complete suppression of GNSS measurement updates.
- **Crucial Invariant**: **Zero state resets, zero position jumps, zero covariance resets**. Estimator continues seamlessly using Phase 7 dead reckoning.

### 4. `GNSS_RECOVERING`:
- **Preconditions**: While in `OUTAGE`, a valid GNSS fix is observed.
- **Action**: Enters recovery monitoring. Validates persistence across $M_{\text{req}} = 3$ consecutive valid samples with bounded innovation. Applies soft covariance-inflated Kalman updates to gradually restore GNSS authority without trajectory teleportation. Once $M \ge 3$ verified, transitions back to `GNSS_HEALTHY`.

---

## 3. Mathematical Hysteresis Counter Logic

Let $k$ index the discrete time step $t_k$:

### Outage Entrance Condition (Causal):
$$N_{\text{fail}, k} = \begin{cases}
N_{\text{fail}, k-1} + 1 & \text{if } \text{invalid or } \text{NIS} > \gamma_{\text{reject}} \\
0 & \text{if } \text{valid and } \text{NIS} \le \gamma_{\text{accept}}
\end{cases}$$

$$\text{State}_k = \text{GNSS\_OUTAGE} \iff (N_{\text{fail}, k} \ge 3) \lor (\Delta t_{\text{gnss}} > 1.0\text{ s})$$

### Recovery Exit Condition (Causal):
$$M_{\text{valid}, k} = \begin{cases}
M_{\text{valid}, k-1} + 1 & \text{if } \text{valid and } \text{NIS} \le \gamma_{\text{accept}} \\
0 & \text{if } \text{invalid or } \text{NIS} > \gamma_{\text{reject}}
\end{cases}$$

$$\text{State}_k = \text{GNSS\_HEALTHY} \iff (\text{State}_{k-1} == \text{GNSS\_RECOVERING}) \land (M_{\text{valid}, k} \ge 3)$$

This strictly eliminates high-frequency oscillation between `HEALTHY` and `OUTAGE` under urban multipath or tree canopy shadows.
