# Phase 8 Failure Modes, Edge Cases & Mitigation Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/11_FAILURE_ANALYSIS.md`  

---

## 1. Executive Summary

This document presents a rigorous fault and failure analysis for the Phase 8 hybrid GNSS/INS navigation system, examining failure modes, edge cases, and safety barriers across both sensor failure and filter anomaly regimes.

---

## 2. Failure Mode & Effects Analysis (FMEA) Matrix

| Fault ID | Hazard / Trigger | Impact on Estimator | Primary Defense Layer | Secondary Defense Layer |
|---|---|---|---|---|
| **FM-01** | Rapid Intermittent GNSS Chattering (Tunnel / Overpass) | Filter state thrashing between GNSS and DR | 4-state causal automaton with temporal hysteresis ($N_{\text{fail}} \ge 3$, $M_{\text{valid}} \ge 3$). | Recovery damping $\alpha(t)$ scaling. |
| **FM-02** | Urban Multipath with Deceptively Low Reported $\sigma_p$ | Track pulled off centerline into building | 3-DOF Chi-square NIS innovation gating ($\chi_3^2 \le 11.345$). | Phase 7 road topological normal constraint ($\mathbf{H}_{\text{map}}$). |
| **FM-03** | Multi-Satellite Constellation Step Jump ($> 20\text{ m}$) | Step discontinuity in trajectory | Physical rate-of-change velocity limit ($v_{\text{apparent}} \le 45\text{ m/s}$). | Max step clamp $\Delta p \le 3.5\text{ m}$. |
| **FM-04** | Extended Deep Blackout ($> 120\text{ s}$) with Speed Error | Along-track drift accumulates along road | Phase 6 ZUPT/ZARU pins drift during traffic halts. | Phase 7 road heading update constrains azimuth. |
| **FM-05** | Sampling Rate Asynchrony (10 Hz IMU vs 1 Hz GNSS) | Filter repeatedly updates static position, fighting forward motion | Duplicate sample detector suppresses updates unless coordinates update. | Velocity Doppler measurement validates motion. |
| **FM-06** | False-Outlier Lock-Out on Recovery Exit | Filter rejects valid GNSS because along-track error grew large | Adaptive Huber covariance inflation when jump detector confirms no physical step. | State machine `RECOVERING` state forces soft injection. |
| **FM-07** | Hard State Snapping on Recovery | Infinite acceleration spike causing crash in downstream ADAS | Zero-teleportation recovery manager enforces soft Kalman gain injection ($\mathbf{K} \boldsymbol{\nu}$). | Step clamping $\le 3.5\text{ m}$. |

---

## 3. Deep Dive: Edge Cases & Mitigations

### 3.1 Edge Case 1: The 1 Hz Repeated GNSS Trap
In consumer smartphones, GNSS telemetry is output at 1 Hz, while IMU data is logged at 10 Hz. When naive filters consume the CSV sample-by-sample, the same coordinates are fed 10 times in 1 second.
- **Consequence**: The filter perceives 9 zero-motion updates, artificially braking forward speed and inflating innovation.
- **Mitigation**: The Phase 8 streaming engine monitors geodetic coordinate delta:
  $$\Delta p = |lat_k - lat_{k-1}| + |lon_k - lon_{k-1}|$$
  If $\Delta p = 0$, the sample is classified as an intermediate IMU prediction step; GNSS measurement updates are applied strictly on true fix arrivals.

### 3.2 Edge Case 2: Deep Outage Topological Branch Ambiguity
At $t > 120\text{ s}$, if longitudinal drift reaches $\pm 100\text{ m}$, the vehicle may approach an intersection before the filter predicts arrival.
- **Consequence**: The spatial index query may evaluate road candidates belonging to diverging branches.
- **Mitigation**: The Phase 7 Markov transition model preserves posterior belief $B(c_t)$ over top-10 candidates. Even if longitudinal position lags, cross-track and heading likelihoods reject perpendicular intersecting roads. When GNSS returns, soft recovery resolves the branch unambiguously.
