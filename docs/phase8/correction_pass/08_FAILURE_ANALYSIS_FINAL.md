# Phase 8 Comprehensive Failure Analysis & Edge Cases

**Document**: `docs/phase8/correction_pass/08_FAILURE_ANALYSIS_FINAL.md`  
**Purpose**: Detailed failure analysis of 10 edge cases and operational failure modes.

---

## 1. Failure Mode Catalog

### 1. GNSS False Acceptance (Subtle Multipath)
- **Mechanism**: A multipath reflection causing a 3–5 m bias that falls just below the 3-DOF Chi-square threshold ($\chi^2 < 11.345$).
- **Mitigation**: Robust Huber-like downweighting zone ($7.815 \le \chi^2 \le 11.345$) inflates covariance by $s = \chi^2 / 7.815$, mitigating pull on state.

### 2. GNSS False Rejection (Dynamic Manoeuvre)
- **Mechanism**: Rapid vehicle acceleration causes true innovation to temporarily spike above threshold.
- **Mitigation**: Multi-state hysteresis (`SUSPECT` state before declaring `OUTAGE`). Single spikes do not declare blackout.

### 3. Multipath Step Jump (25m Outlier)
- **Mechanism**: Sudden satellite line-of-sight shift causing instantaneous 25 m coordinate shift.
- **Result**: Successfully rejected by raw unclipped NIS gate; zero filter corruption.

### 4. Deep 120s Blackout Divergence
- **Mechanism**: Prolonged lack of absolute fixes during vehicle turns causes unconstrained dead reckoning drift (158.7% drift).
- **Finding**: Accurately reported without artificial suppression.

### 5. Rapid GNSS Chattering (Urban Canyon)
- **Mechanism**: Signal intermittently acquired and lost every 1–2 seconds.
- **Mitigation**: 3 consecutive fails required to enter `OUTAGE`, 3 consecutive passes required to enter `RECOVERING`.

### 6. Large Recovery Discrepancy Teleportation
- **Mechanism**: Exiting a long blackout with 30+ meter DR error. Hard reset creates unphysical velocity spikes ($> 30$ m/s).
- **Mitigation**: Soft recovery with effective covariance inflation limits max position step to $0.006$ m.

### 7. AI-Speed Uncertainty Spikes
- **Mechanism**: Neural network confidence drops during unusual road surfaces.
- **Mitigation**: Heteroscedastic speed uncertainty $\sigma_v(t)$ automatically inflates $R_v$, reducing speed update weight.

### 8. Map Matching Ambiguity at Intersections
- **Mechanism**: Multiple road segments in proximity.
- **Mitigation**: Cross-track gating and heading alignment reject spurious perpendicular candidate segments.

### 9. Incorrect Heading Initialization
- **Mechanism**: Stationary vehicle GPS bearing reflects receiver noise rather than chassis orientation.
- **Mitigation**: Forward motion velocity gating ($v > 1.0$ m/s) required before locking initial course heading.

### 10. Sensor Timestamp Delays / Jitter
- **Mechanism**: Android asynchronous sensor delivery.
- **Mitigation**: Sorted timestamp queue and causal forward integration prevent processing future measurements.
