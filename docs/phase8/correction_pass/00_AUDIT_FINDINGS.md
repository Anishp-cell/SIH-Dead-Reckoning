# Phase 8 Audit Findings: Discrepancies and Root Causes

**Document**: `docs/phase8/correction_pass/00_AUDIT_FINDINGS.md`  
**Purpose**: Document all architectural, mathematical, and procedural discrepancies identified in Phase 8 prior to correction.

---

## 1. Discrepancy Summary

### Finding 1: Reference-State Leakage in Baseline Operational Benchmarks
- **Initial State**: In early diagnostic runs, the filter state was occasionally initialized using the OXTS/RTK VBOX reference position, velocity, and orientation.
- **Scientific Impact**: Injecting ground truth invalidates operational claims, as real smartphone navigation systems do not have access to RTK truth.
- **Root Cause**: Convenience debugging code in test scripts bypassed the operational initializer.

### Finding 2: Static Heading Initialization Mismatch
- **Initial State**: Initializing at $t = 4580$s selected a stationary epoch where smartphone GPS heading was $257.82^\circ$ (noise during standstill), whereas true road heading was $349.70^\circ$ ($91.88^\circ$ error).
- **Scientific Impact**: Caused initial filter divergence before vehicle motion aligned the heading.
- **Correction**: Initialized pre-window at $t = 4585.0$s during forward movement ($v > 1.0$ m/s, heading $350.94^\circ$).

### Finding 3: Dataset Provenance Distinction (Real vs Synthetic)
- **Initial State**: `S-S1.csv` is smartphone GPS telemetry where coordinates are held constant for 4 seconds at a time while the car is moving.
- **Scientific Impact**: Calling stepped, quantized data "real 1 Hz GNSS" caused false innovation spikes.
- **Correction**: Structured `SyntheticGNSSGenerator` from `gnss/synthetic.py` produces realistic 1 Hz GNSS telemetry ($\sigma_p = 1.8$m, $\sigma_v = 0.25$ m/s) with rigorous provenance labeling (`SYNTHETIC_GNSS`).

### Finding 4: Soft Recovery Mathematics vs Code Mismatch
- **Initial State**: Documentation described covariance inflation, but code scaled $\delta x$ post-hoc by $\alpha$ without modifying $P^+$, creating mathematical inconsistency between state correction and covariance.
- **Correction**: Implemented true effective covariance inflation $R_{{\text{eff}}} = R / \alpha(t)$, recomputing Kalman gain $K$ and Joseph covariance update with $R_{{\text{eff}}}$.

### Finding 5: Hard Reset vs Soft Recovery Benchmark Discrepancy
- **Initial State**: Hard reset baseline was previously tested under different noise seeds, preventing direct 1-to-1 comparison.
- **Correction**: Soft recovery and Hard reset evaluated on identical trajectory, IMU measurements, and random seeds.

### Finding 6: Truncated Benchmark Windows
- **Initial State**: Benchmarks previously terminated immediately at outage end, preventing post-outage recovery settling evaluation.
- **Correction**: Mandatory pre-window (15s) + exact outage duration + post-recovery window (20–30s).
