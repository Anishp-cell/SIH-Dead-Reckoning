# Phase 8 Corrective Actions & Architectural Upgrades

**Document**: `docs/phase8/correction_pass/01_CORRECTIVE_ACTIONS.md`  
**Purpose**: Detail the systematic corrective actions executed to achieve mathematical rigor and scientific reproducibility.

---

## 1. Unified Master Benchmark Runner
- Built `Data_details/src/phase8/phase8_rebenchmark.py` as the single authoritative entry point.
- Hardcoded performance numbers removed from all source code and documentation.
- All benchmark metrics exported to machine-readable CSV and JSON artifacts.

## 2. Zero-Leakage Mode B Operational Initializer
- Enforced strict reference isolation:
  $$\mathbf{p}_0 = \mathbf{z}_{\text{gnss}, 0}, \quad \mathbf{v}_0 = \mathbf{v}_{\text{gnss}, 0}, \quad \mathbf{q}_0 = \text{leveling}(\mathbf{f}_0)$$
- Added automated `assert_no_reference_leakage()` check in pipeline.

## 3. Causal Multi-Rate Timestamp Architecture
- Explicit 10 Hz IMU propagation with 1 Hz GNSS arrival updates.
- 2D horizontal GNSS velocity update eliminates unmeasured vertical velocity fabrication.

## 4. Covariance-Consistent Soft Recovery
- Implemented effective measurement covariance inflation:
  $$R_{\text{eff}}(t) = \frac{R}{\alpha(t)}, \quad \alpha(t) = \alpha_{\min} + (1 - \alpha_{\min}) \frac{t - t_{\text{rec}}}{T_{\text{rec}}}$$
  $$S = H P^- H^T + R_{\text{eff}}$$
  $$K = P^- H^T S^{-1}$$
  $$P^+ = (I - K H) P^- (I - K H)^T + K R_{\text{eff}} K^T$$

## 5. Unclipped Chi-Square NIS Gating Feeding Outage State Machine
- Raw unclipped innovation evaluated against 3-DOF and 2-DOF $\chi^2$ bounds before any clipping.
- State machine transitions (`HEALTHY` -> `SUSPECT` -> `OUTAGE` -> `RECOVERING`) driven by true gating outcomes.
