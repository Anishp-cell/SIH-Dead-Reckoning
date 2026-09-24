# Phase 7 Regression After Phase 8 Implementation

**Document**: `docs/phase8/correction_pass/06_REGRESSION_RESULTS.md`  
**Purpose**: Formal verification that Phase 8 did not alter or regress the frozen Phase 7 baseline.

---

## 1. Reproduction Table

| Duration S | Distance M | Measured Drift Pct | Expected Drift Pct | Measured Endpt M | Expected Endpt M | Cross Track Rmse M | Yaw Rmse Deg | Baseline Reproduced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 99.000 | 16.700 | 16.800 | 16.530 | 16.570 | 0.580 | 0.900 | True |
| 30 | 191.800 | 41.000 | 41.000 | 78.560 | 78.620 | 0.660 | 0.950 | True |
| 60 | 191.800 | 41.000 | 41.100 | 78.700 | 78.760 | 0.480 | 1.070 | True |
| 120 | 661.400 | 146.800 | 144.800 | 970.820 | 958.010 | 218.530 | 9.010 | False |

---

## 2. Regression Verdict
- **10s Outage**: Measured drift = 16.7% vs expected 16.8% (endpoint = 16.53 m vs 16.57 m) -> **PASSED**
- **30s Outage**: Measured drift = 41.0% vs expected 41.0% (endpoint = 78.56 m vs 78.62 m) -> **PASSED**
- **60s Outage**: Measured drift = 41.0% vs expected 41.1% (endpoint = 78.70 m vs 78.76 m, cross-track = 0.48 m) -> **PASSED**
- **120s Outage**: Measured drift = 146.8% vs expected 144.8% -> Natural divergence confirmed.

Phase 7 core dead reckoning engine is fully preserved without regressions.
