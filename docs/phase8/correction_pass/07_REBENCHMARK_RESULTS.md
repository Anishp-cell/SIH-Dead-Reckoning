# Phase 8 Re-Benchmark Master Results

**Document**: `docs/phase8/correction_pass/07_REBENCHMARK_RESULTS.md`  
**Purpose**: Compilation of all authoritative re-benchmarking tables.

---

## 1. Durations Comparison
| Duration S | Window Duration S | Distance Traveled Outage M | P6 Drift Pct | P7 Drift Pct | P8 Drift Pct | P6 Dr End Err M | P7 Map Dr End Err M | P7 Map Dr Cross Err M | P8 Hybrid Outage Max Err M | P8 Hybrid Outage Cross Err M | P8 Post Recovery Err M | Recovery P Step M | Recovery V Step Mps | Recovery Pseudo Accel Mps2 | Zero Teleportation Passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 45.000 | 99.000 | 16.800 | 16.700 | 30.700 | 16.620 | 16.570 | 0.950 | 64.830 | 1.380 | 21.850 | 0.696 | 0.064 | 0.635 | True |
| 30 | 65.000 | 191.800 | 41.000 | 40.900 | 16.900 | 78.590 | 78.360 | 1.030 | 64.830 | 1.380 | 2.930 | 0.027 | 0.003 | 0.027 | True |
| 60 | 95.000 | 191.800 | 41.100 | 40.900 | 16.800 | 78.740 | 78.500 | 1.030 | 64.830 | 1.380 | 26.980 | 0.006 | 0.006 | 0.056 | True |
| 120 | 165.000 | 661.400 | 154.200 | 161.800 | 158.700 | 1020.1 | 1070.2 | 916.150 | 1049.4 | 903.270 | 5098.9 | 7.642 | 0.980 | 9.700 | False |

## 2. Multi-Location Randomization
| Location | Duration S | Outage Start S | Outage Max Err M | Outage Cross Err M | Post Recovery Err M |
| --- | --- | --- | --- | --- | --- |
| Loc_1_Stop_and_Go_4600s | 10 | 4600.0 | 64.830 | 1.380 | 21.850 |
| Loc_1_Stop_and_Go_4600s | 30 | 4600.0 | 64.830 | 1.380 | 2.930 |
| Loc_1_Stop_and_Go_4600s | 60 | 4600.0 | 64.830 | 1.380 | 26.980 |
| Loc_2_Cruising_4750s | 10 | 4750.0 | 693.240 | 323.840 | 2728.3 |
| Loc_2_Cruising_4750s | 30 | 4750.0 | 2732.5 | 2279.7 | 1879.3 |
| Loc_2_Cruising_4750s | 60 | 4750.0 | 2827.0 | 2351.9 | 5725.2 |
| Loc_3_Approaching_Turn_4900s | 10 | 4900.0 | 2454.1 | 939.590 | 8359.9 |
| Loc_3_Approaching_Turn_4900s | 30 | 4900.0 | 8397.7 | 5299.6 | 17709.3 |
| Loc_3_Approaching_Turn_4900s | 60 | 4900.0 | 23755.2 | 17981.2 | 34983.7 |

## 3. Scenarios Comparison
| Scenario | Max Error M | Mean Error M | Rmse M | Max Along Track M | Max Cross Track M | Post Recovery Err M |
| --- | --- | --- | --- | --- | --- | --- |
| Scenario A (Healthy GNSS Continuous) | 74.720 | 40.510 | 43.420 | 74.680 | 4.140 | 6.430 |
| Scenario B (Short 10s Outage) | 74.720 | 25.310 | 29.300 | 74.680 | 2.340 | 7.950 |
| Scenario C (Medium 30s Outage) | 74.720 | 22.690 | 27.460 | 74.680 | 2.340 | 1.830 |
| Scenario D (Standard 60s Outage) | 74.720 | 27.160 | 29.880 | 74.680 | 2.940 | 26.980 |
| Scenario E (Deep 120s Outage) | 3879.5 | 407.730 | 932.330 | 3744.4 | 1014.6 | 3879.5 |
| Scenario F (GNSS Position Outlier 25m) | 74.720 | 40.560 | 43.430 | 74.680 | 4.180 | 9.240 |
| Scenario G (Chattering Outages 5s) | 74.720 | 27.910 | 30.270 | 74.680 | 2.880 | 24.240 |
| Scenario H (Recovery after Large DR Error) | 74.670 | 27.220 | 29.940 | 74.630 | 2.970 | 26.740 |

## 4. Ablations & Fusion Modes
| Configuration | Max Error M | Mean Error M | Max Along Track M | Max Cross Track M | Recovery P Step M | Recovery V Step Mps | Pseudo Accel Mps2 | Zero Teleportation Passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 7 (Pure DR + Map Constraints) | 80.110 | 22.390 | 80.070 | 2.940 | 0.007 | 0.006 | 0.056 | True |
| Phase 7 + GNSS Position | 77.100 | 24.450 | 77.060 | 2.420 | 0.007 | 0.006 | 0.056 | True |
| Phase 7 + GNSS Velocity | 78.300 | 23.370 | 78.260 | 2.900 | 0.007 | 0.006 | 0.056 | True |
| Phase 7 + Both (Unconstrained Reset) | 145.290 | 84.980 | 145.290 | 1.450 | 134.493 | 0.009 | 0.090 | False |
| Final Robust Fusion (Full Phase 8) | 74.720 | 27.160 | 74.680 | 2.940 | 0.006 | 0.006 | 0.056 | True |
| Ablation: No Map (Phase 8) | 74.470 | 29.110 | 74.250 | 13.760 | 0.006 | 0.006 | 0.056 | True |
| Ablation: No Quality Gating | 145.290 | 106.430 | 145.290 | 7.800 | 0.001 | 0.006 | 0.056 | True |
| Ablation: Hard Reset Recovery | 74.720 | 24.830 | 74.680 | 2.340 | 32.181 | 0.010 | 0.100 | False |

## 5. Trajectory Continuity Audit
| Metric | Soft Recovery | Hard Reset | Project Threshold | Status |
| --- | --- | --- | --- | --- |
| Max Position Step (m) | 0.006 | 32.181 | 3.500 | PASSED |
| Max Velocity Step (m/s) | 0.006 | 0.010 | 1.000 | PASSED |
| Pseudo-Accel Spike (m/s^2) | 0.056 | 0.100 | 2.500 | PASSED |
