# Phase 8 Scientific Benchmark Protocol & Scenario Specifications

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/09_BENCHMARK_PROTOCOL.md`  

---

## 1. Objective and Provenance Standard

This protocol defines the exact, repeatable scientific benchmarking procedure for evaluating Phase 8 (GNSS + INS Fusion, Outage Detection, and Smooth Recovery).

### Data Provenance Taxonomy
All evaluation data sources must be tagged with explicit provenance:
1. `REAL_GNSS`: Raw telemetry captured from smartphone GNSS receiver (`gps_lat`, `gps_lon`, `gps_acc_m`, `gps_sats`, `gps_speed_mps`, `gps_heading_deg` from `s1_filtered_causal_imu.csv`).
2. `SYNTHETIC_GNSS`: Controlled signal degradation or blackout injected into GNSS observations according to predefined scenarios, with known ground truth for causal testing.
3. `REFERENCE_ONLY`: Millimeter-grade RTK / VBOX reference trajectory (`V-S1.csv`). Used strictly for post-run accuracy assessment; NEVER exposed to the filter or estimator.

---

## 2. Benchmark Scenarios (A through H)

| Scenario ID | Name | Duration / Profile | Injected Degradation | Target Behavior |
|---|---|---|---|---|
| **Scenario A** | Continuous Healthy GNSS | Entire run (0s outage) | Nominal smartphone GNSS noise ($\sigma \approx 2-3\text{ m}$) | Filter stays in `GNSS_HEALTHY`; continuous 15-state calibration. |
| **Scenario B** | Short Underpass / Bridge | 10 s outage ($t \in [300, 310]$) | Total loss of GNSS satellites ($N_{\text{sats}} = 0$) | Instant detection at $N_{\text{fail}} = 3$ (0.3s); seamless dead reckoning. |
| **Scenario C** | Urban Canyon Outage | 30 s outage ($t \in [300, 330]$) | Total loss of GNSS signals | Smooth drift growth bounded by Phase 7 map constraints; soft recovery. |
| **Scenario D** | Standard Tunnel Outage | 60 s outage ($t \in [300, 360]$) | Total loss of GNSS signals | Controlled dead reckoning; along-track drift bounded; zero teleportation on exit. |
| **Scenario E** | Extreme Deep Blackout | 120 s outage ($t \in [300, 420]$) | Total loss of GNSS signals | Stress test for longitudinal stability; map topology maintains road containment. |
| **Scenario F** | Severe Multipath / Urban Noise | 60 s degraded ($t \in [200, 260]$) | High variance ($\sigma_p = 15.0\text{ m}$), degraded DOP | NIS gating rejects or downweights updates; filter prevents track distortion. |
| **Scenario G** | Multi-Satellite Jump Fault | Instantaneous at $t = 350\text{ s}$ | $25.0\text{ m}$ step jump in position | Outlier rejection via 3-DOF Chi-square test ($\chi_3^2 > 11.345$); zero step change. |
| **Scenario H** | Intermittent Chattering Outages | 60 s cyclic ($t \in [250, 310]$) | 5 s outage / 5 s recovery cyclic | State machine hysteresis prevents rapid state thrashing; smooth convergence. |

---

## 3. Quantitative Evaluation Metrics

### 3.1 Tracking Accuracy
1. **Max Horizontal Error ($e_{\max}$)**:
   $$e_{\max} = \max_{t} \|\hat{\mathbf{p}}_{E,N}(t) - \mathbf{p}_{\text{ref},E,N}(t)\|$$
2. **Mean Absolute Error ($\text{MAE}_p$)**:
   $$\text{MAE}_p = \frac{1}{N} \sum_{k=1}^N \|\hat{\mathbf{p}}_{E,N}(t_k) - \mathbf{p}_{\text{ref},E,N}(t_k)\|$$
3. **Drift Rate during Outage ($\%_{\text{drift}}$)**:
   $$\%_{\text{drift}} = \frac{\|\hat{\mathbf{p}}(t_{\text{end}}) - \mathbf{p}_{\text{ref}}(t_{\text{end}})\| - \|\hat{\mathbf{p}}(t_{\text{start}}) - \mathbf{p}_{\text{ref}}(t_{\text{start}})\|}{d_{\text{traveled}}} \times 100\%$$

### 3.2 Recovery Continuity Metrics (Zero-Teleportation Guarantee)
Evaluated across the recovery window $[t_{\text{recovery}}^-, t_{\text{recovery}}^+]$ where $\Delta t = 0.1\text{ s}$:
1. **Position Step Discontinuity ($\Delta p_{\text{step}}$)**:
   $$\Delta p_{\text{step}} = \|\hat{\mathbf{p}}(t_{\text{recovery}}^+) - \hat{\mathbf{p}}(t_{\text{recovery}}^-)\| \le 3.50\text{ m}$$
2. **Velocity Discontinuity ($\Delta v_{\text{step}}$)**:
   $$\Delta v_{\text{step}} = \|\hat{\mathbf{v}}(t_{\text{recovery}}^+) - \hat{\mathbf{v}}(t_{\text{recovery}}^-)\| \le 1.00\text{ m/s}$$
3. **Heading Discontinuity ($\Delta \psi_{\text{step}}$)**:
   $$\Delta \psi_{\text{step}} = |\hat{\psi}(t_{\text{recovery}}^+) - \hat{\psi}(t_{\text{recovery}}^-)| \le 3.0^\circ$$
4. **Pseudo-Acceleration Spike ($a_{\text{pseudo}}$)**:
   $$a_{\text{pseudo}} = \frac{\|\hat{\mathbf{v}}(t_{\text{recovery}}^+) - \hat{\mathbf{v}}(t_{\text{recovery}}^-)\|}{\Delta t} \le 2.50\text{ m/s}^2$$

---

## 4. Ablation Study Protocol

To isolate and prove the contribution of each algorithmic innovation in Phase 8:
1. **Full Phase 8 Stack**: GNSS + Map Matching + Vehicle Physics (NHC/ZUPT) + AI Speed Model + Soft Recovery.
2. **Ablation 1 (No Map)**: GNSS + Vehicle Physics + AI Speed (pure sensor fusion, no road constraints during blackout).
3. **Ablation 2 (No Quality Gating)**: Naive GNSS fusion accepting all measurements without NIS or rate-of-change rejection.
4. **Ablation 3 (Hard Reset Recovery)**: Snapping position directly to GNSS fix on recovery (demonstrating why soft recovery is essential to avoid infinite acceleration spikes).
5. **Ablation 4 (Pure Phase 7 Baseline)**: Map-aided dead reckoning without GNSS fusion during the outage period.
