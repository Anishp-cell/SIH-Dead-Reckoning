# Phase 6 Failure Analysis & Limitations

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase6/PHASE_6_FAILURE_ANALYSIS.md`  

---

## 1. Overview & No-False-Claims Statement

Phase 6 achieved massive improvements in dead reckoning accuracy:
- Lateral velocity RMSE reduced by **173x** (from $11.62\text{ m/s}$ to $0.07\text{ m/s}$).
- Stationary position creep reduced by **59x** (from $426.2\text{ m}$ to $7.2\text{ m}$).
- 60s outage trajectory drift reduced by **4.55x (78.0% error reduction)** (from $269.1\%$ down to $59.1\%$).

However, in accordance with the project's **No-False-Claims Policy**, this document thoroughly dissects where vehicle physics constraints succeed, where they degrade, and what failure modes remain strictly unsolvable without external map constraints (Phase 7).

---

## 2. Investigated Failure Modes

### 2.1 Failure Mode 1: Long-Duration Outage Heading Drift (120 s Blackout)
- **Observation**: While Phase 6 reduced 120s drift from $723.9\%$ ($4,788\text{ m}$) down to $302.9\%$ ($2,003\text{ m}$), the trajectory still exhibits significant error over 2 minutes of driving.
- **Root Cause Analysis**:
  - During the moving portion of the 120s outage, the vehicle drove straight for long stretches without sharp turns.
  - As proven in [`PHASE_6_OBSERVABILITY.md`](file:///d:/python/SIH%2026%20ISRO/docs/phase6/PHASE_6_OBSERVABILITY.md), absolute yaw $\psi$ is **unobservable** during straight, constant-velocity motion.
  - While NHC forces $v_{\text{lat}} \approx 0$, it only aligns the velocity vector with the current estimated heading. If the gyroscope bias slowly shifts the heading by $10^\circ$, NHC obligingly points the velocity $10^\circ$ off the true road!
  - Over 120s, unobserved gyro bias integrated into $63.1^\circ$ of heading error, causing cross-track displacement.
- **Why Vehicle Physics Alone Cannot Fix This**:
  - No internal physics constraint can know whether a straight road is oriented at $45^\circ$ or $55^\circ$ East of North.
  - **Phase 7 Necessity**: Road network alignment via OpenStreetMap (OSM) map matching is mandatory to clamp long-term heading drift to valid road centerlines.

---

### 2.2 Failure Mode 2: Cornering Slip vs NHC Conflict
- **Observation**: During high-speed turns, centrifugal acceleration pushes the tires into a slip angle $\alpha = \arctan(v_y / v_x) \approx 2^\circ - 4^\circ$.
- **Mitigation Performance**:
  - Fixed NHC forced $v_{\text{lat}} = 0$, generating high innovation residuals ($\nu_{\text{lat}} > 1.5\text{ m/s}$) that slightly distorted attitude.
  - The Phase 6 Disturbance Detector successfully identified lateral cornering acceleration and adaptively inflated $\sigma_{\text{lat}}$ from $0.20\text{ m/s}$ up to $0.80\text{ m/s}$, maintaining $100\%$ NIS acceptance across all turning maneuvers without filter divergence.

---

### 2.3 Failure Mode 3: Crawling Traffic vs Stationary Detection
- **Observation**: If a vehicle creeps forward at $0.2\text{ m/s}$ in heavy congestion, acceleration variance is nearly zero and specific force equals gravity.
- **Mitigation Performance**:
  - The multi-signal detector prevented false ZUPT clamping by enforcing the joint condition $C_v = (\hat{v}_{\text{fwd}} < 0.5\text{ m/s})$ and requiring $N_{\text{enter}} \ge 5$ consecutive samples ($0.5\text{ s}$).
  - During genuine moving intervals, ZUPT was rejected $100\%$ of the time, avoiding catastrophic velocity locking.

---

### 2.4 Failure Mode 4: Smartphone Mounting Lever-Arm Uncertainty
- **Observation**: The smartphone is mounted in a dashboard cradle offset from the vehicle's rear axle/center of rotation by an unknown vector $\mathbf{r}_{ib}^b \approx [1.0, 0.2, 0.4]^T\text{ m}$.
- **Effect**: During yaw rotation $\dot{\psi}$, tangential velocity $\mathbf{v}_{\text{lever}} = \boldsymbol{\omega} \times \mathbf{r}_{ib}^b$ adds a small lateral velocity $v_{\text{lat, lever}} \approx \dot{\psi} r_x \approx (0.2\text{ rad/s})(1.0\text{ m}) = 0.20\text{ m/s}$.
- **Resolution**:
  - Because exact survey calipers are unavailable in IO-VNBD, Phase 6 exposes $\mathbf{r}_{ib}^b$ as a configuration parameter (default $\mathbf{0}$).
  - The adaptive disturbance weighting absorbs the $0.20\text{ m/s}$ lever-arm velocity during turns by expanding $\sigma_{\text{lat}}$, preventing artificial attitude distortion.

---

## 3. Summary of Remaining Deficiencies (Phase 7 Roadmap)

| Deficiency | Cause | Phase 6 Status | Phase 7 Solution |
|:-----------|:------|:--------------:|:-----------------|
| **Global Heading Drift** | Straight motion yaw unobservability | Substantially reduced, but drifts over 120s | OpenStreetMap (OSM) heading snapping |
| **Cross-Track Position Offset** | Integrated heading errors | Reduced 4.55x at 60s, but drifts over long runs | Road centerline projection & topological map matching |
| **Absolute Initial Position** | Internal sensors only measure displacement | Requires initial GNSS anchor | Map-matched road segment identification |
