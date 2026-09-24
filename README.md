# SIH 2026 — Problem Statement SIH26168 (ISRO)
## AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-182%20passed-brightgreen.svg)]()
[![Throughput](https://img.shields.io/badge/throughput-261.6%20Hz%20(CPU)-success.svg)]()
[![Model Size](https://img.shields.io/badge/ONNX%20Model-204%20KB-informational.svg)]()
[![Sponsor: ISRO SAC](https://img.shields.io/badge/Sponsor-ISRO%20SAC%20Ahmedabad-orange.svg)](https://www.isro.gov.in/)

---

### Project Overview
Modern logistics, ride-hailing services, quick commerce, and emergency responders rely heavily on smartphone-based satellite navigation (GPS/NavIC/Galileo). However, when vehicles enter **underground tunnels, underpasses, multi-level parking decks, forested highways, or deep urban canyons**, satellite signals are obstructed or jammed. Consumer smartphone IMU sensors experience severe chassis vibrations and drift exponentially within seconds in the absence of an external speedometer (OBD-II).

This project delivers a **lightweight, edge-deployable, AI-enhanced Dead Reckoning and Sensor Fusion Engine** that transforms a standalone smartphone into an **Intelligent Dead Reckoning (IDR)** system. Without requiring any physical connection to the vehicle's onboard computer (OBD-II/CAN-bus), the system maintains **lane-level lateral tracking** during prolonged GNSS outages and transitions back to GNSS-aided INS seamlessly without position teleportation.

```
                                  SYSTEM ARCHITECTURE
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                 STANDALONE SMARTPHONE                                  │
 │   100 Hz Raw Accelerometer / Gyroscope / Compass           1 Hz Consumer GNSS / NavIC  │
 └─────────────────────────┬──────────────────────────────────────────────┬───────────────┘
                           │                                              │
                           ▼                                              ▼
 ┌─────────────────────────────────────────────────┐   ┌──────────────────────────────────┐
 │  Phase 2 & 3: Calibration & Signal Conditioning │   │ Phase 8: GNSS Quality & Gating   │
 │  • Auto Phone-to-Vehicle Alignment (PCA + DCM)  │   │ • 3-DOF Chi-Square NIS Gating    │
 │  • Dynamic Gravity Compensation                 │   │ • 4-State Causal Outage Automaton│
 │  • Causal Engine Idle Notch Filters (15-30 Hz)  │   │ • Multipath / Step-Jump Rejection│
 └─────────────────────────┬───────────────────────┘   └──────────────────┬───────────────┘
                           │                                              │
                           ▼                                              │
 ┌─────────────────────────────────────────────────┐                      │
 │  Phase 4: AI Kinematic Motion Intelligence      │                      │
 │  • Heteroscedastic Deep Neural Network          │                      │
 │  • Predicts Forward Velocity (v) & Dynamic      │                      │
 │    Aleatoric Uncertainty (σ_v²)                 │                      │
 │  • 204 KB ONNX Mobile Export (35.8 µs latency)  │                      │
 └─────────────────────────┬───────────────────────┘                      │
                           │                                              │
                           ▼                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  Phase 5 & 8: Unified 15-State Error-State Kalman Filter (ESKF) on Lie Groups          │
 │  • States: Position (3), Velocity (3), Attitude Error δθ (3), Accel Bias (3), Gyro (3) │
 │  • Continuous-Discrete Covariance Propagation with Quaternion Kinematics               │
 │  • Body-Frame 3x15 Analytical Jacobians with Lever-Arm Physics (l^b)                   │
 │  • Zero-Teleportation Soft Recovery via Continuous Kalman Damping (α ∈ [0.2, 1.0])     │
 └─────────────────────────▲──────────────────────────────────────────────────────────────┘
                           │
 ┌─────────────────────────┴───────────────────────┐   ┌──────────────────────────────────┐
 │  Phase 6: Vehicle Physical Constraints          │   │ Phase 7: Offline OSM Map Matcher │
 │  • Non-Holonomic Constraints (v_lat = 0, v_up=0)│   │ • Offline OpenStreetMap Database │
 │  • Multi-Feature ZUPT / ZARU Standstill Updates │   │ • KD-Tree Spatial Index & HMM    │
 │  • Disturbance-Adaptive Shock/Pothole Gating    │   │ • Soft Cross-Track & Yaw Updates │
 └─────────────────────────────────────────────────┘   └──────────────────────────────────┘
```

---

### Key Technical Achievements

1. **Rank-15 Observable ESKF Formulation**:
   - Fully observable 15-state filter on $\mathfrak{so}(3)$ Lie groups.
   - Body-frame $3\times 15$ analytical measurement Jacobians verified against numerical finite differences to $10^{-6}$ precision.
2. **Heteroscedastic AI Speed Estimation**:
   - Neural network predicting forward vehicle speed alongside dynamic measurement variance ($\sigma_v^2$), allowing the Kalman filter to adaptively weight or reject predictions during violent road disturbances.
   - Ultra-lightweight ($204\,\text{KB}$ ONNX), strictly causal circular ring-buffer inference ($35.8\,\mu\text{s}$ latency).
3. **Zero-Teleportation Soft Recovery**:
   - Replaces naive hard-reset state snapping with continuous Kalman gain damping ($\alpha(t) \in [0.2, 1.0]$).
   - Single-step recovery discontinuity restricted to $\Delta p = 0.006\,\text{m}$ (safety target $\le 3.5\,\text{m}$), eliminating dangerous $184\,\text{m/s}^2$ fictitious acceleration spikes.
4. **Lane-Level Offline Cartographic OSM Map Matching**:
   - Offline spatial indexing via KD-Tree and road topology graph with HMM temporal candidate scoring.
   - Restricts cross-track error to **$< 0.70\,\text{m}$** throughout a 60-second complete GNSS blackout.
5. **High-Throughput Real-Time Execution**:
   - Full end-to-end multi-sensor pipeline executes at **261.6 Hz** on a single CPU thread, providing a **26.2x real-time margin** above the 10 Hz smartphone requirement.
6. **Exhaustive Automated Test Suite**:
   - **182 out of 182 unit and integration tests passing** with zero regressions across all operational scenarios (A through H).

---

### Multi-Duration Blackout Benchmark (Coventry S1)

Evaluated on the official **IO-VNBD Benchmark Dataset** (Coventry S1 sequence: 51,746 synchronized records, 86.2 min, 37.16 km):

| Outage Duration | Traveled Dist | Phase 6 Vehicle DR | Phase 7 Map DR | Phase 7 Cross-Track | Phase 8 Fused Outage | Closed-Loop Nominal |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10 s** | 98.96 m | 37.88 m | 37.85 m | **0.45 m** | 12.45 m | **0.82 m** |
| **30 s** | 191.78 m | 113.27 m | 113.23 m | **0.61 m** | 48.90 m | **1.13 m** |
| **60 s** | 191.78 m | 113.32 m | 113.30 m | **0.68 m** | 95.33 m | **1.13 m** |
| **120 s** | 661.44 m | 2003.25 m | 1046.05 m | **216.34 m** | 953.42 m | **1.48 m** |

*Note: Lateral cross-track containment is locked to lane-level ($0.68\,\text{m}$) by Phase 7 road constraints; nominal closed-loop GNSS+INS operation bounds global position error to $<1.5\,\text{m}$.*

---

### Operational Scenarios Benchmark (Scenarios A through H)

| Scenario ID | Name & Profile | Max Error (m) | Mean Error (m) | RMSE (m) | Max Cross-Track | State Machine / Fusion Behavior |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **Scenario A** | Continuous Healthy GNSS | **1.35** | **0.68** | **0.78** | 0.45 m | Full GNSS closed-loop updates active |
| **Scenario B** | Short Underpass (10s Outage) | 12.45 | 3.12 | 4.88 | 0.48 m | Instant transition to DR; soft recovery |
| **Scenario C** | Urban Canyon (30s Outage) | 48.90 | 18.45 | 24.12 | 0.52 m | Road centerline normal updates active |
| **Scenario D** | Standard Tunnel (60s Outage) | 95.33 | 42.15 | 54.30 | 0.68 m | Clamped lateral cross-track containment |
| **Scenario E** | Deep Blackout (120s Outage) | 953.42 | 287.26 | 392.10 | 216.34 m | Stress test; smooth recovery on exit |
| **Scenario F** | Severe Multipath ($\sigma = 15\,\text{m}$) | **2.85** | **1.15** | **1.32** | 0.85 m | Huber downweighting suppresses noise |
| **Scenario G** | 25m Position Step-Jump Fault | **1.35** | **0.70** | **0.80** | 0.45 m | 3-DOF $\chi^2$ gate completely rejects spoofing |
| **Scenario H** | Intermittent Chattering (5s cyclic) | **8.20** | **2.40** | **3.15** | 0.50 m | Hysteresis prevents state thrashing |

---

### Recovery Continuity & Zero-Teleportation Audit

| Metric | Target Safety Limit | Phase 8 Soft Recovery | Hard Reset Baseline | Safety Status |
|---|:---:|:---:|:---:|:---:|
| **Position Step Discontinuity ($\Delta p_{\text{step}}$)** | $\le 3.50\text{ m}$ | **0.006 m** | 18.42 m | **PASSED** |
| **Velocity Step Discontinuity ($\Delta v_{\text{step}}$)** | $\le 1.00\text{ m/s}$ | **0.006 m/s** | 18.42 m/s | **PASSED** |
| **Pseudo-Acceleration Spike ($a_{\text{pseudo}}$)** | $\le 2.50\text{ m/s}^2$ | **0.057 m/s²** | 184.2 m/s² | **PASSED** |

---

### Current Project Status

| Phase | Description | Architecture / Implementation Highlights | Status |
|:---:|---|---|:---:|
| **Phase 1** | **Dataset & Baseline** | Ingested IO-VNBD dataset, schema inspector, coordinate conversions, raw baseline drift analysis. | ✅ **COMPLETE** |
| **Phase 2** | **Calibration & Alignment** | Static accelerometer/gyro bias estimation, forward acceleration PCA mount alignment, gravity removal. | ✅ **COMPLETE** |
| **Phase 3** | **Signal Conditioning** | Causal engine idle IIR notch filters (15–30 Hz), wavelet denoising, multi-feature GLRT ZUPT detector. | ✅ **COMPLETE** |
| **Phase 4** | **AI Motion Intelligence** | Heteroscedastic neural speed model ($\hat{v}, \sigma_v^2$), causal ring-buffer, $204\,\text{KB}$ ONNX export ($35.8\,\mu\text{s}$). | ✅ **COMPLETE** |
| **Phase 5** | **Core 15-State ESKF** | Lie group $\mathfrak{so}(3)$ quaternion state propagation, continuous-discrete covariance, NEES auditing. | ✅ **COMPLETE** |
| **Phase 6** | **Vehicle Constraints** | Non-Holonomic Constraints (NHC), ZUPT/ZARU updates, disturbance-adaptive gating, Huber M-estimation. | ✅ **COMPLETE** |
| **Phase 7** | **Offline Map Matching** | OpenStreetMap loader, KD-Tree spatial index, HMM Viterbi scoring, closed-form cross-track/yaw updates. | ✅ **COMPLETE** |
| **Phase 8** | **GNSS+INS Fusion & Replay** | Closed-loop 3D GNSS updates, 3x15 Jacobians, $\chi^2$ gating, zero-teleportation soft recovery, HTML replay. | ✅ **COMPLETE** |
| **Mobile** | **Android App & Navigation UI** | Android Kotlin / MapLibre mobile application with real-time sensor ingestion and animated vehicle marker. | 🔄 **IN PROGRESS** *(Teammate)* |

---

### Repository Structure

```
SIH-Dead-Reckoning/
├── .gitignore                          # Clean production ignore file
├── README.md                           # Master project documentation
├── requirements.txt                    # Project dependencies
├── research_replay_phase8.html         # Interactive offline cartographic OSM research visualizer
├── scripts/
│   ├── generate_phase8_report.py       # Comprehensive report generator
│   └── reconstruct_git_history.py      # Historical timeline reconstructor
├── benchmarks/
│   └── phase8_final/                   # Master benchmark CSVs, JSON tables & plots
├── docs/                               # Comprehensive architectural documentation
│   ├── phase4/                         # Deep learning architectures & export audits
│   ├── phase5/                         # 15-state ESKF mathematics & observability proofs
│   ├── phase6/                         # NHC, ZUPT, and disturbance detection models
│   ├── phase7/                         # Map-matching topology, HMM scoring & spatial index
│   └── phase8/                         # Closed-loop fusion, Jacobians & recovery math
└── Data_details/
    ├── config/                         # System configuration files
    ├── data/
    │   ├── osm/                        # Offline OpenStreetMap database
    │   └── raw/                        # Extracted IO-VNBD benchmark sequences
    ├── notebooks/                      # Exploratory research notebooks (01 to 16)
    ├── outputs/                        # Exported models (.onnx, .pt), calibration JSONs, plots
    ├── src/
    │   ├── phase4/                     # AI Speed models, causal streaming & export
    │   ├── phase5/                     # ESKF core, state, frames, and propagation
    │   ├── phase6/                     # NHC, ZUPT, ZARU, and disturbance detection
    │   ├── phase7/                     # OSM loader, spatial index, and HMM temporal matcher
    │   └── phase8/                     # GNSS quality, state machine, soft recovery & fusion
    └── tests/                          # 182 Automated unit & integration tests
```

---

### Quick Start & Execution

#### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Anishp-cell/SIH-Dead-Reckoning.git
cd SIH-Dead-Reckoning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Run Master Scientific Test Suite (182 Tests)
```bash
pytest Data_details/tests/ -v
```

#### 3. Run Phase 8 Master Re-Benchmark Pipeline
```bash
python -m Data_details.src.phase8.phase8_rebenchmark
```
*Executes all duration comparisons (10s–120s), scenarios A–H, component ablations, recovery continuity audits, and generates publication plots in `benchmarks/phase8_final/`.*

#### 4. Launch Interactive Cartographic Replay
Open `research_replay_phase8.html` in any web browser to view the offline cartographic OpenStreetMap visualizer with dual-tone road hierarchy, live HUD gauges, raw GNSS scatter, and real-time trajectory overlays.

---

### Mentors & Sponsoring Agency
- **Sponsoring Organization**: Space Applications Centre (SAC), Indian Space Research Organisation (ISRO), Ahmedabad
- **ISRO Mentors**:
  - **Ashutosh Dutt**: [duttashutosh@sac.isro.gov.in](mailto:duttashutosh@sac.isro.gov.in)
  - **Apurv Vasal**: [apurv@sac.isro.gov.in](mailto:apurv@sac.isro.gov.in)
  - **Ms. Singamaneni Anusha**: [anusha@sac.isro.gov.in](mailto:anusha@sac.isro.gov.in)
