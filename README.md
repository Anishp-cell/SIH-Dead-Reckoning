# SIH 2026 — Problem Statement SIH26168 (ISRO)
## AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation

### Project Objective
Develop a lightweight, edge-deployable intelligent navigation engine that allows a smartphone to continue estimating a vehicle's position during GNSS/GPS blackouts (tunnels, urban canyons, dense forests, parking structures) using smartphone IMU sensors, AI/ML, sensor fusion, vehicle kinematics, and offline road maps.

---

### Phase 1: IO-VNBD Dataset Understanding, Exploration & Baseline
Phase 1 implementation resides in [`Data_details/`](file:///d:/python/SIH%2026%20ISRO/Data_details):

- **Repository Inventory**: 731 files cataloged across 8 drivers in the UK, France, and Nigeria (~2.08 GB uncompressed).
- **Selected Primary Sequence**: `S1` (Driver A, Coventry UK) — 51,746 synchronized records (86.2 min, 37.16 km, 9 roundabouts).
- **Coordinate Conversion**: WGS84 Geodetic to local East-North-Up (ENU) tangent plane metric coordinates.
- **Sensor Quality**: Verified 10.00 Hz sampling rate, 0 duplicate timestamps, and identified 43 stationary bias calibration windows.
- **Raw Inertial Baseline**: Quantified exponential drift across 10s, 30s, 60s, and 120s synthetic GNSS outages (33%–60% drift), empirically proving the necessity of AI/ML velocity estimation for Phase 2+.

### Phase 2: Sensor Preprocessing, Calibration & Frame Alignment
Phase 2 establishes the physical sensor calibration and coordinate alignment layer:
- **Static Sensor Bias Estimation**: Estimated gyroscope bias ($b_g$) and accelerometer bias ($b_a$) using verified stationary intervals.
- **Attitude Estimation**: Implemented complementary and quaternion filter tracking roll, pitch, and yaw from IMU data.
- **Phone-to-Vehicle Alignment**: Computes rotation matrix $\mathbf{R}_{v}^{p}$ aligning the smartphone's coordinate frame to the vehicle body frame (forward, lateral, vertical).
- **Gravity Compensation**: Removes the 1g gravity vector from accelerometer measurements in the navigation frame.

### Phase 3: Signal Processing, Vibration Filtering & Causal ML Windows
Phase 3 builds the causal signal processing pipeline:
- **Frequency Analysis**: Identified high-frequency engine and chassis vibration peaks (15–35 Hz) and road noise.
- **Causal Filtering**: Designed low-pass Butterworth and causal filter chains (cutoff at 3.5 Hz) preventing forward-backward filter leakage.
- **Zero-Velocity Update (ZUPT)**: Implemented energy-based and generalized likelihood ratio test (GLRT) standstill detector.
- **12-Channel Processed IMU Representation**: Exported time-synchronized, causal filtered feature representations (`s1_filtered_causal_imu.csv`).
- **ML-Ready Windowing**: Segmented 10,344 preconditioned temporal windows (`ml_ready_windows_s1.npz`) preserving causal boundaries.

---

### Quick Start & Project Execution (Phases 1–3)

#### A. Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install base dependencies
pip install -r requirements.txt
```

#### B. Phase 1 Pipeline
```bash
# Runs inventory, schema discovery, data quality, and raw baseline
python -m Data_details.src.pipeline_runner
```

#### C. Phase 2 Pipeline
```bash
# Runs sensor calibration, frame alignment, attitude estimation, and gravity compensation
python -m Data_details.src.phase2_pipeline
```

#### D. Phase 3 Pipeline
```bash
# Runs causal filtering, vibration analysis, ZUPT detector, and exports 12-channel processed IMU windows
python -m Data_details.src.phase3_pipeline
```

#### E. Automated Unit Testing
```bash
# Execute all unit tests across Phases 1, 2, and 3
pytest Data_details/tests/ -v
```

---

### Current Project Status
- **Phase 1 (Dataset Understanding & Baseline)**: ✅ COMPLETE
- **Phase 2 (Sensor Calibration & Alignment)**: ✅ COMPLETE
- **Phase 3 (Signal Processing & Causal Features)**: ✅ COMPLETE
- **Phase 4 (AI/ML Motion Estimation)**: ❌ DISCARDED / RESET
- **Phase 4.1 (Heading & Benchmark Audit)**: ❌ DISCARDED / RESET
- **Phase 5 (Fusion & Map Matching)**: ❌ NOT STARTED
