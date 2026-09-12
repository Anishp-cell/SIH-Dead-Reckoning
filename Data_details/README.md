# SIH 2026 ISRO: AI-ML Based Intelligent Dead Reckoning System
## Problem Statement SIH26168 — Phase 1: IO-VNBD Dataset Understanding, Exploration & Raw Baseline

### Overview
This package contains the complete Phase 1 implementation for the Smart India Hackathon 2026 project sponsored by the Indian Space Research Organisation (ISRO). Phase 1 analyzes and benchmarks the **IO-VNBD** (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning), establishes a ground-truth and local coordinate transformation pipeline, and evaluates a classical raw inertial dead reckoning baseline to quantify exponential drift during GNSS outages.

---

### Package Structure
```
Data_details/
├── config/
│   └── config.yaml                     # Unified pipeline configuration
├── data/
│   ├── raw/                            # Cached raw CSV files
│   └── processed/                      # Preprocessed & standardized data
├── src/
│   ├── __init__.py
│   ├── dataset_inventory.py            # Task 1: Scans repository, LFS info, metadata
│   ├── dataset_loader.py               # Dataset loading & column standardization
│   ├── schema_inspector.py             # Task 3 & 4: Schema discovery & Data Dictionary
│   ├── timestamp_analysis.py           # Task 6: dt distribution, sampling rate & jitter
│   ├── data_quality.py                 # Task 7: Missing data, impossible values, distributions
│   ├── sensor_analysis.py              # Task 8: Time-series multi-axis sensor plotting
│   ├── gps_analysis.py                 # Task 9: Geodesic distances, GPS speed & trajectory
│   ├── coordinate_transform.py         # Task 10: WGS84 Geodetic to local metric ENU frame
│   ├── stationary_analysis.py          # Task 11: Stationary window detection & sensor bias
│   ├── blackout_simulator.py           # Task 12: Configurable synthetic GNSS blackout
│   ├── inertial_baseline.py            # Task 13: Classical raw inertial dead reckoning
│   ├── metrics.py                      # Task 14: Trajectory error, RMSE, endpoint & drift %
│   ├── failure_analysis.py             # Task 15: Error correlation with dynamic maneuvers
│   ├── report_generator.py             # Task 2 & 16: Automated markdown report generation
│   ├── pipeline_runner.py              # Orchestrates end-to-end execution of Phase 1
│   └── create_notebooks.py             # Generates the 6 interactive exploration notebooks
├── notebooks/
│   ├── 01_repository_exploration.ipynb
│   ├── 02_sequence_exploration.ipynb
│   ├── 03_sensor_quality.ipynb
│   ├── 04_gps_trajectory.ipynb
│   ├── 05_coordinate_conversion.ipynb
│   └── 06_gnss_blackout_baseline.ipynb
├── tests/
│   ├── test_inventory_and_loader.py
│   └── test_core_modules.py
└── outputs/
    ├── inventory/                      # Repository scan and sequence selection YAML
    ├── tables/                         # Schemas, Data Dictionary, and baseline metrics CSVs
    ├── plots/                          # Time-series, trajectory, and error plots
    ├── reports/                        # Research and plain-language summary reports
    └── logs/                           # Pipeline run logs
```

---

### How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run All Unit Tests**:
   ```bash
   pytest Data_details/tests/ -v
   ```

3. **Run the Full Phase 1 Pipeline**:
   ```bash
   python -m Data_details.src.pipeline_runner
   ```

4. **Regenerate Interactive Notebooks**:
   ```bash
   python Data_details/src/create_notebooks.py
   ```

---

### Key Empirical Findings
- **Sampling Frequency**: Verified at exactly **10.00 Hz** (mean $\Delta t = 100.0\text{ ms}$, jitter standard deviation $= 2.4\text{ ms}$).
- **Representative Sequence `S1`**: 51,746 time-synchronized rows (~86.2 min, 37.16 km distance, 9 roundabouts, Coventry UK).
- **Stationary Periods**: 43 distinct stationary windows detected across sequence `S1` (speed $< 0.5\text{ m/s}$, duration $\ge 5\text{ s}$), suitable for sensor bias calibration and Zero-Velocity Updates (ZUPT).
- **Classical Dead Reckoning Baseline**:
  - 10s GNSS outage: **61.8 m** endpoint error (**49.2% drift**)
  - 30s GNSS outage: **142.4 m** endpoint error (**47.4% drift**)
  - 60s GNSS outage: **284.9 m** endpoint error (**60.2% drift**)
  - 120s GNSS outage: **405.9 m** endpoint error (**33.0% drift**)
- **SIH Target**: $< 10\%$ positional drift ($< 5\text{ m}$ over $50\text{ m}$, $< 100\text{ m}$ over $1\text{ km}$).
- **Conclusion**: Naive double integration fails rapidly due to sensor bias, gravity leakage, and vibration noise. AI/ML velocity estimation, attitude calibration, and state fusion (EKF/ESKF) are essential for Phase 2+.

---

## Phase 2 & Phase 3 Execution Guide

### Phase 2: Sensor Preprocessing & Calibration Layer
```bash
python -m Data_details.src.phase2_pipeline
```
Outputs saved in `Data_details/outputs/phase2/` (`calibration_ablation_study.csv`, `phase2_comparison.csv`).

### Phase 3: Signal Processing, Vibration Filtering & Causal ML Windows
```bash
python -m Data_details.src.phase3_pipeline
```
Outputs saved in `Data_details/outputs/phase3/` (`phase3_dead_reckoning_benchmarks.csv`, `ml_ready_windows_s1.npz`).


---

### Project Status & Roadmap
- **Phase 1 (Dataset Understanding & Exploration)**: ✅ COMPLETE
- **Phase 2 (Sensor Calibration & Alignment)**: ✅ COMPLETE
- **Phase 3 (Signal Processing & Causal Features)**: ✅ COMPLETE
- **Phase 4 (AI/ML Motion Estimation)**: ❌ DISCARDED / RESET
- **Phase 4.1 (Heading & Benchmark Audit)**: ❌ DISCARDED / RESET
- **Phase 5 (Multi-Sensor Fusion & Map Matching)**: ❌ NOT STARTED


