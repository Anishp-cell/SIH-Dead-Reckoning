# Project Reset Report: Baseline Restored to End of Phase 3

**Smart India Hackathon 2026 — Problem Statement SIH26168 (ISRO)**  
**AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation**  
**Timestamp of Reset**: September 12, 2026  
**Status**: Baseline successfully reset to the end of Phase 3. All Phase 4 and Phase 4.1 artifacts have been permanently purged.

---

## 1. Executive Summary
In accordance with instructions, all source code, models, notebooks, test suites, evaluation scripts, and experimental artifacts associated with **Phase 4** and **Phase 4.1** have been completely and irreversibly removed from the repository.

The repository is now in an exact, pristine **End of Phase 3** baseline state:
- **Phase 1**: [COMPLETE & VERIFIED]
- **Phase 2**: [COMPLETE & VERIFIED]
- **Phase 3**: [COMPLETE & VERIFIED]
- **Phase 4**: [REMOVED / DISCARDED]
- **Phase 4.1**: [REMOVED / DISCARDED]
- **Phase 5**: [NOT STARTED]

---

## 2. Inventory of Removed / Purged Artifacts

### A. Phase 4 Source Code (`Data_details/src/` & Workspace Root)
- `phase4_gpu_check.py` (Workspace root)
- `Data_details/src/phase4_prepare.py`
- `Data_details/src/ml_speed_dataset.py`
- `Data_details/src/ml_speed_model.py`
- `Data_details/src/ml_speed_trainer.py`
- `Data_details/src/ml_speed_evaluator.py`
- `Data_details/src/phase4_pipeline.py`
- `Data_details/src/create_phase4_notebooks.py`

### B. Phase 4.1 Source Code (`Data_details/src/`)
- `Data_details/src/phase4_1_benchmark.py`
- `Data_details/src/phase4_1_heading.py`
- `Data_details/src/phase4_1_leakage_audit.py`
- `Data_details/src/phase4_1_split.py`

### C. Phase 4 & Phase 4.1 Test Suites
- `Data_details/tests/phase4/` directory (including `test_ml_models.py` and test caches)

### D. Phase 4 Interactive Exploration Notebooks (`Data_details/notebooks/`)
- `17_speed_dataset_exploration.ipynb`
- `18_causal_tcn_training.ipynb`
- `19_ai_velocity_dead_reckoning.ipynb`

### E. Phase 4 Output Directories & Artifacts (`Data_details/outputs/phase4/`)
- **Entire directory deleted**: `Data_details/outputs/phase4/`
  - All model runs & checkpoints (`tcn_speed_run_001/`, `gru_speed_run_001/`, `cnn1d_speed_run_001/`, `multitask_multitask_run_001/`, `uncertainty_run_001/`, `best_model.pt`, `final_model.pt`)
  - All training logs and metrics (`training_log.csv`, `metrics.json`)
  - All processed tensor datasets (`phase4_dataset_s1.npz`)
  - All normalization configurations (`feature_normalization.json`)
  - All split manifests (`phase4_split_manifest.json`)
  - All tables (`phase4_model_comparison.csv`, `phase4_dead_reckoning_benchmarks.csv`, `phase4_ablation.csv`)
  - All diagnostic and trajectory plots (`blackout_trajectory_*.png`, `loss_curve.png`, `prediction_plot.png`, `error_distribution.png`)
  - All research notes & reports (`PHASE4_MATHEMATICAL_FOUNDATIONS.md`, `PHASE4_ML_MOTION_ESTIMATION_REPORT.md`, `PHASE4_EXPLANATION_SIMPLE.md`, `PHASE4_RESEARCH_NOTES.md`)

### F. Phase 4.1 Output Directories & Artifacts (`Data_details/outputs/phase4_1/`)
- **Entire directory deleted**: `Data_details/outputs/phase4_1/`
  - All checkpoints (`phase4_1_checkpoints/`)
  - All tables & benchmarks (`phase4_1_benchmarks.csv`, `phase4_1_ablation.csv`, `speed_metrics.csv`, `heading_metrics.csv`)
  - All manifests (`phase4_1_split_manifest.json`, `phase4_1_reproduction_manifest.json`)
  - All audit reports (`PHASE4_1_LEAKAGE_AUDIT.md`, `PHASE4_1_GENERALIZATION_AUDIT.md`, `PHASE4_1_EXPERIMENT_MATRIX.md`, `PHASE4_1_EVALUATION_REPORT.md`)
  - All trajectory plots (`blackout_comparison_*.png`)

### G. Documentation & Pipeline Configurations
- Removed all Phase 4 execution commands, scripts, and checkpoint directory links from root `README.md`.
- Removed all Phase 4 execution commands and guides from `Data_details/README.md`.

---

## 3. Inventory of Preserved Artifacts

All files belonging to Phases 1, 2, and 3 have been preserved with 100% integrity without modification or deletion:

### A. Raw Data & Datasets
- Original IO-VNBD dataset files and directories intact.
- Cached raw and processed data in `Data_details/data/` preserved.

### B. Phase 1 Assets
- **Source Modules**: `dataset_inventory.py`, `dataset_loader.py`, `schema_inspector.py`, `gps_analysis.py`, `sensor_analysis.py`, `timestamp_analysis.py`, `stationary_analysis.py`, `data_quality.py`, `inertial_baseline.py`, `blackout_simulator.py`, `metrics.py`, `pipeline_runner.py`, `report_generator.py`, `create_notebooks.py`.
- **Notebooks**: `01_repository_exploration.ipynb` through `06_gnss_blackout_baseline.ipynb`.
- **Outputs**: `inventory/`, `tables/`, `plots/`, `reports/`.
- **Tests**: `test_core_modules.py`, `test_inventory_and_loader.py`.

### C. Phase 2 Assets
- **Source Modules**: `sensor_calibration.py`, `phone_vehicle_alignment.py`, `attitude_estimation.py`, `gravity_compensation.py`, `calibration_metrics.py`, `coordinate_transform.py`, `failure_analysis.py`, `phase2_pipeline.py`, `phase2_report.py`, `create_phase2_notebooks.py`.
- **Notebooks**: `07_attitude_analysis.ipynb` through `11_phase2_dead_reckoning_comparison.ipynb`.
- **Outputs**: `Data_details/outputs/phase2/` (calibration JSONs: `accel_bias.json`, `gyro_bias.json`, `phone_vehicle_rotation.json`; tables, plots, `PHASE2_CALIBRATION_REPORT.md`, `PHASE2_EXPLANATION_SIMPLE.md`).
- **Tests**: `Data_details/tests/phase2/` (`test_alignment_and_gravity.py`, `test_attitude.py`, `test_calibration.py`).

### D. Phase 3 Assets (CRITICAL - Unaltered)
- **Source Modules**: `spectral_analysis.py`, `filter_design.py`, `vibration_analysis.py`, `adaptive_filtering.py`, `wavelet_denoising.py`, `zupt_detector.py`, `signal_quality.py`, `signal_analysis.py`, `phase3_metrics.py`, `phase3_pipeline.py`, `phase3_report.py`, `create_phase3_notebooks.py`.
- **Notebooks**: `12_signal_frequency_analysis.ipynb` through `16_phase3_dead_reckoning_benchmark.ipynb`.
- **Outputs**: `Data_details/outputs/phase3/`:
  - **12-Channel Processed IMU Dataset**: `Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv` (35.3 MB, 51,746 rows).
  - **ML-Ready Temporal Windows**: `Data_details/outputs/phase3/processed/ml_ready_windows_s1.npz` (10,344 windows of $(30, 8)$ shape).
  - **Window Summary**: `Data_details/outputs/phase3/processed/ml_ready_windows_summary.json`.
  - **Reports & Research**: `PHASE3_SIGNAL_PROCESSING_REPORT.md`, `PHASE3_EXPLANATION_SIMPLE.md`, `MATHEMATICAL_FOUNDATIONS_PHASE1_TO_PHASE3.md`, `PHASE3_RESEARCH_NOTES.md`, `future_ml_targets.md`, `phase1_phase2_math_audit.md`.
  - **Tables & Benchmarks**: `phase3_dead_reckoning_benchmarks.csv`, `filter_characterization_table.csv`, `filter_quality_benchmarks.csv`, etc.
  - **Tests**: `Data_details/tests/phase3/` (all 7 test suites).

---

## 4. Verification Results

### A. Repository-Wide Discarded Keyword Audit
A repository-wide search was conducted for the designated keywords:
- `phase4`: **0 references in code or outputs**
- `phase4_1`: **0 references**
- `tcn`: **0 references in active code** (only forward-looking recommendation note in Phase 3 report preserved)
- `gru`: **0 references in active code** (only forward-looking notes in Phase 2/3 reports preserved)
- `lstm`: **0 references in active code** (only literature review citation in Phase 3 notes preserved)
- `ml_speed`: **0 references**
- `dead_reckoning AI`: **0 references**
- `speed model`: **0 references**
- `heading model`: **0 references**

### B. Automated Unit Test Execution
All unit tests across Phases 1, 2, and 3 were executed in the project virtual environment:

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\python\SIH 26 ISRO
collected 64 items

Data_details/tests/phase2/test_alignment_and_gravity.py ....             [  6%]
Data_details/tests/phase2/test_attitude.py ..........                   [ 21%]
Data_details/tests/phase2/test_calibration.py ...                       [ 26%]
Data_details/tests/phase3/test_causal_filter_state.py ..                 [ 29%]
Data_details/tests/phase3/test_filter_design.py ....                    [ 35%]
Data_details/tests/phase3/test_math_audit_rotations.py ......           [ 45%]
Data_details/tests/phase3/test_robust_filtering.py ...                  [ 50%]
Data_details/tests/phase3/test_spectral_analysis.py ...                 [ 54%]
Data_details/tests/phase3/test_upgrades.py .......                      [ 65%]
Data_details/tests/phase3/test_wavelet_denoising.py ..                  [ 68%]
Data_details/tests/test_core_modules.py ...............                 [ 92%]
Data_details/tests/test_inventory_and_loader.py .....                   [100%]

============================= 64 passed in 7.60s ==============================
```
**Result**: **64 / 64 tests passed (100% PASS)**. Zero failures, zero regressions.

### C. Phase 3 Pipeline Independence Check
Phase 3 pipeline modules were checked for independent loading and execution:
```bash
python -c "import Data_details.src.phase3_pipeline; print('Phase 3 pipeline module loaded cleanly!')"
# Output: Phase 3 pipeline module loaded cleanly!
```
The Phase 3 pipeline runs strictly against Phase 1 and 2 dependencies and requires no Phase 4 components.

---

## 5. Current State Confirmation

```
================================================================================
PROJECT STATUS MATRIX
================================================================================
Phase 1 (IO-VNBD Dataset Understanding & Baseline)    :  [COMPLETE]      [OK]
Phase 2 (Sensor Preprocessing & Calibration)           :  [COMPLETE]      [OK]
Phase 3 (Signal Processing, Vibration & Causal Windows):  [COMPLETE]      [OK]
Phase 4 (AI/ML Motion Estimation)                      :  [REMOVED]       [OK]
Phase 4.1 (Heading Estimation & Benchmark Audit)       :  [REMOVED]       [OK]
Phase 5 (Multi-Sensor Fusion & Map Matching)           :  [NOT STARTED]   [OK]
================================================================================
```

The reset is complete. No further actions or designs beyond the reset have been initiated.
