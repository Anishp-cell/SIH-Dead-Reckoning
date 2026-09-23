# Pre-Phase-8 Audit: Phase 4 Training vs. Runtime Feature Parity

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/03_PHASE4_RUNTIME_AUDIT.md`  

---

## 1. Executive Summary & Root-Cause Finding

A primary finding of this audit is that **the runtime evaluation benchmark (`blackout_benchmarks.py`) was not feeding the neural network the feature representation it was trained on**.

- **Training Feature Pipeline**: In `Data_details/src/phase4/dataset.py`, Model F was trained on the raw calibrated vehicle-frame kinematic channels (`acc_fwd_veh`, `acc_lat_veh`, `acc_up_veh`, `gyro_roll_veh`, `gyro_pitch_veh`, `gyro_yaw_veh`). These signals preserve the broadband mechanical vibrations of the vehicle chassis and tires that the 1D-CNN relies on to infer road speed.
- **Runtime Benchmark Pipeline**: In `Data_details/src/phase7/evaluation/blackout_benchmarks.py`, an ad-hoc override selected low-pass filtered signals (`acc_fwd_veh_filtered`, `acc_lat_veh_filtered`, etc.).
- **Impact**: The low-pass filter stripped high-frequency vibrations. On the vertical axis (`acc_up_veh`), correlation between the training feature and the runtime input collapsed to **$r = 0.2044$**, and variance was attenuated by **$41\%$**.

---

## 2. Channel-by-Channel Parity Analysis (Sequence S1 Full Dataset)

| Channel Index | Training Feature Name | Evaluation Feature Used | Training Mean | Evaluation Mean | Training Std | Evaluation Std | Correlation ($r$) | Parity Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `acc_fwd_veh` | `acc_fwd_veh_filtered` | 0.0740 | 0.0755 | 1.0607 | 0.8768 | 0.7266 | **MISMATCH (Filtered)** |
| 1 | `acc_lat_veh` | `acc_lat_veh_filtered` | 0.0133 | 0.0122 | 1.0659 | 0.8846 | 0.7363 | **MISMATCH (Filtered)** |
| 2 | `acc_up_veh` | `acc_up_veh_filtered` | -9.8474 | -9.8483 | 0.5132 | 0.3012 | **0.2044** | **CRITICAL MISMATCH** |
| 3 | `gyro_roll_veh` | `gyro_roll_veh_filtered` | -0.0039 | -0.0039 | 0.1373 | 0.1308 | 0.9078 | Near Parity |
| 4 | `gyro_pitch_veh` | `gyro_pitch_veh_filtered` | 0.0031 | 0.0031 | 0.1057 | 0.0770 | **0.4056** | **SEVERE MISMATCH** |
| 5 | `gyro_yaw_veh` | `gyro_yaw_veh_filtered` | -0.0014 | -0.0014 | 0.0515 | 0.0382 | **0.4864** | **SEVERE MISMATCH** |
| 6 | `jerk_fwd` | `jerk_fwd` | -0.0001 | -0.0001 | 10.0811 | 10.0811 | 1.0000 | EXACT PARITY |
| 7 | `acc_horiz_norm` | `acc_horiz_norm` | 1.2418 | 1.2418 | 0.8515 | 0.8515 | 1.0000 | EXACT PARITY |
| 8 | `gyro_norm` | `gyro_norm` | 0.1334 | 0.1334 | 0.1221 | 0.1221 | 1.0000 | EXACT PARITY |
| 9 | `ori_pitch_deg` | `ori_pitch_deg` | -81.1130 | -81.1130 | 2.7234 | 2.7234 | 1.0000 | EXACT PARITY |
| 10 | `vibration_energy` | `vibration_energy` | 388.9400 | 388.9400 | 40.4357 | 40.4357 | 1.0000 | EXACT PARITY |
| 11 | `is_stationary` | `is_stationary` | 0.1171 | 0.1171 | 0.3216 | 0.3216 | 1.0000 | EXACT PARITY |

---

## 3. Physical & Mechanistic Impact on Neural Speed Inference

1. **Vibration Power Attenuation**:
   Deep neural speed estimation models for smartphones (e.g. Model F / IO-Net architectures) rely heavily on vertical acceleration and pitch/yaw micro-oscillations to disambiguate engine RPM and tire rolling frequencies from steady-state tilt. 
   When `acc_up_veh_filtered` was provided, the model received an artificial flatline signal that suppressed predicted velocity towards zero during cruising.
2. **Normalization Scaling Distortion**:
   Normalization parameters in `normalization.json` were calculated on the unfiltered signals:
   - For `acc_up_veh`: $\mu = -9.8390\text{ m/s}^2, \sigma = 0.5317\text{ m/s}^2$.
   - Normalizing the filtered signal (where standard deviation is only $0.3012$) compressed the effective $z$-score input distribution by $43\%$, shifting the activations into saturated regions of the ReLU/ELU activation functions.

---

## 4. Corrective Action & Verification

### Fix Implemented:
`Data_details/src/phase7/evaluation/blackout_benchmarks.py` and `Data_details/src/phase6/evaluation/blackout_benchmarks.py` have been restored to use the exact `FEATURE_COLUMNS` specification from `dataset.py`:
```python
feature_cols = [
    "acc_fwd_veh", "acc_lat_veh", "acc_up_veh",
    "gyro_roll_veh", "gyro_pitch_veh", "gyro_yaw_veh",
    "jerk_fwd", "acc_horiz_norm", "gyro_norm",
    "ori_pitch_deg", "vibration_energy", "is_stationary",
]
```

### Verification:
An automated regression test (`test_training_runtime_feature_parity`) has been integrated into `Data_details/tests/correction_pass/` to assert exact identity between training feature names and runtime streaming inputs.
