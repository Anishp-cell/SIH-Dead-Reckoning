# Phase 8 Data Provenance & Sensor Verification

**Document**: `docs/phase8/correction_pass/03_PROVENANCE_FINAL.md`  
**Purpose**: Document the provenance of datasets, sensor telemetry, and truth sources.

---

## 1. Dataset Analysis (IO-VNBD S1 Sequence)
- **Primary Telemetry (`S-S1.csv`)**: Smartphone sensor data logged via Android device mounted inside the vehicle. Contains accelerometer (100 Hz), gyroscope (100 Hz), and GPS fix data.
- **Reference Trajectory (`V-S1.csv`)**: OXTS RTK-corrected inertial navigation system (RT3000) providing ground truth position, velocity, and roll/pitch/heading at 100 Hz.

## 2. Provenance Taxonomy
| Label | Description | Permitted Uses |
|-------|-------------|----------------|
| `REAL_GNSS` | Unaltered smartphone GPS telemetry | Standstill calibration, real-world noise analysis |
| `SYNTHETIC_GNSS` | Controlled degradation applied to reference positions (1.8m noise, 1 Hz) | Controlled reproducible benchmarks |
| `REFERENCE_ONLY` | High-precision OXTS RTK truth | Performance evaluation and ground-truth initialized Mode A diagnostic |

## 3. Reference-State Leakage Audit
- Automated assertion `assert_no_reference_leakage()` guarantees that no reference state enters the operational filter state vector during Mode B benchmarks.
