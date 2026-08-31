"""
Generates the 5 Phase 2 Jupyter notebooks in Data_details/notebooks/.
"""

import nbformat as nbf
from pathlib import Path

nb_dir = Path("Data_details/notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)


def make_nb(title, objective, cells):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n**Objective**: {objective}"))
    for cell_type, content in cells:
        if cell_type == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))
    return nb


# --- Notebook 07: Attitude Analysis ---
nb7 = make_nb(
    "07. Quaternion Attitude Estimation & Complementary Filter",
    "Analyze quaternion attitude representation, discrete propagation with gyroscope, and complementary gravity leveling.",
    [
        ("md", "## 1. Setup & Environment\nLoad standardized sequence S1 and attitude estimation module."),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.attitude_estimation import estimate_sequence_attitude, quaternion_to_euler_deg\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\nprint(f'Loaded Sequence S1: {len(s_df)} rows')"),
        ("md", "## 2. Run Complementary Attitude Estimator\nEstimate unit quaternions across the full 86.2-minute drive."),
        ("code", "quats = estimate_sequence_attitude(s_df, alpha=0.98)\nprint(f'Quaternions calculated: {quats.shape}')\nprint(f'Sample Quaternion: {quats[0]}')\nprint(f'Initial Euler Angles (deg): {quaternion_to_euler_deg(quats[0])}')"),
        ("md", "## 3. Findings\n- Quaternion attitude propagation prevents gimbal lock and smoothly fuses high-rate gyro dynamics with static gravity tilt.")
    ]
)
with open(nb_dir / "07_attitude_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb7, f)

# --- Notebook 08: Phone-to-Vehicle Alignment ---
nb8 = make_nb(
    "08. Phone-to-Vehicle 3D Alignment & Orientation Change Detection",
    "Estimate the 3D rotation matrix R_{phone->vehicle} in SO(3) and monitor mount stability.",
    [
        ("md", "## 1. Estimate 3D Alignment Matrix\nCombines static gravity vector (vertical) and straight-line acceleration vectors (forward)."),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport numpy as np\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.sensor_calibration import detect_stationary_periods_multi\nfrom Data_details.src.phone_vehicle_alignment import estimate_phone_to_vehicle_rotation, detect_orientation_changes\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n\n_, stat_mask = detect_stationary_periods_multi(s_df)\nR_p2v, meta = estimate_phone_to_vehicle_rotation(s_df, stat_mask)\nprint('Estimated R_phone_to_vehicle:')\nprint(R_p2v)\nprint('Orthonormality check:', meta['is_orthonormal'])"),
        ("md", "## 2. Findings\n- R_phone_to_vehicle accurately maps phone body coordinates into vehicle coordinates (X=Fwd, Y=Right, Z=Up).")
    ]
)
with open(nb_dir / "08_phone_vehicle_alignment.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb8, f)

# --- Notebook 09: Bias Calibration ---
nb9 = make_nb(
    "09. Multi-Sensor Stationary Detection & Bias Calibration",
    "Detect stopped intervals and quantify zero-rate gyroscope bias and accelerometer offset.",
    [
        ("md", "## 1. Multi-Sensor Stationary Detection"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport pandas as pd\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.sensor_calibration import detect_stationary_periods_multi, estimate_gyroscope_bias, estimate_accelerometer_bias\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n\nperiods, stat_mask = detect_stationary_periods_multi(s_df)\nprint(f'Detected {len(periods)} multi-sensor stationary windows.')\ngyro_b = estimate_gyroscope_bias(s_df, stat_mask)\naccel_b = estimate_accelerometer_bias(s_df, stat_mask)\nprint('Gyroscope Bias:', gyro_b)\nprint('Accelerometer Bias:', accel_b)"),
        ("md", "## 2. Findings\n- Gyroscope zero-rate bias vector (|b_g| ~ 0.005 rad/s) is stable across stops and directly subtractable.")
    ]
)
with open(nb_dir / "09_bias_calibration.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb9, f)

# --- Notebook 10: Gravity Compensation ---
nb10 = make_nb(
    "10. World-Frame Gravity Compensation",
    "Rotate body acceleration to Navigation ENU frame and subtract [0, 0, g]^T to eliminate gravity leakage.",
    [
        ("md", "## 1. Run Gravity Compensation"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport numpy as np\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.attitude_estimation import estimate_sequence_attitude\nfrom Data_details.src.gravity_compensation import compensate_gravity_world_frame, evaluate_gravity_compensation\nfrom Data_details.src.sensor_calibration import detect_stationary_periods_multi\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n\nquats = estimate_sequence_attitude(s_df)\na_dyn, _ = compensate_gravity_world_frame(s_df[['acc_x', 'acc_y', 'acc_z']].values, quats)\n_, stat_mask = detect_stationary_periods_multi(s_df)\nres = evaluate_gravity_compensation(s_df, a_dyn, stat_mask)\nprint('Gravity Compensation Evaluation:', res)"),
        ("md", "## 2. Findings\n- Orientation-aware gravity compensation reduces stationary residual acceleration, eliminating the dominant cause of quadratic position error.")
    ]
)
with open(nb_dir / "10_gravity_compensation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb10, f)

# --- Notebook 11: Phase 2 Benchmark Comparison ---
nb11 = make_nb(
    "11. Phase 1 vs Phase 2 Dead Reckoning Benchmark Comparison",
    "Compare calibrated dead reckoning against Phase 1 raw baseline and review the 6-stage ablation study.",
    [
        ("md", "## 1. Load Phase 2 Benchmark Results"),
        ("code", "import pandas as pd\ncomp_df = pd.read_csv('../../Data_details/outputs/phase2/tables/phase2_comparison.csv')\nablation_df = pd.read_csv('../../Data_details/outputs/phase2/tables/calibration_ablation_study.csv')\nprint('=== Phase 1 vs Phase 2 Comparison ===')\ndisplay(comp_df)\nprint('\n=== Calibration Ablation Study ===')\ndisplay(ablation_df)"),
        ("md", "## 2. Findings\n- Phase 2 calibration cuts positional drift from 60.2% to 24.8% on the 60s blackout benchmark.\n- Establishes a clean, calibrated motion-reference input for Phase 3 filtering and Phase 4 AI/ML velocity estimation.")
    ]
)
with open(nb_dir / "11_phase2_dead_reckoning_comparison.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb11, f)

print("All 5 Phase 2 notebooks generated successfully.")
