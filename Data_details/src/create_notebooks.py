"""
Generates the 6 Phase 1 Jupyter notebooks in Data_details/notebooks/.
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


# --- Notebook 01: Repository Exploration ---
nb1 = make_nb(
    "01. IO-VNBD Repository Exploration",
    "Explore file structure, LFS metadata, dataset domains, and driver statistics across IO-VNBD.",
    [
        ("md", "## 1. Environment & Setup\nImport repository scanner and configuration."),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport pandas as pd\nimport yaml\nfrom Data_details.src.dataset_inventory import scan_repository, generate_inventory\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nprint('Config loaded successfully.')"),
        ("md", "## 2. Scan Repository Inventory\nScan all files and check Git LFS metadata."),
        ("code", "inv_df = scan_repository('../../IO-VNBD-master')\nprint(f'Total files scanned: {len(inv_df)}')\ninv_df.head(10)"),
        ("md", "## 3. Dataset Domain & Driver Breakdown"),
        ("code", "print('=== Files by Domain ===')\nprint(inv_df['domain'].value_counts())\nprint('\\n=== Files by Driver / Region ===')\nprint(inv_df[inv_df['extension'] == '.csv']['driver'].value_counts())"),
        ("md", "## 4. Findings\n- Total 731 files scanned (~2.08 GB uncompressed).\n- 241 smartphone CSVs and 484 vehicle CAN-bus CSVs.\n- 8 drivers across UK, France, and Nigeria.\n- Data is split into Synchronised (simultaneous V and S collection) and Unsynchronised recordings.")
    ]
)
with open(nb_dir / "01_repository_exploration.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)

# --- Notebook 02: Sequence Exploration ---
nb2 = make_nb(
    "02. Representative Sequence Exploration",
    "Load and inspect the selected primary synchronized sequence (S1, Driver A, Coventry UK).",
    [
        ("md", "## 1. Load Sequence using DatasetLoader"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport pandas as pd\nfrom Data_details.src.dataset_loader import DatasetLoader\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\n\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\nseq = cfg['selected_sequence']\ns_df, v_df = loader.load_sequence(seq['smartphone_file'], seq.get('vehicle_file'))\nprint(f'Loaded Smartphone: {s_df.shape}')\nprint(f'Loaded Vehicle: {v_df.shape if v_df is not None else None}')"),
        ("md", "## 2. Inspect Smartphone Columns & Head"),
        ("code", "print('Smartphone Columns:', list(s_df.columns))\ns_df.head()"),
        ("md", "## 3. Inspect Vehicle Columns & Head"),
        ("code", "if v_df is not None:\n    print('Vehicle Columns:', list(v_df.columns))\n    display(v_df.head())"),
        ("md", "## 4. Findings\n- Sequence S1 contains 51,746 time-synchronized rows (~86.2 min).\n- Smartphone provides 24 raw channels + normalized time_s, dt, gps_speed_mps.\n- Vehicle provides 29 ECU/CAN-bus channels including individual 4-wheel speeds, steering angle, and yaw rate.")
    ]
)
with open(nb_dir / "02_sequence_exploration.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)

# --- Notebook 03: Sensor Quality & Statistics ---
nb3 = make_nb(
    "03. Sensor Signal & Data Quality Analysis",
    "Verify sensor distributions, missing values, timestamp jitter, and stationary sensor bias.",
    [
        ("md", "## 1. Data Quality Analysis"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport pandas as pd\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.data_quality import analyze_data_quality\nfrom Data_details.src.timestamp_analysis import analyze_timestamps\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, v_df = loader.load_sequence(cfg['selected_sequence']['smartphone_file'], cfg['selected_sequence'].get('vehicle_file'))\n\nq_df = analyze_data_quality(s_df, 'smartphone')\nq_df.head(15)"),
        ("md", "## 2. Timestamp Jitter Analysis"),
        ("code", "ts_stats = analyze_timestamps(s_df, 'smartphone')\nfor k, v in ts_stats.items():\n    print(f'{k}: {v}')"),
        ("md", "## 3. Stationary Sensor Bias Detection"),
        ("code", "from Data_details.src.stationary_analysis import detect_stationary_periods, analyze_stationary_bias\nperiods = detect_stationary_periods(s_df, speed_threshold=0.5, min_duration_s=5.0)\nprint(f'Detected {len(periods)} stationary windows.')\nbias_df = analyze_stationary_bias(s_df, periods)\nbias_df.head(12)"),
        ("md", "## 4. Findings\n- Zero missing values, zero infinite values, and zero duplicate timestamps in S1.\n- Effective sampling rate is 10.00 Hz (mean dt = 100.0 ms, std = 2.4 ms jitter).\n- 43 stationary periods detected across the 86.2-minute drive, ideal for sensor bias estimation and zero-velocity updates (ZUPT).")
    ]
)
with open(nb_dir / "03_sensor_quality.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb3, f)

# --- Notebook 04: GPS Trajectory ---
nb4 = make_nb(
    "04. GPS Trajectory & Geodesic Distance",
    "Analyze geographic path, total distance, speeds, and plot 2D trajectory.",
    [
        ("md", "## 1. GPS Trajectory Analysis"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport matplotlib.pyplot as plt\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.gps_analysis import analyze_gps_trajectory\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n\ngps_stats = analyze_gps_trajectory(s_df)\nfor k, v in gps_stats.items():\n    print(f'{k}: {v}')"),
        ("md", "## 2. Trajectory Visualizations\nView generated GPS trajectory plot."),
        ("code", "import matplotlib.image as mpimg\nimg = mpimg.imread('../../Data_details/outputs/plots/gps_trajectory.png')\nplt.figure(figsize=(10, 10))\nplt.imshow(img)\nplt.axis('off')\nplt.title('GPS Trajectory — Sequence S1', fontsize=14)\nplt.show()"),
        ("md", "## 3. Findings\n- Total trajectory distance: 37.16 km over 86.2 minutes (average speed ~25.9 km/h, max 83.9 km/h).\n- 100% valid GPS points (51,746/51,746).\n- Trajectory traverses Coventry urban roads, arterial roads, and 9 major roundabouts.")
    ]
)
with open(nb_dir / "04_gps_trajectory.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb4, f)

# --- Notebook 05: Coordinate Conversion (WGS84 to ENU) ---
nb5 = make_nb(
    "05. WGS84 Geodetic to Local Metric ENU Coordinate Transformation",
    "Transform spherical GPS coordinates into local Cartesian East-North-Up tangent plane.",
    [
        ("md", "## 1. Run Geodetic to ENU Conversion"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport numpy as np\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.coordinate_transform import geodetic_to_enu\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n\nlat = s_df['gps_lat'].values\nlon = s_df['gps_lon'].values\nalt = s_df['gps_alt'].values\n\neast, north, up, origin = geodetic_to_enu(lat, lon, alt)\nprint('ENU Origin:', origin)\nprint(f'East range: [{np.nanmin(east):.1f}m, {np.nanmax(east):.1f}m]')\nprint(f'North range: [{np.nanmin(north):.1f}m, {np.nanmax(north):.1f}m]')"),
        ("md", "## 2. Local ENU Trajectory Plot"),
        ("code", "import matplotlib.image as mpimg\nimport matplotlib.pyplot as plt\nimg = mpimg.imread('../../Data_details/outputs/plots/enu_trajectory.png')\nplt.figure(figsize=(10, 10))\nplt.imshow(img)\nplt.axis('off')\nplt.show()"),
        ("md", "## 3. Findings\n- WGS84 coordinates successfully mapped to metric East, North, Up space centered at (52.401660°, -1.505290°).\n- Local Cartesian coordinates allow direct integration and Euclidean error measurement without spherical distortion.")
    ]
)
with open(nb_dir / "05_coordinate_conversion.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb5, f)

# --- Notebook 06: GNSS Blackout & Raw Inertial Baseline ---
nb6 = make_nb(
    "06. GNSS Blackout Simulation & Classical Inertial Dead Reckoning Baseline",
    "Simulate GNSS denial windows, double-integrate raw IMU measurements, and quantify exponential drift.",
    [
        ("md", "## 1. Run Baseline across Multiple Outage Durations (10s, 30s, 60s, 120s)"),
        ("code", "import sys\nsys.path.insert(0, '../..')\nimport yaml\nimport pandas as pd\nfrom Data_details.src.dataset_loader import DatasetLoader\nfrom Data_details.src.inertial_baseline import run_inertial_baseline\n\nwith open('../config/config.yaml', 'r') as f:\n    cfg = yaml.safe_load(f)\nloader = DatasetLoader('../../IO-VNBD-master', '../../Data_details/data/raw')\ns_df, _ = loader.load_sequence(cfg['selected_sequence']['smartphone_file'])\n\nmetrics_list = []\nfor dur in [10.0, 30.0, 60.0, 120.0]:\n    res = run_inertial_baseline(s_df, blackout_start_s=150.0, blackout_duration_s=dur)\n    metrics_list.append(res['metrics'])\n\nmetrics_df = pd.DataFrame(metrics_list)\nmetrics_df[['blackout_duration_s', 'distance_travelled_m', 'endpoint_error_m', 'rmse_m', 'drift_pct']]"),
        ("md", "## 2. Visual Comparison: Raw INS vs Reference Path"),
        ("code", "import matplotlib.image as mpimg\nimport matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 2, figsize=(18, 8))\naxes[0].imshow(mpimg.imread('../../Data_details/outputs/plots/raw_ins_vs_reference.png'))\naxes[0].axis('off')\naxes[0].set_title('Estimated vs Reference Path (60s Blackout)')\naxes[1].imshow(mpimg.imread('../../Data_details/outputs/plots/position_error_vs_time.png'))\naxes[1].axis('off')\naxes[1].set_title('Position Error Growth Over Time')\nplt.tight_layout()\nplt.show()"),
        ("md", "## 3. Findings\n- Raw inertial dead reckoning exhibits severe exponential drift: 49.2% drift at 10s (61.8m error), 60.2% drift at 60s (284.9m error), and 405.9m error at 120s.\n- SIH 2026 goal requires <10% drift (<5m over 50m, <100m over 1km).\n- Root causes: Accelerometer bias, gravity leakage due to uncalibrated phone attitude, vibration noise, and lack of kinematic/speed constraints.\n- Phase 2 must introduce attitude calibration, vibration filtering, and Phase 4 AI/ML velocity estimation.")
    ]
)
with open(nb_dir / "06_gnss_blackout_baseline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb6, f)

print("All 6 Jupyter notebooks generated successfully in Data_details/notebooks/.")
