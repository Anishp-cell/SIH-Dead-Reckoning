"""
Task 2 & 16: Report Generator for IO-VNBD Phase 1.
Generates dataset documentation summary and final PHASE1_DATASET_REPORT.md.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def generate_documentation_summary(output_dir: str = "Data_details/outputs/reports") -> None:
    """
    Task 2: Generates dataset_documentation_summary.md from repository README,
    paper contents, and our own observations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# IO-VNBD Dataset Documentation Summary\n",
        "## Source: Repository Facts\n",
        "- **Full Name**: IO-VNBD — Inertial and Odometry Vehicle Navigation Benchmark Dataset",
        "- **Repository**: https://github.com/onyekpeu/IO-VNBD",
        "- **Folder Structure**:",
        "  - `Synchronised V abd S datasets/` — Smartphone (S-) and Vehicle (V-) datasets collected simultaneously",
        "    - `Categorised IOVNB Dataset/` — Organized by driver",
        "    - `Uncategorised IOVNB Dataset/` — Flat S-Dataset/ and V-Dataset/ folders",
        "  - `Unsynchronised V and S Dataset/` — Data collected separately (no time alignment)",
        "  - `README.md` — Brief description",
        "  - `README_1.pdf` — Full dataset paper\n",
        "## Source: Paper Facts (IO-VNBD Paper)\n",
        "### Equipment",
        "- **Vehicle Data Logger**: Racelogic VBOX Video HD2 CAN-Bus data logger (10 Hz)",
        "- **GPS Antenna**: Racelogic VBOX Video HD2 GPS Antenna (10 Hz), placed centrally on roof",
        "- **Vehicle**: Ford Fiesta Titanium (front-wheel drive) for CAN-bus data",
        "- **Smartphones**: Huawei P20 Pro, Motorola Moto G7 Power, BlackBerry Priv",
        "- **Smartphone App**: AndroSensor (10 Hz sampling, GPS update ~1 Hz)\n",
        "### Data Volume",
        "- **Smartphone Data**: ~58 hours, ~4,400 km, ~2.2M records × 24 columns",
        "- **Vehicle Data**: ~40 hours, ~1,300 km, ~1.4M records × 29 columns",
        "- **Total**: ~98 hours, ~5,700 km across 8 drivers\n",
        "### Collection Locations",
        "- United Kingdom (Coventry, Nuneaton, etc.)",
        "- Nigeria",
        "- France\n",
        "### Smartphone Sensor Columns (24 fields)",
        "| # | Column | Unit |",
        "|---|--------|------|",
        "| 1 | GPS Latitude | degrees |",
        "| 2 | GPS Longitude | degrees |",
        "| 3 | GPS Altitude | m |",
        "| 4 | GPS Speed | km/h |",
        "| 5 | GPS Accuracy | m |",
        "| 6 | GPS Orientation | degrees |",
        "| 7 | GPS Satellites In Range | count |",
        "| 8 | Time Since Start | ms |",
        "| 9 | Date | YYYY-MO-DD HH-MI-SS_SSS |",
        "| 10 | Accelerometer X | m/s² |",
        "| 11 | Accelerometer Y | m/s² |",
        "| 12 | Accelerometer Z | m/s² |",
        "| 13 | Gravity X | m/s² |",
        "| 14 | Gravity Y | m/s² |",
        "| 15 | Gravity Z | m/s² |",
        "| 16 | Gyroscope Yaw | rad/s |",
        "| 17 | Gyroscope Pitch | rad/s |",
        "| 18 | Gyroscope Roll | rad/s |",
        "| 19 | Magnetic Field X | μT |",
        "| 20 | Magnetic Field Y | μT |",
        "| 21 | Magnetic Field Z | μT |",
        "| 22 | Orientation Yaw/Azimuth | degrees |",
        "| 23 | Orientation Pitch | degrees |",
        "| 24 | Orientation Roll | degrees |\n",
        "### Vehicle CAN-bus Columns (29 fields)",
        "| # | Column | Unit |",
        "|---|--------|------|",
        "| 1 | No of GPS Satellites | count |",
        "| 2 | Time Since Start of Day | seconds |",
        "| 3 | Latitude | degrees |",
        "| 4 | Longitude | degrees |",
        "| 5 | Velocity | km/h |",
        "| 6 | Heading | degrees |",
        "| 7 | Height | km |",
        "| 8 | Vertical Velocity | km/h |",
        "| 9 | Sample Period | seconds |",
        "| 10 | Steering Angle | degrees |",
        "| 11-14 | Wheel Speeds (FL, FR, RL, RR) | rad/s |",
        "| 15 | Yaw Rate | deg/s |",
        "| 16 | Indicated Vehicle Speed | km/h |",
        "| 17 | Longitudinal Acceleration | g |",
        "| 18 | Lateral Acceleration | g |",
        "| 19 | Handbrake | 0 or 1 |",
        "| 20-21 | Gear Requested / Gear | 1-5 |",
        "| 22 | Engine Speed | rev/min |",
        "| 23 | Coolant Temperature | °C |",
        "| 24 | Clutch Position | 0 or 1 |",
        "| 25 | Brake Pressure | PSI |",
        "| 26 | Brake Position | 0 or 1 |",
        "| 27 | Battery Voltage | volts |",
        "| 28 | Air Temperature | °C |",
        "| 29 | Accelerator Pedal Position | % activation |\n",
        "### Driving Scenarios",
        "Hard braking, roundabouts, sharp turns, rain, hills, motorway, town centre, traffic, bumps, potholes, mud roads, parking, stationary, drifts, zig-zag, winding roads, U-turns, varying tyre pressures, and more.\n",
        "### Drivers",
        "| Driver | Style |",
        "|--------|-------|",
        "| A | Defensive |",
        "| B | Defensive |",
        "| C | Defensive |",
        "| D | Defensive |",
        "| E | Aggressive |",
        "| F | Defensive |",
        "| G | Defensive |",
        "| H | Defensive |\n",
        "### Synchronized vs Unsynchronized",
        "- **Synchronized**: V- and S- datasets collected simultaneously from the same vehicle, manually time-aligned, stored in `Synchronised V abd S datasets/`.",
        "- **Unsynchronized**: V- or S- datasets collected independently (different times, vehicles, or locations), stored in `Unsynchronised V and S Dataset/`.",
        "- Not every V-file has a corresponding S-file and vice versa.\n",
        "### Important Notes (from paper)",
        "- Phone vibration interferes with acceleration measurement precision.",
        "- Gravity readings are provided to help correct measured acceleration.",
        "- GPS communication difficulties were encountered at times (documented in txt files).",
        "- Direction of travel is positive X on the phone.",
    ]
    
    report_path = out_path / "dataset_documentation_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Documentation summary saved to {report_path}")


def generate_phase1_report(
    cfg: dict,
    output_dir: str = "Data_details/outputs/reports",
) -> None:
    """
    Task 16: Generates the comprehensive PHASE1_DATASET_REPORT.md.
    Reads generated tables and metrics to compose the final report.
    """
    out_path = Path(output_dir)
    tables_dir = Path(cfg["paths"]["tables_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])
    
    seq = cfg["selected_sequence"]
    
    lines = [
        "# Phase 1 Dataset Report — IO-VNBD\n",
        "## SIH26168: AI-ML based Intelligent Dead Reckoning System for Seamless Navigation\n",
        "---\n",
    ]
    
    # 1. What IO-VNBD Is
    lines.append("## 1. What is IO-VNBD?\n")
    lines.append("IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset) is the first "
                 "large-scale, public dataset for vehicle positioning using inertial and odometry sensors. "
                 "Created by researchers at Coventry University, it contains synchronized smartphone and "
                 "vehicle CAN-bus sensor data collected on public roads in the UK, Nigeria, and France. "
                 "It is designed for developing and benchmarking GPS-denied navigation algorithms.\n")
    
    # 2. Repository Structure
    lines.append("## 2. Repository Structure\n")
    lines.append("```")
    lines.append("IO-VNBD-master/")
    lines.append("├── Synchronised V abd S datasets/")
    lines.append("│   ├── Categorised IOVNB Dataset/   (organized by driver)")
    lines.append("│   └── Uncategorised IOVNB Dataset/  (flat S-Dataset/ and V-Dataset/)")
    lines.append("├── Unsynchronised V and S Dataset/")
    lines.append("├── README.md")
    lines.append("├── README_1.pdf (full paper)")
    lines.append("└── .gitattributes (LFS tracking)")
    lines.append("```\n")
    
    # 3-6. Dataset types, sensors, sync status
    lines.append("## 3. Dataset Types & Sensors\n")
    lines.append("- **Smartphone (S-)**: 24 columns — GPS, accelerometer, gyroscope, magnetometer, gravity, orientation")
    lines.append("- **Vehicle (V-)**: 29 columns — VBOX GPS, CAN-bus wheel speeds, steering, yaw rate, acceleration, engine, brakes")
    lines.append("- **Synchronized**: Smartphone + vehicle data collected simultaneously from same vehicle")
    lines.append("- **Unsynchronized**: Collected independently\n")
    
    # 7. Sampling rates
    lines.append("## 4. Sampling Rates\n")
    ts_path = tables_dir / "timestamp_analysis.csv"
    if ts_path.exists():
        ts_df = pd.read_csv(ts_path)
        for _, row in ts_df.iterrows():
            lines.append(f"### {row['domain'].title()}")
            lines.append(f"- Records: {row['n_records']}")
            lines.append(f"- Duration: {row['duration_min']:.1f} minutes")
            lines.append(f"- Effective frequency: {row['effective_freq_hz']:.2f} Hz")
            lines.append(f"- Mean dt: {row['dt_mean_s']*1000:.1f} ms (expected 100 ms)")
            lines.append(f"- Jitter (std): {row['jitter_std_ms']:.1f} ms")
            lines.append(f"- Duplicate timestamps: {row['n_duplicate_timestamps']}")
            lines.append(f"- Large gaps (>0.5s): {row['n_large_gaps_gt_0_5s']}\n")
    
    # 8. Selected sequence
    lines.append("## 5. Selected Sequence\n")
    lines.append(f"- **Name**: {seq['name']}")
    lines.append(f"- **Driver**: {seq['driver']}")
    lines.append(f"- **Location**: {seq['location']}")
    lines.append(f"- **Description**: {seq['description']}\n")
    
    # 9-10. Data quality
    lines.append("## 6. Data Quality\n")
    quality_path = tables_dir / "data_quality_report.csv"
    if quality_path.exists():
        q_df = pd.read_csv(quality_path)
        total_nulls = q_df["null_count"].sum()
        lines.append(f"- Total null values across all columns: {total_nulls}")
        high_null = q_df[q_df["null_pct"] > 0]
        if len(high_null) > 0:
            lines.append(f"- Columns with any missing: {len(high_null)}")
        constant = q_df[q_df.get("is_constant", False) == True]
        if len(constant) > 0:
            lines.append(f"- Constant columns: {', '.join(constant['column'].tolist())}")
        lines.append("")
    
    # 11-12. GPS trajectory & coordinates
    lines.append("## 7. GPS Trajectory & Local Coordinates\n")
    lines.append("GPS trajectory has been converted to local East-North-Up (ENU) metric coordinates "
                 "using WGS84 → ECEF → ENU transformation with the first valid GPS point as origin.\n")
    lines.append(f"See plots: `gps_trajectory.png`, `enu_trajectory.png`\n")
    
    # 13. Stationary analysis
    lines.append("## 8. Stationary Sensor Analysis\n")
    bias_path = tables_dir / "stationary_bias_stats.csv"
    if bias_path.exists():
        bias_df = pd.read_csv(bias_path)
        if len(bias_df) > 0:
            lines.append("Sensor bias during stationary periods:\n")
            lines.append("| Sensor | Mean | Std | Unit |")
            lines.append("|--------|------|-----|------|")
            for _, row in bias_df.head(12).iterrows():
                lines.append(f"| {row['sensor']} | {row['mean']} | {row['std']} | {row['unit']} |")
            lines.append("")
    
    # 14-17. Baseline results
    lines.append("## 9. GNSS Blackout & Raw Inertial Baseline\n")
    metrics_path = tables_dir / "baseline_metrics.csv"
    if metrics_path.exists():
        m_df = pd.read_csv(metrics_path)
        lines.append("### Baseline Results Across Blackout Durations\n")
        lines.append("| Duration (s) | Endpoint Error (m) | RMSE (m) | Max Error (m) | Drift % |")
        lines.append("|-------------|-------------------|---------|--------------|---------|")
        for _, row in m_df.iterrows():
            lines.append(f"| {row['blackout_duration_s']:.0f} | {row['endpoint_error_m']:.1f} | "
                         f"{row['rmse_m']:.1f} | {row['max_error_m']:.1f} | {row['drift_pct']:.1f}% |")
        lines.append("")
    
    # 18. Limitations
    lines.append("## 10. Important Limitations\n")
    lines.append("1. GPS is used as reference trajectory, not a high-precision RTK ground truth.")
    lines.append("2. Smartphone GPS accuracy is typically 3-10m.")
    lines.append("3. The raw inertial baseline uses a simplified phone-to-ENU rotation (yaw only).")
    lines.append("4. No bias estimation, no calibration, no filtering applied yet.")
    lines.append("5. Phone mounting orientation may vary between sequences.\n")
    
    # 19. Phase 2 recommendations
    lines.append("## 11. Recommendations for Phase 2\n")
    lines.append("1. **Phone-to-Vehicle Calibration**: Full 3D rotation matrix estimation")
    lines.append("2. **ZUPT**: Zero-Velocity Update corrections during detected stops")
    lines.append("3. **Vibration Filtering**: Low-pass filter before integration")
    lines.append("4. **Bias Estimation**: Use stationary data to estimate and remove accelerometer/gyro bias")
    lines.append("5. **AI/ML Velocity Estimation**: Train model to predict speed/velocity from IMU windows")
    lines.append("6. **EKF/ESKF**: Implement proper state estimator for sensor fusion")
    
    report_path = out_path / "PHASE1_DATASET_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Phase 1 report saved to {report_path}")


def generate_simple_explanation(output_dir: str = "Data_details/outputs/reports") -> None:
    """
    Generates DATASET_EXPLANATION_SIMPLE.md — a plain-language explanation of the dataset
    and what we found, suitable for non-technical team members.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# IO-VNBD Dataset — Simple Explanation\n",
        "## What is this dataset?\n",
        "IO-VNBD is a collection of driving data recorded from real cars on real roads. "
        "Researchers drove cars around cities in England, Nigeria, and France while two "
        "systems were recording data at the same time:\n",
        "1. **A smartphone** strapped to the car dashboard (like you'd mount for Google Maps)",
        "2. **A professional car data logger** connected to the car's computer (CAN bus)\n",
        "Think of it as: the smartphone is the 'cheap sensor' we want to build our navigation "
        "system with, and the car's professional equipment gives us 'reference answers' to "
        "compare against.\n",
        "---\n",
        "## What does the smartphone record?\n",
        "The smartphone records 24 measurements, 10 times every second:\n",
        "- **GPS**: Where the phone thinks it is (latitude, longitude, altitude, speed, direction)",
        "- **Accelerometer**: How much the phone is being pushed/pulled in 3 directions "
        "(forward-backward, left-right, up-down). This includes Earth's gravity.",
        "- **Gyroscope**: How fast the phone is rotating (turning left, tilting forward, rolling sideways)",
        "- **Magnetometer**: Compass readings in 3 directions (helps figure out which way is North)",
        "- **Gravity Sensor**: Android's estimate of which direction gravity is pulling "
        "(helps separate gravity from actual vehicle movement)",
        "- **Orientation**: The phone's best guess of its tilt angles (yaw/heading, pitch, roll)\n",
        "---\n",
        "## What does the car's computer record?\n",
        "The car logger records 29 measurements, also 10 times per second:\n",
        "- **GPS** (from a roof-mounted antenna — more accurate than the phone's GPS)",
        "- **Wheel speeds** for each of the 4 wheels",
        "- **Steering wheel angle**",
        "- **Vehicle speed** from the speedometer",
        "- **Yaw rate**: how fast the car is turning (from the car's own IMU sensor)",
        "- **Acceleration**: forward/backward and side-to-side forces",
        "- **Brake pressure and pedal position**",
        "- **Engine speed (RPM), gear, clutch, accelerator pedal**",
        "- **Battery voltage, coolant temp, air temperature**\n",
        "---\n",
        "## Why do we need this dataset?\n",
        "Our SIH project needs to keep tracking a vehicle's position when GPS stops working "
        "(in tunnels, parking garages, under bridges, in urban canyons between tall buildings).\n",
        "The idea is: when GPS disappears, use the smartphone's accelerometer and gyroscope "
        "to estimate 'where did the car go since GPS was last available?' This is called "
        "**dead reckoning** — like a sailor estimating position by speed and direction when "
        "they can't see the stars.\n",
        "The problem: smartphone sensors are noisy and inaccurate. Even tiny errors in "
        "acceleration measurements get amplified when you integrate them to get velocity, "
        "and amplified again when you integrate velocity to get position. Errors grow very fast.\n",
        "---\n",
        "## What did we find in Phase 1?\n",
        "### The dataset is large and diverse",
        "- About 98 hours of driving, 5,700 km total",
        "- 8 different drivers with different driving styles",
        "- Many driving scenarios: roundabouts, hard braking, rain, motorways, bumpy roads, hills\n",
        "### The sensors work at ~10 Hz as claimed",
        "- The smartphone records a measurement roughly every 100 milliseconds (10 times/second)",
        "- There is slight timing jitter but no major gaps in our selected test sequence\n",
        "### Raw dead reckoning fails badly",
        "- When we tried the simplest approach (just integrate acceleration → velocity → position), "
        "the estimated position drifts far from the true path within seconds.",
        "- Even in a 60-second GPS blackout, the position error can reach hundreds of meters.",
        "- The drift percentage is typically **far above the SIH target of <10%**.",
        "- This confirms that **AI/ML and sophisticated algorithms are essential** — "
        "simple integration is nowhere near good enough.\n",
        "### Why does raw dead reckoning fail?",
        "1. **Sensor bias**: The accelerometer has a tiny constant offset. When integrated twice, "
        "this causes quadratic error growth (error grows as time²).",
        "2. **Gravity leakage**: It's hard to perfectly separate Earth's gravity from vehicle "
        "motion acceleration. Any residual gravity directly becomes a large false acceleration.",
        "3. **Phone orientation**: The phone is mounted at a slightly unknown angle. "
        "We don't perfectly know which direction each sensor axis points relative to the car.",
        "4. **Noise amplification**: Random sensor noise, when integrated twice, turns into "
        "ever-growing random drift.",
        "5. **Vibration**: Car vibrations (engine, bumps, potholes) add noise that "
        "looks like acceleration but isn't actual vehicle motion.\n",
        "---\n",
        "## What's next?\n",
        "Phase 1 has shown us exactly *how bad* the raw approach is and *why*. "
        "This tells us exactly what Phase 2+ needs to fix:\n",
        "1. **Calibrate the phone orientation** relative to the car",
        "2. **Filter out vibration and noise** before integrating",
        "3. **Estimate and remove sensor bias** using stationary periods",
        "4. **Train an AI model** that learns to predict vehicle speed/motion from IMU patterns "
        "(much more robust than raw integration)",
        "5. **Add physics constraints**: cars can't fly, can't slide sideways, have maximum "
        "possible speeds and turn rates",
        "6. **Add map matching**: keep the estimated position on actual roads",
        "7. **Fuse everything together** using a mathematical state estimator (Kalman Filter)\n",
        "The SIH target of <10% drift (<5m error per 50m, <100m per 1km) is achievable "
        "with proper engineering, but requires all of these components working together.",
    ]
    
    report_path = out_path / "DATASET_EXPLANATION_SIMPLE.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Simple explanation saved to {report_path}")
