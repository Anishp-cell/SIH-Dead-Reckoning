"""
Task 8: Sensor Signal Visualization for IO-VNBD sequences.
Generates multi-panel time-series plots for accelerometer, gyroscope, magnetometer,
gravity, orientation, and GPS metrics.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_smartphone_sensors(
    df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/plots",
    sequence_name: str = "S1",
) -> None:
    """Generates comprehensive time-series sensor plots for smartphone data."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    time_min = df["time_s"].values / 60.0
    duration_str = f"{df['time_s'].iloc[-1] / 60:.1f} min"
    
    # === Plot 1: IMU Sensors (Accelerometer + Gyroscope) ===
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Smartphone IMU Sensors — Sequence {sequence_name} ({duration_str})",
                 fontsize=14, fontweight="bold")
    
    # Accelerometer
    ax = axes[0]
    for col, label, color in [("acc_x", "Acc X", "#e74c3c"), ("acc_y", "Acc Y", "#2ecc71"), ("acc_z", "Acc Z", "#3498db")]:
        if col in df.columns:
            ax.plot(time_min, df[col], linewidth=0.4, alpha=0.8, color=color, label=label)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Accelerometer (includes gravity)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gyroscope
    ax = axes[1]
    for col, label, color in [("gyro_x", "Gyro X (Yaw)", "#e74c3c"), ("gyro_y", "Gyro Y (Pitch)", "#2ecc71"), ("gyro_z", "Gyro Z (Roll)", "#3498db")]:
        if col in df.columns:
            ax.plot(time_min, df[col], linewidth=0.4, alpha=0.8, color=color, label=label)
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Gyroscope")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Magnetometer
    ax = axes[2]
    for col, label, color in [("mag_x", "Mag X", "#e74c3c"), ("mag_y", "Mag Y", "#2ecc71"), ("mag_z", "Mag Z", "#3498db")]:
        if col in df.columns:
            ax.plot(time_min, df[col], linewidth=0.4, alpha=0.8, color=color, label=label)
    ax.set_ylabel("Magnetic Field (μT)")
    ax.set_title("Magnetometer")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gravity vector
    ax = axes[3]
    for col, label, color in [("grav_x", "Grav X", "#e74c3c"), ("grav_y", "Grav Y", "#2ecc71"), ("grav_z", "Grav Z", "#3498db")]:
        if col in df.columns:
            ax.plot(time_min, df[col], linewidth=0.4, alpha=0.8, color=color, label=label)
    ax.set_ylabel("Gravity (m/s²)")
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Gravity Vector (Android sensor fusion)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "sensor_imu_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensor_imu_timeseries.png")
    
    # === Plot 2: Orientation + GPS ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Smartphone Orientation & GPS — Sequence {sequence_name} ({duration_str})",
                 fontsize=14, fontweight="bold")
    
    # Orientation
    ax = axes[0]
    for col, label, color in [("ori_yaw_deg", "Yaw (Azimuth)", "#e74c3c"), ("ori_pitch_deg", "Pitch", "#2ecc71"), ("ori_roll_deg", "Roll", "#3498db")]:
        if col in df.columns:
            ax.plot(time_min, df[col], linewidth=0.5, alpha=0.8, color=color, label=label)
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Orientation (Android sensor fusion)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # GPS Speed
    ax = axes[1]
    if "gps_speed_kmh" in df.columns:
        ax.plot(time_min, df["gps_speed_kmh"], linewidth=0.6, color="#e67e22", label="GPS Speed")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("GPS Speed")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # GPS Accuracy + Satellites
    ax = axes[2]
    if "gps_acc_m" in df.columns:
        ax.plot(time_min, df["gps_acc_m"], linewidth=0.6, color="#9b59b6", label="GPS Accuracy (m)")
    ax.set_ylabel("Accuracy (m)")
    ax.set_xlabel("Time (minutes)")
    ax.set_title("GPS Accuracy")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "sensor_orientation_gps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensor_orientation_gps.png")


def plot_vehicle_sensors(
    df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/plots",
    sequence_name: str = "S1",
) -> None:
    """Generates time-series sensor plots for vehicle ECU data."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    time_min = df["time_s"].values / 60.0
    duration_str = f"{df['time_s'].iloc[-1] / 60:.1f} min"
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Vehicle ECU Sensors — Sequence {sequence_name} ({duration_str})",
                 fontsize=14, fontweight="bold")
    
    # Speed + Wheel speeds
    ax = axes[0]
    if "v_speed_kmh" in df.columns:
        ax.plot(time_min, df["v_speed_kmh"], linewidth=0.6, color="#e74c3c", label="Vehicle Speed")
    if "v_gps_speed_kmh" in df.columns:
        ax.plot(time_min, df["v_gps_speed_kmh"], linewidth=0.4, color="#3498db", alpha=0.7, label="GPS Speed")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Vehicle & GPS Speed")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Acceleration
    ax = axes[1]
    if "v_long_acc_g" in df.columns:
        ax.plot(time_min, df["v_long_acc_g"], linewidth=0.4, color="#e74c3c", label="Longitudinal (g)")
    if "v_lat_acc_g" in df.columns:
        ax.plot(time_min, df["v_lat_acc_g"], linewidth=0.4, color="#3498db", label="Lateral (g)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Vehicle CAN Accelerations")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Yaw rate + Steering
    ax = axes[2]
    if "v_yaw_rate_degs" in df.columns:
        ax.plot(time_min, df["v_yaw_rate_degs"], linewidth=0.4, color="#2ecc71", label="Yaw Rate (deg/s)")
    ax2 = ax.twinx()
    if "v_steering_deg" in df.columns:
        ax2.plot(time_min, df["v_steering_deg"], linewidth=0.3, color="#e67e22", alpha=0.6, label="Steering (deg)")
        ax2.set_ylabel("Steering Angle (deg)", color="#e67e22")
    ax.set_ylabel("Yaw Rate (deg/s)")
    ax.set_title("Yaw Rate & Steering Angle")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Brake + Engine
    ax = axes[3]
    if "v_brake_psi" in df.columns:
        ax.plot(time_min, df["v_brake_psi"], linewidth=0.4, color="#e74c3c", label="Brake Pressure (PSI)")
    ax2 = ax.twinx()
    if "v_engine_rpm" in df.columns:
        ax2.plot(time_min, df["v_engine_rpm"], linewidth=0.3, color="#3498db", alpha=0.5, label="Engine RPM")
        ax2.set_ylabel("Engine Speed (RPM)", color="#3498db")
    ax.set_ylabel("Brake Pressure (PSI)")
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Brake Pressure & Engine Speed")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "sensor_vehicle_ecu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensor_vehicle_ecu.png")


if __name__ == "__main__":
    import yaml
    import sys
    sys.path.insert(0, ".")
    from Data_details.src.dataset_loader import DatasetLoader
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    with open("Data_details/config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    loader = DatasetLoader(cfg["paths"]["repo_root"], cfg["paths"]["data_cache_dir"])
    seq = cfg["selected_sequence"]
    s_df, v_df = loader.load_sequence(seq["smartphone_file"], seq.get("vehicle_file"))
    
    plot_smartphone_sensors(s_df, cfg["paths"]["plots_dir"], seq["name"])
    if v_df is not None:
        plot_vehicle_sensors(v_df, cfg["paths"]["plots_dir"], seq["name"])
    
    print("Sensor visualization complete!")
