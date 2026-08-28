"""
Task 13 & 14: Classical Raw Inertial Dead Reckoning Baseline for IO-VNBD.
Implements a basic inertial navigation pipeline: gravity removal, acceleration integration
to velocity, velocity integration to position. Uses actual timestamps.

WARNING: This is intentionally a naive baseline to demonstrate the dead reckoning problem.
It is NOT a production-grade INS. No sophisticated alignment, bias correction, or filtering.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.blackout_simulator import simulate_blackout, get_blackout_window_info
from Data_details.src.metrics import compute_trajectory_metrics, compute_error_timeseries

logger = logging.getLogger(__name__)


def gravity_removal_simple(
    acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray,
    grav_x: np.ndarray, grav_y: np.ndarray, grav_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes gravity from accelerometer readings using Android's gravity sensor.
    
    linear_acc = raw_acc - gravity_vector
    
    Assumption: The gravity sensor output from Android sensor fusion is a reasonable
    estimate of the gravity component. This is the simplest defensible approach.
    """
    lin_x = acc_x - grav_x
    lin_y = acc_y - grav_y
    lin_z = acc_z - grav_z
    
    return lin_x, lin_y, lin_z


def phone_to_enu_simple(
    lin_x: np.ndarray, lin_y: np.ndarray, lin_z: np.ndarray,
    ori_yaw_deg: np.ndarray, ori_pitch_deg: np.ndarray, ori_roll_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplistic rotation of phone-frame linear acceleration to local East-North
    using Android orientation (azimuth/yaw).
    
    This is a VERY simplified approach:
    - Uses only the yaw/azimuth angle for 2D horizontal projection
    - Ignores pitch/roll effects on horizontal acceleration decomposition
    - Assumes phone is roughly upright (pitch ~ -90°, roll ~ 0° in typical mount)
    
    For a phone mounted ~vertically on a car dashboard:
    - Phone X-axis ≈ vehicle lateral (left-right)
    - Phone Y-axis ≈ vehicle longitudinal (forward, up the screen)
    - Phone Z-axis ≈ vertical (out of screen)
    
    We project X, Y into East-North using azimuth heading.
    
    DOCUMENTED LIMITATION: This ignores the complex 3D rotation matrix that
    would be needed for rigorous phone-to-navigation frame transformation.
    """
    yaw_rad = np.radians(ori_yaw_deg)
    
    # Phone Y points up the screen -> approximate forward direction
    # Phone X points right -> approximate lateral direction
    # Azimuth = angle from North, clockwise
    
    # Forward (phone Y) projected to East-North
    # East = forward * sin(azimuth) + lateral * cos(azimuth)
    # North = forward * cos(azimuth) - lateral * sin(azimuth)
    
    acc_east = lin_y * np.sin(yaw_rad) + lin_x * np.cos(yaw_rad)
    acc_north = lin_y * np.cos(yaw_rad) - lin_x * np.sin(yaw_rad)
    
    return acc_east, acc_north


def integrate_dead_reckoning(
    acc_east: np.ndarray,
    acc_north: np.ndarray,
    dt: np.ndarray,
    initial_vel_east: float = 0.0,
    initial_vel_north: float = 0.0,
    initial_pos_east: float = 0.0,
    initial_pos_north: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Double-integrates acceleration to get velocity then position using
    trapezoidal integration with actual timestamps.
    
    Returns:
        vel_east, vel_north, pos_east, pos_north (all numpy arrays)
    """
    n = len(acc_east)
    vel_east = np.zeros(n)
    vel_north = np.zeros(n)
    pos_east = np.zeros(n)
    pos_north = np.zeros(n)
    
    vel_east[0] = initial_vel_east
    vel_north[0] = initial_vel_north
    pos_east[0] = initial_pos_east
    pos_north[0] = initial_pos_north
    
    for i in range(1, n):
        d = dt[i]
        
        # Trapezoidal integration: velocity
        vel_east[i] = vel_east[i-1] + 0.5 * (acc_east[i-1] + acc_east[i]) * d
        vel_north[i] = vel_north[i-1] + 0.5 * (acc_north[i-1] + acc_north[i]) * d
        
        # Trapezoidal integration: position
        pos_east[i] = pos_east[i-1] + 0.5 * (vel_east[i-1] + vel_east[i]) * d
        pos_north[i] = pos_north[i-1] + 0.5 * (vel_north[i-1] + vel_north[i]) * d
    
    return vel_east, vel_north, pos_east, pos_north


def run_inertial_baseline(
    df: pd.DataFrame,
    blackout_start_s: float,
    blackout_duration_s: float,
) -> Dict[str, Any]:
    """
    Runs the full raw inertial dead reckoning baseline during a simulated GNSS blackout.
    
    Steps:
    1. Convert full GPS trajectory to ENU for reference
    2. Identify blackout window
    3. At blackout start: initialize DR with GPS-derived position and velocity
    4. During blackout: integrate smartphone IMU only
    5. Compare DR trajectory against GPS reference
    
    Returns dict with metrics, reference/estimated trajectories
    """
    # Step 1: Full ENU reference trajectory
    lat = df["gps_lat"].values.astype(float)
    lon = df["gps_lon"].values.astype(float)
    alt = df["gps_alt"].values.astype(float) if "gps_alt" in df.columns else np.zeros_like(lat)
    
    ref_east, ref_north, ref_up, origin = geodetic_to_enu(lat, lon, alt)
    
    # Step 2: Extract blackout window
    time_s = df["time_s"].values
    blackout_end_s = blackout_start_s + blackout_duration_s
    
    # Find indices for blackout window
    bo_start_idx = np.searchsorted(time_s, blackout_start_s)
    bo_end_idx = np.searchsorted(time_s, blackout_end_s)
    
    if bo_start_idx >= len(time_s) - 10:
        logger.error("Blackout start is too close to end of data")
        return {}
    
    bo_end_idx = min(bo_end_idx, len(time_s))
    
    # Step 3: Initialize dead reckoning at blackout start
    init_east = ref_east[bo_start_idx]
    init_north = ref_north[bo_start_idx]
    
    # Estimate initial velocity from GPS speed and heading
    if "gps_speed_mps" in df.columns and "gps_heading_deg" in df.columns:
        speed = df["gps_speed_mps"].iloc[bo_start_idx]
        heading = df["gps_heading_deg"].iloc[bo_start_idx]
        # GPS heading: 0=North, 90=East, clockwise
        init_vel_east = speed * np.sin(np.radians(heading))
        init_vel_north = speed * np.cos(np.radians(heading))
    else:
        init_vel_east = 0.0
        init_vel_north = 0.0
    
    # Step 4: Extract IMU data during blackout
    bo_slice = slice(bo_start_idx, bo_end_idx)
    
    required_cols = ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
                     "ori_yaw_deg", "ori_pitch_deg", "ori_roll_deg"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return {}
    
    acc_x = df["acc_x"].values[bo_slice]
    acc_y = df["acc_y"].values[bo_slice]
    acc_z = df["acc_z"].values[bo_slice]
    grav_x = df["grav_x"].values[bo_slice]
    grav_y = df["grav_y"].values[bo_slice]
    grav_z = df["grav_z"].values[bo_slice]
    ori_yaw = df["ori_yaw_deg"].values[bo_slice]
    ori_pitch = df["ori_pitch_deg"].values[bo_slice]
    ori_roll = df["ori_roll_deg"].values[bo_slice]
    dt = df["dt"].values[bo_slice]
    bo_time = time_s[bo_slice]
    
    # Gravity removal
    lin_x, lin_y, lin_z = gravity_removal_simple(acc_x, acc_y, acc_z, grav_x, grav_y, grav_z)
    
    # Rotate to ENU
    acc_east, acc_north = phone_to_enu_simple(lin_x, lin_y, lin_z, ori_yaw, ori_pitch, ori_roll)
    
    # Step 5: Integrate
    vel_e, vel_n, pos_e, pos_n = integrate_dead_reckoning(
        acc_east, acc_north, dt,
        init_vel_east, init_vel_north,
        init_east, init_north,
    )
    
    # Step 6: Compare against reference
    ref_e_bo = ref_east[bo_slice]
    ref_n_bo = ref_north[bo_slice]
    
    # Distance travelled during blackout (from reference)
    dist_ref = enu_cumulative_distance(ref_e_bo, ref_n_bo)
    total_dist = dist_ref[-1]
    
    # Metrics
    metrics = compute_trajectory_metrics(
        ref_e_bo, ref_n_bo, pos_e, pos_n, bo_time, total_dist,
    )
    
    error_ts = compute_error_timeseries(ref_e_bo, ref_n_bo, pos_e, pos_n, bo_time)
    
    metrics["blackout_start_s"] = blackout_start_s
    metrics["blackout_duration_s"] = blackout_duration_s
    metrics["initial_speed_mps"] = float(np.sqrt(init_vel_east**2 + init_vel_north**2))
    
    result = {
        "metrics": metrics,
        "error_timeseries": error_ts,
        "est_east": pos_e,
        "est_north": pos_n,
        "ref_east": ref_e_bo,
        "ref_north": ref_n_bo,
        "time_s": bo_time,
        "origin": origin,
        "full_ref_east": ref_east,
        "full_ref_north": ref_north,
        "full_time_s": time_s,
    }
    
    return result


def plot_baseline_comparison(
    result: Dict[str, Any],
    output_dir: str = "Data_details/outputs/plots",
    sequence_name: str = "S1",
) -> None:
    """Plots DR trajectory vs reference trajectory and error over time."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    metrics = result["metrics"]
    
    # === Plot 1: Trajectory comparison ===
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Full reference in light gray
    ax.plot(result["full_ref_east"], result["full_ref_north"],
            color="#cccccc", linewidth=0.5, alpha=0.5, label="Full Reference")
    
    # Reference during blackout
    ax.plot(result["ref_east"], result["ref_north"],
            color="#2ecc71", linewidth=2, label="GPS Reference (blackout window)", zorder=3)
    
    # DR estimate
    ax.plot(result["est_east"], result["est_north"],
            color="#e74c3c", linewidth=2, linestyle="--",
            label="Raw INS Dead Reckoning", zorder=4)
    
    # Start/end markers
    ax.plot(result["ref_east"][0], result["ref_north"][0],
            "go", markersize=12, label="Blackout Start", zorder=5)
    ax.plot(result["ref_east"][-1], result["ref_north"][-1],
            "bs", markersize=10, label="Blackout End (GPS)", zorder=5)
    ax.plot(result["est_east"][-1], result["est_north"][-1],
            "r^", markersize=12, label="DR End (estimated)", zorder=5)
    
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Raw Inertial Dead Reckoning vs GPS Reference — {sequence_name}\n"
                 f"Blackout: {metrics['blackout_duration_s']:.0f}s | "
                 f"Endpoint Error: {metrics['endpoint_error_m']:.1f}m | "
                 f"Drift: {metrics['drift_pct']:.1f}%")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "raw_ins_vs_reference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved raw_ins_vs_reference.png")
    
    # === Plot 2: Error over time ===
    error_ts = result["error_timeseries"]
    t_relative = error_ts["time_s"].values - error_ts["time_s"].values[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Dead Reckoning Position Error — {sequence_name} "
                 f"({metrics['blackout_duration_s']:.0f}s blackout)",
                 fontsize=14, fontweight="bold")
    
    ax = axes[0]
    ax.plot(t_relative, error_ts["error_m"].values, color="#e74c3c", linewidth=1.5)
    ax.axhline(metrics["rmse_m"], color="#3498db", linestyle="--",
               label=f"RMSE = {metrics['rmse_m']:.1f} m")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Absolute Position Error Over Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Velocity magnitude comparison (if available)
    ax = axes[1]
    # Plot East and North error components
    de = error_ts["est_east"].values - error_ts["ref_east"].values
    dn = error_ts["est_north"].values - error_ts["ref_north"].values
    ax.plot(t_relative, de, color="#e74c3c", linewidth=0.8, alpha=0.8, label="East Error")
    ax.plot(t_relative, dn, color="#3498db", linewidth=0.8, alpha=0.8, label="North Error")
    ax.set_ylabel("Error Component (m)")
    ax.set_xlabel("Time since blackout start (s)")
    ax.set_title("East & North Position Error Components")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "position_error_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved position_error_vs_time.png")


def save_baseline_metrics(
    metrics_list: list,
    output_dir: str = "Data_details/outputs/tables",
) -> None:
    """Saves baseline metrics for multiple blackout durations as CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    csv_path = out_path / "baseline_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Baseline metrics saved to {csv_path}")


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
    s_df, _ = loader.load_sequence(seq["smartphone_file"], seq.get("vehicle_file"))
    
    bl_cfg = cfg["blackout_simulation"]
    start_s = bl_cfg["default_start_time_s"]
    durations = bl_cfg["durations_s"]
    
    all_metrics = []
    for dur in durations:
        logger.info(f"\n--- Running baseline for {dur:.0f}s blackout ---")
        result = run_inertial_baseline(s_df, start_s, dur)
        if result:
            all_metrics.append(result["metrics"])
            if dur == bl_cfg["primary_test_duration_s"]:
                plot_baseline_comparison(result, cfg["paths"]["plots_dir"], seq["name"])
    
    save_baseline_metrics(all_metrics, cfg["paths"]["tables_dir"])
    
    print(f"\n{'='*60}")
    print("INERTIAL BASELINE COMPLETE")
    print(f"{'='*60}")
    for m in all_metrics:
        print(f"  {m['blackout_duration_s']:.0f}s blackout: "
              f"endpoint={m['endpoint_error_m']:.1f}m, "
              f"RMSE={m['rmse_m']:.1f}m, "
              f"drift={m['drift_pct']:.1f}%")
    print(f"{'='*60}\n")
