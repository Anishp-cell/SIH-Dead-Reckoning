"""
Task 9: GPS Trajectory Analysis for IO-VNBD sequences.
Plots GPS trajectories, calculates distances, speed statistics, and ENU conversion.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.coordinate_transform import (
    geodetic_to_enu, cumulative_distance_m, enu_cumulative_distance,
)

logger = logging.getLogger(__name__)


def analyze_gps_trajectory(
    df: pd.DataFrame,
    lat_col: str = "gps_lat",
    lon_col: str = "gps_lon",
    alt_col: str = "gps_alt",
    speed_col: str = "gps_speed_kmh",
) -> Dict[str, Any]:
    """Analyzes a GPS trajectory: distance, duration, speed stats, valid point count."""
    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)
    
    valid_mask = ~(np.isnan(lat) | np.isnan(lon)) & (np.abs(lat) > 0.1) & (np.abs(lon) > 0.001)
    n_valid = int(valid_mask.sum())
    n_total = len(lat)
    
    # Cumulative distance
    cum_dist = cumulative_distance_m(lat, lon)
    total_dist_m = cum_dist[-1]
    
    # Duration
    time_s = df["time_s"].values
    duration_s = time_s[-1] - time_s[0]
    
    # Speed stats
    speed_stats = {}
    if speed_col in df.columns:
        speed = df[speed_col].dropna()
        speed_stats = {
            "speed_mean_kmh": round(float(speed.mean()), 2),
            "speed_max_kmh": round(float(speed.max()), 2),
            "speed_std_kmh": round(float(speed.std()), 2),
        }
    
    stats = {
        "n_total_points": n_total,
        "n_valid_gps_points": n_valid,
        "valid_gps_pct": round(n_valid / n_total * 100, 2),
        "duration_s": round(duration_s, 1),
        "duration_min": round(duration_s / 60.0, 1),
        "total_distance_m": round(total_dist_m, 1),
        "total_distance_km": round(total_dist_m / 1000.0, 2),
        "lat_min": round(float(np.nanmin(lat[valid_mask])), 6) if n_valid > 0 else None,
        "lat_max": round(float(np.nanmax(lat[valid_mask])), 6) if n_valid > 0 else None,
        "lon_min": round(float(np.nanmin(lon[valid_mask])), 6) if n_valid > 0 else None,
        "lon_max": round(float(np.nanmax(lon[valid_mask])), 6) if n_valid > 0 else None,
    }
    stats.update(speed_stats)
    
    return stats


def plot_gps_trajectory(
    df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/plots",
    sequence_name: str = "S1",
    lat_col: str = "gps_lat",
    lon_col: str = "gps_lon",
) -> None:
    """Plots latitude vs longitude GPS trajectory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)
    valid = ~(np.isnan(lat) | np.isnan(lon))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by time
    time_s = df["time_s"].values
    scatter = ax.scatter(
        lon[valid], lat[valid],
        c=time_s[valid] / 60.0,
        cmap="viridis", s=1, alpha=0.7,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Time (minutes)")
    
    ax.plot(lon[valid][0], lat[valid][0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(lon[valid][-1], lat[valid][-1], "ro", markersize=10, label="End", zorder=5)
    
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(f"GPS Trajectory — Sequence {sequence_name}")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "gps_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved gps_trajectory.png")


def plot_enu_trajectory(
    df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/plots",
    sequence_name: str = "S1",
    lat_col: str = "gps_lat",
    lon_col: str = "gps_lon",
    alt_col: str = "gps_alt",
) -> dict:
    """Converts GPS to ENU and plots the metric trajectory. Returns origin info."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)
    alt = df[alt_col].values.astype(float) if alt_col in df.columns else np.zeros_like(lat)
    
    east, north, up, origin_info = geodetic_to_enu(lat, lon, alt)
    
    valid = ~(np.isnan(east) | np.isnan(north))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    time_s = df["time_s"].values
    scatter = ax.scatter(
        east[valid], north[valid],
        c=time_s[valid] / 60.0,
        cmap="viridis", s=1, alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax, label="Time (minutes)")
    
    ax.plot(east[valid][0], north[valid][0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(east[valid][-1], north[valid][-1], "ro", markersize=10, label="End", zorder=5)
    
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Local ENU Trajectory — Sequence {sequence_name}\n"
                 f"Origin: ({origin_info['lat0_deg']:.6f}°, {origin_info['lon0_deg']:.6f}°)")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "enu_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved enu_trajectory.png")
    
    return origin_info


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
    
    # Smartphone GPS
    s_stats = analyze_gps_trajectory(s_df)
    plot_gps_trajectory(s_df, cfg["paths"]["plots_dir"], seq["name"])
    origin_info = plot_enu_trajectory(s_df, cfg["paths"]["plots_dir"], seq["name"])
    
    print(f"\n{'='*60}")
    print("GPS TRAJECTORY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Valid GPS points: {s_stats['n_valid_gps_points']}/{s_stats['n_total_points']} ({s_stats['valid_gps_pct']}%)")
    print(f"Duration: {s_stats['duration_min']} min")
    print(f"Total distance: {s_stats['total_distance_km']} km")
    if "speed_mean_kmh" in s_stats:
        print(f"Avg speed: {s_stats['speed_mean_kmh']} km/h, Max: {s_stats['speed_max_kmh']} km/h")
    print(f"ENU origin: ({origin_info['lat0_deg']:.6f}, {origin_info['lon0_deg']:.6f})")
    print(f"{'='*60}\n")
