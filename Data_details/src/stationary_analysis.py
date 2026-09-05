"""
Task 11: Stationary Period Analysis for IO-VNBD sequences.
Detects periods where the vehicle appears stationary, analyzes accelerometer and gyro bias.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def detect_stationary_periods(
    df: pd.DataFrame,
    speed_col: str = "gps_speed_mps",
    speed_threshold: float = 0.5,
    min_duration_s: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Detects stationary periods where speed is below threshold for minimum duration.
    
    Returns:
        List of dicts with start_idx, end_idx, start_time_s, end_time_s, duration_s
    """
    if speed_col not in df.columns:
        # Try GPS speed in km/h
        if "gps_speed_kmh" in df.columns:
            speed = df["gps_speed_kmh"].values / 3.6
        elif "v_speed_kmh" in df.columns:
            speed = df["v_speed_kmh"].values / 3.6
        else:
            logger.warning("No speed column found for stationary detection")
            return []
    else:
        speed = df[speed_col].values
    
    time_s = df["time_s"].values
    is_slow = speed < speed_threshold
    
    # Find contiguous stationary regions
    periods = []
    in_period = False
    start_idx = 0
    
    for i in range(len(is_slow)):
        if is_slow[i] and not in_period:
            start_idx = i
            in_period = True
        elif not is_slow[i] and in_period:
            duration = time_s[i-1] - time_s[start_idx]
            if duration >= min_duration_s:
                periods.append({
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "start_time_s": float(time_s[start_idx]),
                    "end_time_s": float(time_s[i-1]),
                    "duration_s": round(duration, 1),
                })
            in_period = False
    
    # Handle if ends stationary
    if in_period:
        duration = time_s[-1] - time_s[start_idx]
        if duration >= min_duration_s:
            periods.append({
                "start_idx": start_idx,
                "end_idx": len(is_slow) - 1,
                "start_time_s": float(time_s[start_idx]),
                "end_time_s": float(time_s[-1]),
                "duration_s": round(duration, 1),
            })
    
    logger.info(f"Found {len(periods)} stationary periods (>= {min_duration_s}s, speed < {speed_threshold} m/s)")
    return periods


def analyze_stationary_bias(
    df: pd.DataFrame,
    periods: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Analyzes sensor bias during stationary periods.
    Returns a DataFrame with bias statistics per period.
    """
    records = []
    
    sensor_cols = {
        "acc_x": "m/s²", "acc_y": "m/s²", "acc_z": "m/s²",
        "grav_x": "m/s²", "grav_y": "m/s²", "grav_z": "m/s²",
        "gyro_x": "rad/s", "gyro_y": "rad/s", "gyro_z": "rad/s",
        "mag_x": "μT", "mag_y": "μT", "mag_z": "μT",
    }
    
    for p_idx, period in enumerate(periods):
        segment = df.iloc[period["start_idx"]:period["end_idx"]+1]
        
        for col, unit in sensor_cols.items():
            if col not in segment.columns:
                continue
            vals = segment[col].dropna().values
            if len(vals) == 0:
                continue
            
            records.append({
                "period_idx": p_idx,
                "start_time_s": period["start_time_s"],
                "duration_s": period["duration_s"],
                "sensor": col,
                "unit": unit,
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
                "n_samples": len(vals),
            })
    
    return pd.DataFrame(records)


def plot_stationary_analysis(
    df: pd.DataFrame,
    periods: List[Dict[str, Any]],
    output_dir: str = "Data_details/outputs/plots",
    sequence_name: str = "S1",
) -> None:
    """Plots sensor signals during stationary periods to visualize bias and noise."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if len(periods) == 0:
        logger.warning("No stationary periods found to plot")
        return
    
    # Use the longest stationary period
    longest = max(periods, key=lambda p: p["duration_s"])
    seg = df.iloc[longest["start_idx"]:longest["end_idx"]+1]
    t = seg["time_s"].values - seg["time_s"].values[0]  # relative time
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Stationary Sensor Analysis — Sequence {sequence_name}\n"
                 f"Period: {longest['duration_s']:.1f}s starting at t={longest['start_time_s']:.0f}s",
                 fontsize=13, fontweight="bold")
    
    # Accelerometer during stationary
    ax = axes[0]
    for col, label, color in [("acc_x", "Acc X", "#e74c3c"), ("acc_y", "Acc Y", "#2ecc71"), ("acc_z", "Acc Z", "#3498db")]:
        if col in seg.columns:
            vals = seg[col].values
            ax.plot(t, vals, linewidth=0.5, alpha=0.8, color=color,
                    label=f"{label} (μ={np.nanmean(vals):.3f}, σ={np.nanstd(vals):.3f})")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Accelerometer (Stationary — shows bias + noise)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Gyroscope during stationary
    ax = axes[1]
    for col, label, color in [("gyro_x", "Gyro X", "#e74c3c"), ("gyro_y", "Gyro Y", "#2ecc71"), ("gyro_z", "Gyro Z", "#3498db")]:
        if col in seg.columns:
            vals = seg[col].values
            ax.plot(t, vals, linewidth=0.5, alpha=0.8, color=color,
                    label=f"{label} (μ={np.nanmean(vals):.5f}, σ={np.nanstd(vals):.5f})")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Gyroscope (Stationary — shows bias + noise)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Gravity vector during stationary
    ax = axes[2]
    for col, label, color in [("grav_x", "Grav X", "#e74c3c"), ("grav_y", "Grav Y", "#2ecc71"), ("grav_z", "Grav Z", "#3498db")]:
        if col in seg.columns:
            vals = seg[col].values
            ax.plot(t, vals, linewidth=0.5, alpha=0.8, color=color,
                    label=f"{label} (μ={np.nanmean(vals):.4f}, σ={np.nanstd(vals):.4f})")
    ax.set_ylabel("Gravity (m/s²)")
    ax.set_xlabel("Time within stationary period (s)")
    ax.set_title("Gravity Vector (Stationary — should be stable)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path / "stationary_sensor_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved stationary_sensor_analysis.png")


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
    
    stat_cfg = cfg.get("stationary_detection", {})
    periods = detect_stationary_periods(
        s_df,
        speed_threshold=stat_cfg.get("speed_threshold_mps", 0.5),
        min_duration_s=stat_cfg.get("min_duration_s", 5.0),
    )
    
    bias_df = analyze_stationary_bias(s_df, periods)
    
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    bias_df.to_csv(tables_dir / "stationary_bias_stats.csv", index=False)
    
    plot_stationary_analysis(s_df, periods, cfg["paths"]["plots_dir"], seq["name"])
    
    print(f"\n{'='*60}")
    print("STATIONARY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Found {len(periods)} stationary periods")
    for i, p in enumerate(periods):
        print(f"  Period {i}: t={p['start_time_s']:.0f}s, duration={p['duration_s']:.1f}s")
    print(f"{'='*60}\n")
