"""
Task 15: Failure Analysis for raw dead reckoning baseline.
Correlates error growth with dynamic driving events: cornering, braking, bumps, etc.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def identify_dynamic_events(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    """
    Identifies dynamic driving events within a window of the smartphone DataFrame.
    Events include high angular velocity (turns), hard braking, and high vibration.
    """
    seg = df.iloc[start_idx:end_idx].copy()
    time_s = seg["time_s"].values
    
    events = []
    
    # Detect high yaw rate (turning)
    if "gyro_x" in seg.columns:
        gyro_mag = np.abs(seg["gyro_x"].values)
        turning = gyro_mag > 0.3  # rad/s threshold
        if np.any(turning):
            events.append({
                "event_type": "High Angular Velocity (Turning)",
                "n_samples": int(turning.sum()),
                "pct_window": round(turning.mean() * 100, 1),
                "peak_value": round(float(gyro_mag.max()), 3),
                "unit": "rad/s",
            })
    
    # Detect hard braking (large negative forward acceleration)
    if "acc_y" in seg.columns and "grav_y" in seg.columns:
        lin_y = seg["acc_y"].values - seg["grav_y"].values
        braking = lin_y < -3.0  # m/s² threshold
        if np.any(braking):
            events.append({
                "event_type": "Hard Braking",
                "n_samples": int(braking.sum()),
                "pct_window": round(braking.mean() * 100, 1),
                "peak_value": round(float(lin_y.min()), 2),
                "unit": "m/s²",
            })
    
    # Detect high vibration (acceleration noise)
    if "acc_z" in seg.columns and "grav_z" in seg.columns:
        lin_z = seg["acc_z"].values - seg["grav_z"].values
        # Rolling std over ~0.5s windows
        window_size = 5
        if len(lin_z) > window_size:
            rolling_std = pd.Series(lin_z).rolling(window_size).std().values
            high_vib = rolling_std > 2.0  # m/s²
            if np.any(high_vib[~np.isnan(high_vib)]):
                events.append({
                    "event_type": "High Vertical Vibration (Bumps/Potholes)",
                    "n_samples": int(np.nansum(high_vib)),
                    "pct_window": round(np.nanmean(high_vib) * 100, 1),
                    "peak_value": round(float(np.nanmax(rolling_std)), 2),
                    "unit": "m/s² std",
                })
    
    # Detect large lateral acceleration (sharp turns)
    if "acc_x" in seg.columns and "grav_x" in seg.columns:
        lin_x = seg["acc_x"].values - seg["grav_x"].values
        sharp_lat = np.abs(lin_x) > 3.0
        if np.any(sharp_lat):
            events.append({
                "event_type": "High Lateral Acceleration (Sharp Turn)",
                "n_samples": int(sharp_lat.sum()),
                "pct_window": round(sharp_lat.mean() * 100, 1),
                "peak_value": round(float(np.abs(lin_x).max()), 2),
                "unit": "m/s²",
            })
    
    # Detect GPS anomalies (large jumps or zero speed inconsistencies)
    if "gps_speed_kmh" in seg.columns:
        speed = seg["gps_speed_kmh"].values
        speed_jumps = np.abs(np.diff(speed))
        big_jumps = speed_jumps > 20  # km/h per sample
        if np.any(big_jumps):
            events.append({
                "event_type": "GPS Speed Anomaly (Large Jump)",
                "n_samples": int(big_jumps.sum()),
                "pct_window": round(big_jumps.mean() * 100, 1),
                "peak_value": round(float(speed_jumps.max()), 1),
                "unit": "km/h per sample",
            })
    
    return pd.DataFrame(events)


def generate_failure_report(
    result: Dict[str, Any],
    df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/reports",
    sequence_name: str = "S1",
) -> None:
    """
    Generates a markdown failure analysis report correlating DR error 
    with detected dynamic events.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    metrics = result["metrics"]
    error_ts = result["error_timeseries"]
    
    # Find time of maximum error
    max_err_idx = error_ts["error_m"].idxmax()
    max_err_time = error_ts.loc[max_err_idx, "time_s"]
    
    # Detect events in blackout window
    time_s = df["time_s"].values
    bo_start = metrics["blackout_start_s"]  
    bo_end = bo_start + metrics["blackout_duration_s"]
    start_idx = np.searchsorted(time_s, bo_start)
    end_idx = np.searchsorted(time_s, bo_end)
    
    events_df = identify_dynamic_events(df, start_idx, end_idx)
    
    lines = [
        f"# Failure Analysis Report — Raw Dead Reckoning Baseline\n",
        f"## Sequence: {sequence_name}\n",
        f"## Blackout Configuration\n",
        f"- **Start**: {bo_start:.1f}s",
        f"- **Duration**: {metrics['blackout_duration_s']:.0f}s",
        f"- **Samples**: {metrics['n_samples']}",
        f"- **Distance Travelled**: {metrics['distance_travelled_m']:.1f}m\n",
        f"## Error Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Endpoint Error | {metrics['endpoint_error_m']:.1f} m |",
        f"| RMSE | {metrics['rmse_m']:.1f} m |",
        f"| Max Error | {metrics['max_error_m']:.1f} m |",
        f"| Drift % | {metrics['drift_pct']:.1f}% |",
        f"| Max Error Time | {max_err_time:.1f}s |\n",
    ]
    
    # Error growth rate
    err = error_ts["error_m"].values
    t_rel = error_ts["time_s"].values - error_ts["time_s"].values[0]
    if len(err) > 10:
        # Error per second at various points
        mid = len(err) // 2
        lines.append(f"## Error Growth Rate\n")
        lines.append(f"- At 25% of blackout: {err[len(err)//4]:.1f}m")
        lines.append(f"- At 50% of blackout: {err[mid]:.1f}m")
        lines.append(f"- At 75% of blackout: {err[3*len(err)//4]:.1f}m")
        lines.append(f"- At end of blackout: {err[-1]:.1f}m")
        
        avg_rate = err[-1] / t_rel[-1] if t_rel[-1] > 0 else 0
        lines.append(f"- Average error growth rate: {avg_rate:.2f} m/s\n")
    
    # Dynamic events
    lines.append(f"## Dynamic Events During Blackout\n")
    if len(events_df) > 0:
        lines.append("| Event Type | Samples | % of Window | Peak Value | Unit |")
        lines.append("|------------|---------|-------------|------------|------|")
        for _, row in events_df.iterrows():
            lines.append(f"| {row['event_type']} | {row['n_samples']} | {row['pct_window']}% | {row['peak_value']} | {row['unit']} |")
    else:
        lines.append("No significant dynamic events detected in this window.\n")
    
    # Root causes
    lines.append(f"\n## Likely Root Causes of Large Drift\n")
    lines.append("1. **Accelerometer Bias Leakage**: Even small constant bias (~0.01 m/s²) "
                 "causes quadratic position growth: error ≈ ½ × bias × t², leading to "
                 "~18m error in 60s.\n")
    lines.append("2. **Gravity Removal Imperfection**: The Android gravity sensor is "
                 "itself a filtered estimate. Any residual gravity component directly "
                 "integrates into position error.\n")
    lines.append("3. **Phone-to-Vehicle Frame Misalignment**: The simplified yaw-only "
                 "rotation ignores the full 3D orientation matrix. Pitch/roll "
                 "contributions leak acceleration between axes.\n")
    lines.append("4. **Sensor Noise Double-Integration**: White noise in acceleration "
                 "becomes a random walk in velocity and Brownian motion in position "
                 "(error grows as t^1.5 for white noise).\n")
    lines.append("5. **No Velocity Constraints**: A real vehicle cannot accelerate "
                 "sideways or exceed physical speed limits, but raw INS has no such "
                 "knowledge.\n")
    
    lines.append(f"\n## Recommendations for Phase 2+\n")
    lines.append("1. Implement proper phone-to-vehicle calibration (full rotation matrix)")
    lines.append("2. Apply ZUPT (Zero Velocity Update) corrections during detected stops")
    lines.append("3. Use vehicle non-holonomic constraints (no sideslip)")
    lines.append("4. Apply low-pass filtering to remove vibration/bumps before integration")
    lines.append("5. Train AI/ML model to predict velocity directly from IMU windows")
    lines.append("6. Implement EKF/ESKF state estimator for proper sensor fusion")
    
    report_path = out_path / "failure_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Failure analysis report saved to {report_path}")


if __name__ == "__main__":
    import yaml
    import sys
    sys.path.insert(0, ".")
    from Data_details.src.dataset_loader import DatasetLoader
    from Data_details.src.inertial_baseline import run_inertial_baseline
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    with open("Data_details/config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    loader = DatasetLoader(cfg["paths"]["repo_root"], cfg["paths"]["data_cache_dir"])
    seq = cfg["selected_sequence"]
    s_df, _ = loader.load_sequence(seq["smartphone_file"], seq.get("vehicle_file"))
    
    bl_cfg = cfg["blackout_simulation"]
    result = run_inertial_baseline(s_df, bl_cfg["default_start_time_s"], bl_cfg["primary_test_duration_s"])
    
    if result:
        generate_failure_report(result, s_df, cfg["paths"]["reports_dir"], seq["name"])
        print("Failure analysis complete!")
