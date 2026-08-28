"""
Task 6: Timestamp Analysis for IO-VNBD sequences.
Verifies sampling frequency, computes dt distribution, detects jitter, gaps, and duplicates.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def analyze_timestamps(df: pd.DataFrame, domain: str = "smartphone") -> Dict[str, Any]:
    """
    Analyzes the time_s and dt columns of a standardized DataFrame.
    Returns a dictionary of timestamp statistics.
    """
    if "time_s" not in df.columns or "dt" not in df.columns:
        logger.error("DataFrame must have 'time_s' and 'dt' columns")
        return {}

    time_s = df["time_s"].values
    dt = df["dt"].values

    # Exclude first dt (which is always set to 0.1 as fill)
    dt_valid = dt[1:]

    stats = {
        "domain": domain,
        "n_records": len(df),
        "time_start_s": float(time_s[0]),
        "time_end_s": float(time_s[-1]),
        "duration_s": float(time_s[-1] - time_s[0]),
        "duration_min": float((time_s[-1] - time_s[0]) / 60.0),
        "dt_mean_s": float(np.mean(dt_valid)),
        "dt_median_s": float(np.median(dt_valid)),
        "dt_std_s": float(np.std(dt_valid)),
        "dt_min_s": float(np.min(dt_valid)),
        "dt_max_s": float(np.max(dt_valid)),
        "dt_p5_s": float(np.percentile(dt_valid, 5)),
        "dt_p25_s": float(np.percentile(dt_valid, 25)),
        "dt_p75_s": float(np.percentile(dt_valid, 75)),
        "dt_p95_s": float(np.percentile(dt_valid, 95)),
        "dt_p99_s": float(np.percentile(dt_valid, 99)),
        "effective_freq_hz": float(1.0 / np.mean(dt_valid)) if np.mean(dt_valid) > 0 else 0,
        "n_duplicate_timestamps": int(np.sum(dt_valid == 0)),
        "n_large_gaps_gt_0_5s": int(np.sum(dt_valid > 0.5)),
        "n_negative_dt": int(np.sum(dt_valid < 0)),
        "jitter_std_ms": float(np.std(dt_valid) * 1000),
    }

    return stats


def plot_timestamp_analysis(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    output_path: str,
    domain: str = "smartphone",
) -> None:
    """Generates dt histogram and dt-over-time plots."""
    dt = df["dt"].values[1:]
    time_s = df["time_s"].values[1:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Timestamp / Sampling Interval Analysis ({domain.title()})", fontsize=14, fontweight="bold")

    # Histogram of dt
    ax1 = axes[0]
    ax1.hist(dt, bins=100, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(stats["dt_mean_s"], color="red", linestyle="--", linewidth=1.5,
                label=f"Mean dt = {stats['dt_mean_s']*1000:.1f} ms")
    ax1.axvline(stats["dt_median_s"], color="orange", linestyle="--", linewidth=1.5,
                label=f"Median dt = {stats['dt_median_s']*1000:.1f} ms")
    ax1.set_xlabel("Sampling Interval dt (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Sampling Intervals")
    ax1.legend()
    ax1.set_xlim(0, min(0.5, np.percentile(dt, 99.5) * 2))

    # dt over time
    ax2 = axes[1]
    ax2.plot(time_s / 60.0, dt * 1000, linewidth=0.4, color="steelblue", alpha=0.7)
    ax2.axhline(100, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Expected 100 ms (10 Hz)")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("dt (milliseconds)")
    ax2.set_title("Sampling Interval Over Time")
    ax2.legend()
    ax2.set_ylim(0, min(500, np.percentile(dt * 1000, 99.5) * 2))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Timestamp analysis plot saved to {output_path}")


def save_timestamp_report(
    s_stats: Dict[str, Any],
    v_stats: Dict[str, Any] = None,
    output_dir: str = "Data_details/outputs/tables",
) -> None:
    """Saves timestamp analysis as CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = [s_stats]
    if v_stats:
        rows.append(v_stats)

    df = pd.DataFrame(rows)
    csv_path = out_path / "timestamp_analysis.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Timestamp report saved to {csv_path}")


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

    s_stats = analyze_timestamps(s_df, "smartphone")
    plot_timestamp_analysis(s_df, s_stats, Path(cfg["paths"]["plots_dir"]) / "sampling_interval_dt_smartphone.png", "smartphone")

    v_stats = None
    if v_df is not None:
        v_stats = analyze_timestamps(v_df, "vehicle")
        plot_timestamp_analysis(v_df, v_stats, Path(cfg["paths"]["plots_dir"]) / "sampling_interval_dt_vehicle.png", "vehicle")

    save_timestamp_report(s_stats, v_stats, cfg["paths"]["tables_dir"])

    print(f"\n{'='*60}")
    print("TIMESTAMP ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Smartphone: {s_stats['n_records']} records, {s_stats['duration_min']:.1f} min, "
          f"effective {s_stats['effective_freq_hz']:.2f} Hz")
    if v_stats:
        print(f"Vehicle:    {v_stats['n_records']} records, {v_stats['duration_min']:.1f} min, "
              f"effective {v_stats['effective_freq_hz']:.2f} Hz")
    print(f"Smartphone jitter: {s_stats['jitter_std_ms']:.1f} ms std")
    print(f"{'='*60}\n")
