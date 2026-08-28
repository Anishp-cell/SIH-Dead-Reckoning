"""
Phase 3 Task 2: Signal Analysis Module for IO-VNBD Smartphone Sensors.
Provides statistical characterization, skewness, kurtosis, signal-to-noise ratio (SNR),
amplitude distribution histograms, and multi-axis time-series visualization.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_signal_statistics(series: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical moments and dispersion metrics for a 1D sensor signal.
    """
    s = pd.Series(series).dropna()
    mean_val = float(s.mean())
    std_val = float(s.std())
    rms_val = float(np.sqrt(np.mean(s ** 2)))
    
    # Skewness & Kurtosis
    skew_val = float(s.skew()) if std_val > 1e-8 else 0.0
    kurt_val = float(s.kurtosis()) if std_val > 1e-8 else 0.0
    
    peak_val = float(np.max(np.abs(s)))
    crest_factor = float(peak_val / rms_val) if rms_val > 1e-8 else 1.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "rms": round(rms_val, 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "peak": round(peak_val, 4),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "crest_factor": round(crest_factor, 4),
    }


def analyze_raw_imu_signals(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyzes all raw accelerometer, gyroscope, and magnetometer channels.
    """
    channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    if "mag_x" in df.columns:
        channels.extend(["mag_x", "mag_y", "mag_z"])
        
    stats = {}
    for col in channels:
        if col in df.columns:
            stats[col] = compute_signal_statistics(df[col].values)
            
    logger.info(f"Computed signal statistics for {len(stats)} channels")
    return stats


def plot_raw_signal_histograms(
    df: pd.DataFrame,
    output_path: str,
    sequence_name: str = "S1",
) -> None:
    """
    Plots amplitude distribution histograms with fitted normal overlays for IMU axes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Phase 3: Raw IMU Signal Amplitude Distributions — Sequence {sequence_name}",
                 fontsize=14, fontweight="bold")
    
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    
    for i, col in enumerate(acc_cols):
        ax = axes[0, i]
        data = df[col].dropna().values
        ax.hist(data, bins=60, density=True, color=colors[i], alpha=0.6, edgecolor="black", linewidth=0.5)
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(mu - 3.5 * std, mu + 3.5 * std, 100)
        p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
        ax.plot(x, p, "k--", linewidth=1.5, label=f"Normal Fit (μ={mu:.2f}, σ={std:.2f})")
        ax.set_title(f"{col.upper()} (m/s²)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Acceleration (m/s²)")
        ax.set_ylabel("Probability Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    for i, col in enumerate(gyro_cols):
        ax = axes[1, i]
        data = df[col].dropna().values
        ax.hist(data, bins=60, density=True, color=colors[i], alpha=0.6, edgecolor="black", linewidth=0.5)
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(mu - 3.5 * std, mu + 3.5 * std, 100)
        p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
        ax.plot(x, p, "k--", linewidth=1.5, label=f"Normal Fit (μ={mu:.3f}, σ={std:.3f})")
        ax.set_title(f"{col.upper()} (rad/s)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Angular Velocity (rad/s)")
        ax.set_ylabel("Probability Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Signal histograms plot saved to {output_path}")
