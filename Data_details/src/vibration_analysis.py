"""
Phase 3 Task 2 & 6: Vehicle Vibration Analysis Module.
Decomposes raw sensor signals into kinematic translation and structural vibration.
Quantifies engine vibration, chassis resonance, and road surface roughness.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.filter_design import apply_butterworth_offline, apply_butterworth_causal

logger = logging.getLogger(__name__)


def decompose_vibration(
    signal_data: np.ndarray,
    cutoff_hz: float = 1.8,
    fs: float = 10.0,
    causal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a raw sensor signal into:
    1. Kinematic Component (low-frequency vehicle translation: < cutoff_hz)
    2. Vibration Component (high-frequency engine/road chatter: >= cutoff_hz)
    
    Returns:
        kinematic_signal, vibration_signal
    """
    if causal:
        kinematic = apply_butterworth_causal(signal_data, cutoff_hz=cutoff_hz, fs=fs, order=2)
    else:
        kinematic = apply_butterworth_offline(signal_data, cutoff_hz=cutoff_hz, fs=fs, order=2)
        
    vibration = signal_data - kinematic
    return kinematic, vibration


def compute_vibration_metrics(
    raw_signal: np.ndarray,
    vibration_signal: np.ndarray,
) -> Dict[str, float]:
    """
    Quantifies vibration energy, RMS, peak amplitude, and vibration ratio.
    """
    vib_rms = float(np.sqrt(np.mean(vibration_signal ** 2)))
    raw_rms = float(np.sqrt(np.mean(raw_signal ** 2)))
    peak_vib = float(np.max(np.abs(vibration_signal)))
    
    vib_energy_ratio = (vib_rms / raw_rms * 100.0) if raw_rms > 1e-6 else 0.0
    crest_factor = (peak_vib / vib_rms) if vib_rms > 1e-6 else 1.0

    return {
        "vibration_rms_mps2": round(vib_rms, 4),
        "vibration_peak_mps2": round(peak_vib, 4),
        "vibration_energy_ratio_pct": round(vib_energy_ratio, 2),
        "vibration_crest_factor": round(crest_factor, 2),
    }


def analyze_vibration_across_states(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
    speed_threshold_mps: float = 8.0,
    fs: float = 10.0,
) -> Dict[str, Any]:
    """
    Compares vibration levels between:
    1. Stationary state (engine idling, 0 vehicle translation)
    2. Cruising state (engine under load + road surface interaction)
    """
    speed = df["gps_speed_mps"].fillna(0.0).values if "gps_speed_mps" in df.columns else np.zeros(len(df))
    cruising_mask = speed > speed_threshold_mps
    
    # Analyze vertical acceleration (acc_z) as primary road/chassis vibration axis
    az = df["acc_z"].values
    _, az_vib = decompose_vibration(az, cutoff_hz=1.8, fs=fs)
    
    stat_metrics = compute_vibration_metrics(az[stationary_mask], az_vib[stationary_mask]) if np.any(stationary_mask) else {}
    cruise_metrics = compute_vibration_metrics(az[cruising_mask], az_vib[cruising_mask]) if np.any(cruising_mask) else {}
    
    logger.info(f"Vibration Analysis: Stationary RMS={stat_metrics.get('vibration_rms_mps2', 0):.4f} m/s^2, "
                f"Cruising RMS={cruise_metrics.get('vibration_rms_mps2', 0):.4f} m/s^2")
    
    return {
        "stationary_vibration": stat_metrics,
        "cruising_vibration": cruise_metrics,
    }


def plot_vibration_decomposition(
    time_s: np.ndarray,
    raw_acc: np.ndarray,
    kinematic_acc: np.ndarray,
    vibration_acc: np.ndarray,
    output_path: str,
    title: str = "Vertical Acceleration Vibration Decomposition",
) -> None:
    """Plots raw, low-pass kinematic, and isolated vibration components."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Phase 3: {title}", fontsize=13, fontweight="bold")
    
    t_min = (time_s - time_s[0]) / 60.0
    
    # 1. Raw
    axes[0].plot(t_min, raw_acc, color="#7f8c8d", linewidth=0.5, alpha=0.8, label="Raw Sensor Signal")
    axes[0].set_ylabel("Accel (m/s²)")
    axes[0].set_title("Total Raw Acceleration (Kinematics + Engine & Road Vibration)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Kinematic
    axes[1].plot(t_min, kinematic_acc, color="#2980b9", linewidth=1.0, label="Kinematic Component (< 1.8 Hz)")
    axes[1].set_ylabel("Accel (m/s²)")
    axes[1].set_title("Low-Frequency Vehicle Kinematics (Maneuvers, Gradients)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Vibration
    axes[2].plot(t_min, vibration_acc, color="#e74c3c", linewidth=0.4, alpha=0.7, label="Vibration Component (>= 1.8 Hz)")
    axes[2].set_ylabel("Vibration (m/s²)")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_title("Isolated Structural & Road Vibration (Removed by Signal Processing)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Vibration decomposition plot saved to {output_path}")
