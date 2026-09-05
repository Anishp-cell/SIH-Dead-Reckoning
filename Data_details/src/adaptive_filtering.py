"""
Phase 3 Tasks 4 & 13: Robust Adaptive Outlier & Spike Handling Module.
Implements Median Absolute Deviation (MAD), Hampel filter, and median filtering
to suppress sensor glitches and impulse spikes without blurring real vehicle dynamics.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def median_filter_1d(
    signal_data: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Applies a rolling 1D median filter.
    Preserves edges and steps (e.g. abrupt braking) while rejecting single-point impulses.
    """
    return pd.Series(signal_data).rolling(window_size, min_periods=1, center=True).median().values


def compute_mad(series: np.ndarray) -> float:
    """
    Computes Median Absolute Deviation (MAD) of a series.
    MAD = median(|x - median(x)|)
    Standard deviation equivalent: sigma_mad = 1.4826 * MAD
    """
    s = series[~np.isnan(series)]
    if len(s) == 0:
        return 0.0
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return float(mad)


def hampel_filter(
    signal_data: np.ndarray,
    window_size: int = 7,
    n_sigmas: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applies the robust Hampel identifier for outlier detection and replacement.
    
    Algorithm:
    1. For each sample k in sliding window of width W:
       - Compute rolling median m_k
       - Compute rolling MAD_k = median(|x_i - m_k|)
       - Compute robust standard deviation S_k = 1.4826 * MAD_k
    2. If |x_k - m_k| > n_sigmas * S_k:
       - Flag x_k as an outlier
       - Replace x_k with median m_k
       
    Parameters:
        signal_data: 1D numpy array
        window_size: odd integer sliding window length (default 7 = 700 ms at 10 Hz)
        n_sigmas: outlier threshold multiplier (default 3.0)
        
    Returns:
        cleaned_signal: 1D array with outliers replaced by rolling median
        outlier_mask: boolean array (True where outlier was detected)
        meta: summary dictionary
    """
    n = len(signal_data)
    cleaned = signal_data.copy()
    outlier_mask = np.zeros(n, dtype=bool)
    
    half_w = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        sub = signal_data[start:end]
        
        med = np.median(sub)
        mad = np.median(np.abs(sub - med))
        s = 1.4826 * mad
        
        # If dispersion is negligible, skip check
        if s < 1e-6:
            continue
            
        if np.abs(signal_data[i] - med) > n_sigmas * s:
            outlier_mask[i] = True
            cleaned[i] = med
            
    n_outliers = int(np.sum(outlier_mask))
    pct_outliers = (n_outliers / n * 100.0) if n > 0 else 0.0
    
    meta = {
        "window_size": window_size,
        "n_sigmas": n_sigmas,
        "n_outliers": n_outliers,
        "outlier_pct": round(pct_outliers, 3),
    }
    
    logger.info(f"Hampel filter: detected {n_outliers} outliers ({pct_outliers:.2f}%)")
    return cleaned, outlier_mask, meta


def plot_hampel_outlier_removal(
    time_s: np.ndarray,
    raw_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    outlier_mask: np.ndarray,
    output_path: str,
    title: str = "Hampel Robust Outlier Filter",
) -> None:
    """Visualizes detected outlier spikes and cleaned signal."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Phase 3 Task 13: {title}", fontsize=13, fontweight="bold")
    
    t_min = (time_s - time_s[0]) / 60.0
    
    ax.plot(t_min, raw_signal, color="#bdc3c7", linewidth=0.8, label="Raw Sensor Signal")
    ax.plot(t_min, cleaned_signal, color="#2980b9", linewidth=1.2, label="Hampel Cleaned Signal (Median Replaced)")
    
    if np.any(outlier_mask):
        ax.scatter(t_min[outlier_mask], raw_signal[outlier_mask], color="#e74c3c", s=30, zorder=5,
                   label=f"Detected Outliers (N={np.sum(outlier_mask)})")
                   
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Hampel plot saved to {output_path}")
