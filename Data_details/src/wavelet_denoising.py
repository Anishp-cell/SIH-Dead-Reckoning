"""
Phase 3 Tasks 4 & 5: Wavelet Multiresolution Denoising Module.
Applies Discrete Wavelet Transform (DWT) multi-level decomposition,
Donoho-Johnstone universal thresholding (VisuShrink / SureShrink),
and signal reconstruction for non-stationary transient preservation.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def wavelet_denoise_signal(
    signal_data: np.ndarray,
    wavelet: str = "sym4",
    level: int = 3,
    threshold_mode: str = "soft",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Denoises a 1D sensor signal using Discrete Wavelet Transform (DWT) thresholding.
    
    Algorithm:
    1. Multi-level DWT decomposition into approximation (cA) and detail (cD) coefficients.
    2. Estimate noise standard deviation sigma from the finest detail coefficients (cD_1) via MAD:
       sigma = median(|cD_1|) / 0.6745
    3. Calculate universal threshold lambda = sigma * sqrt(2 * ln(N))
    4. Apply soft or hard thresholding to detail coefficients cD_j.
    5. Reconstruct denoised signal via Inverse DWT (IDWT).
    
    Parameters:
        signal_data: 1D numpy array
        wavelet: wavelet family (default 'sym4' for symmetric smooth transients)
        level: decomposition level (default 3)
        threshold_mode: 'soft' or 'hard'
        
    Returns:
        denoised_signal, metadata_dict
    """
    n = len(signal_data)
    if n < 8:
        return signal_data.copy(), {"threshold": 0.0, "sigma": 0.0}
        
    # 1. Decompose
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    cA = coeffs[0]
    cDs = coeffs[1:]
    
    # 2. Estimate noise standard deviation from finest detail coefficients (cD_1)
    cD1 = cDs[-1]
    sigma = float(np.median(np.abs(cD1)) / 0.6745)
    
    # 3. Universal threshold (Donoho & Johnstone)
    threshold = float(sigma * np.sqrt(2 * np.log(n)))
    
    # 4. Threshold detail coefficients
    thresholded_cDs = []
    for cD in cDs:
        cD_thresh = pywt.threshold(cD, value=threshold, mode=threshold_mode)
        thresholded_cDs.append(cD_thresh)
        
    # 5. Reconstruct
    new_coeffs = [cA] + thresholded_cDs
    denoised = pywt.waverec(new_coeffs, wavelet)
    
    # Match original length if odd length padding occurred
    denoised = denoised[:n]
    
    meta = {
        "wavelet": wavelet,
        "level": level,
        "mode": threshold_mode,
        "sigma_noise": round(sigma, 5),
        "threshold": round(threshold, 5),
        "noise_reduction_pct": round(float((1.0 - np.std(denoised) / (np.std(signal_data) + 1e-8)) * 100), 2),
    }
    
    return denoised, meta


def plot_wavelet_comparison(
    time_s: np.ndarray,
    raw_signal: np.ndarray,
    wavelet_signal: np.ndarray,
    butter_signal: np.ndarray,
    output_path: str,
    signal_name: str = "Forward Acceleration (m/s²)",
) -> None:
    """Plots side-by-side comparison between Wavelet and Butterworth denoising."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Phase 3 Task 4: Wavelet Denoising vs Butterworth Filter Comparison — {signal_name}",
                 fontsize=13, fontweight="bold")
    
    t_rel = (time_s - time_s[0])
    
    # 1. Overlaid Comparison
    axes[0].plot(t_rel, raw_signal, color="#bdc3c7", linewidth=0.8, alpha=0.7, label="Raw Sensor Signal")
    axes[0].plot(t_rel, butter_signal, color="#e74c3c", linewidth=1.5, alpha=0.8, label="Butterworth Low-Pass (1.5 Hz)")
    axes[0].plot(t_rel, wavelet_signal, color="#2980b9", linewidth=1.8, label="Wavelet Denoised (sym4, Level 3)")
    axes[0].set_ylabel(signal_name)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Filtered Signals Overlaid")
    
    # 2. Residuals (Raw - Filtered)
    res_butter = raw_signal - butter_signal
    res_wavelet = raw_signal - wavelet_signal
    axes[1].plot(t_rel, res_butter, color="#e74c3c", linewidth=0.5, alpha=0.6, label="Butterworth Removed Residual")
    axes[1].plot(t_rel, res_wavelet, color="#2980b9", linewidth=0.5, alpha=0.6, label="Wavelet Removed Residual")
    axes[1].set_ylabel("Removed Noise (m/s²)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Residual Noise Removed by Filters")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wavelet comparison plot saved to {output_path}")
