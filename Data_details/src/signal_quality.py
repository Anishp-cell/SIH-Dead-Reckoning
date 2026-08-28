"""
Phase 3 Task 5 & 10: Signal Quality, Distortion Metrics and Event Preservation Module.
Quantifies noise reduction, signal distortion, peak preservation during hard maneuvers,
spectral attenuation, and phase delay for candidate filters.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_filter_quality_metrics(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    filter_name: str,
    fs: float = 10.0,
) -> Dict[str, Any]:
    """
    Computes rigorous quality and distortion metrics between raw and filtered signals.
    """
    # 1. Dispersion & RMS
    raw_std = float(np.std(raw_signal))
    filt_std = float(np.std(filtered_signal))
    raw_rms = float(np.sqrt(np.mean(raw_signal ** 2)))
    filt_rms = float(np.sqrt(np.mean(filtered_signal ** 2)))
    
    # 2. Noise Reduction Percentage
    noise_reduction_pct = ((raw_std - filt_std) / raw_std * 100.0) if raw_std > 1e-6 else 0.0
    rms_ratio = (filt_rms / raw_rms) if raw_rms > 1e-6 else 1.0
    
    # 3. Peak Preservation Ratio
    peak_raw = float(np.max(np.abs(raw_signal)))
    peak_filt = float(np.max(np.abs(filtered_signal)))
    peak_preservation_pct = (peak_filt / peak_raw * 100.0) if peak_raw > 1e-6 else 100.0
    
    # 4. Correlation / Signal Fidelity (Pearson r)
    if raw_std > 1e-6 and filt_std > 1e-6:
        corr_matrix = np.corrcoef(raw_signal, filtered_signal)
        correlation_r = float(corr_matrix[0, 1])
    else:
        correlation_r = 1.0
        
    # 5. Residual Noise Energy
    residual = raw_signal - filtered_signal
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    
    # 6. High-Frequency Spectral Attenuation (> 2.5 Hz)
    fft_raw = np.abs(np.fft.rfft(raw_signal - np.mean(raw_signal)))
    fft_filt = np.abs(np.fft.rfft(filtered_signal - np.mean(filtered_signal)))
    freqs = np.fft.rfftfreq(len(raw_signal), d=1.0 / fs)
    
    hf_mask = freqs >= 2.5
    raw_hf_pwr = float(np.sum(fft_raw[hf_mask] ** 2)) if np.any(hf_mask) else 1e-6
    filt_hf_pwr = float(np.sum(fft_filt[hf_mask] ** 2)) if np.any(hf_mask) else 0.0
    hf_attenuation_pct = ((raw_hf_pwr - filt_hf_pwr) / raw_hf_pwr * 100.0) if raw_hf_pwr > 1e-6 else 0.0

    return {
        "filter_name": filter_name,
        "noise_reduction_pct": round(noise_reduction_pct, 2),
        "correlation_fidelity": round(correlation_r, 4),
        "peak_preservation_pct": round(peak_preservation_pct, 2),
        "rms_ratio": round(rms_ratio, 4),
        "residual_rms_mps2": round(residual_rms, 4),
        "hf_attenuation_pct": round(hf_attenuation_pct, 2),
    }


def evaluate_event_preservation(
    df: pd.DataFrame,
    filtered_signals: Dict[str, np.ndarray],
    event_indices: Tuple[int, int],
    event_name: str = "Hard_Braking",
) -> pd.DataFrame:
    """
    Evaluates whether dynamic maneuvers (e.g. hard braking, sharp turns)
    are retained or excessively blurred by candidate filters.
    """
    start, end = event_indices
    sub_raw = df["acc_y"].values[start:end]
    peak_raw = float(np.min(sub_raw))  # Maximum deceleration
    
    records = []
    for fname, filt_arr in filtered_signals.items():
        sub_filt = filt_arr[start:end]
        peak_filt = float(np.min(sub_filt))
        attenuation_pct = (abs(peak_raw - peak_filt) / abs(peak_raw) * 100.0) if abs(peak_raw) > 1e-6 else 0.0
        
        records.append({
            "event": event_name,
            "filter": fname,
            "raw_peak_mps2": round(peak_raw, 3),
            "filtered_peak_mps2": round(peak_filt, 3),
            "peak_attenuation_pct": round(attenuation_pct, 2),
            "status": "Preserved" if attenuation_pct < 20.0 else "Overfiltered",
        })
        
    return pd.DataFrame(records)
