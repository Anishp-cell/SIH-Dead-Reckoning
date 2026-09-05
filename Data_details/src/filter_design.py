"""
Phase 3 Tasks 4, 8, 9, 10: Digital Filter Design, Causal vs Non-Causal Engine.
Implements Moving Average, Butterworth Low-Pass and Band-Pass filters.
Maintains two strictly separated operational modes:
- Mode A: Offline Zero-Phase Research Filter (filtfilt - uses future samples)
- Mode B: Real-Time Causal Streaming Filter (lfilter with persistent state zi)
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ============================================================================
# 1. MOVING AVERAGE FILTER
# ============================================================================

def moving_average_filter(
    signal_data: np.ndarray,
    window_size: int = 5,
    mode: str = "causal",
) -> np.ndarray:
    """
    Applies a moving average filter.
    
    Parameters:
        signal_data: 1D numpy array
        window_size: number of samples (default 5 = 500 ms at 10 Hz)
        mode: 'causal' (uses past samples only) or 'centered' (uses future samples)
    """
    if window_size <= 1:
        return signal_data.copy()
        
    if mode == "causal":
        # Strictly causal rolling mean
        weights = np.ones(window_size) / window_size
        filtered = np.convolve(signal_data, weights, mode="full")[:len(signal_data)]
        # For initial transient, use expanding mean
        for i in range(min(window_size, len(signal_data))):
            filtered[i] = np.mean(signal_data[:i+1])
        return filtered
    else:
        # Centered non-causal moving average
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal_data, kernel, mode="same")


# ============================================================================
# 2. BUTTERWORTH LOW-PASS & BAND-PASS FILTER GENERATOR
# ============================================================================

def design_butterworth_lowpass(
    cutoff_hz: float = 1.5,
    fs: float = 10.0,
    order: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Designs a discrete Butterworth low-pass digital filter.
    
    Parameters:
        cutoff_hz: 3dB cutoff frequency in Hz
        fs: sampling frequency (10.0 Hz)
        order: filter order (default 2)
        
    Returns:
        b, a: numerator and denominator filter polynomials
    """
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def design_butterworth_bandpass(
    lowcut_hz: float = 0.1,
    highcut_hz: float = 2.5,
    fs: float = 10.0,
    order: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Designs a discrete Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    low = max(lowcut_hz / nyquist, 0.01)
    high = min(highcut_hz / nyquist, 0.99)
    b, a = signal.butter(order, [low, high], btype="band", analog=False)
    return b, a


# ============================================================================
# 3. OPERATIONAL MODES: OFFLINE ZERO-PHASE VS REAL-TIME CAUSAL
# ============================================================================

def apply_butterworth_offline(
    signal_data: np.ndarray,
    cutoff_hz: float = 1.5,
    fs: float = 10.0,
    order: int = 2,
) -> np.ndarray:
    """
    MODE A: Offline Zero-Phase Forward-Backward Filtering (filtfilt).
    
    WARNING:
    This filter passes the signal forward and backward.
    It accesses FUTURE samples x[k+1...N].
    It CANNOT be deployed on a smartphone in real time.
    Used strictly as an offline research benchmark upper bound.
    """
    b, a = design_butterworth_lowpass(cutoff_hz, fs, order)
    # Ensure signal is longer than padlen
    padlen = 3 * max(len(a), len(b))
    if len(signal_data) <= padlen:
        return signal_data.copy()
    return signal.filtfilt(b, a, signal_data)


def apply_butterworth_causal(
    signal_data: np.ndarray,
    cutoff_hz: float = 1.5,
    fs: float = 10.0,
    order: int = 2,
) -> np.ndarray:
    """
    MODE B: Real-Time Causal IIR Filtering (lfilter).
    
    Operates strictly forward in time.
    Calculates y[k] using only past inputs x[k-i] and past outputs y[k-j].
    Zero future sample leakage. Deployed on real-time hardware.
    """
    b, a = design_butterworth_lowpass(cutoff_hz, fs, order)
    # Initialize state with initial value to prevent startup step transients
    zi = signal.lfilter_zi(b, a) * signal_data[0]
    filtered, _ = signal.lfilter(b, a, signal_data, zi=zi)
    return filtered


# ============================================================================
# 4. STREAMING CAUSAL FILTER CLASS (FOR EMBEDDED / REAL-TIME USE)
# ============================================================================

class RealTimeCausalFilter:
    """
    Stateful real-time digital filter processing one sensor measurement at a time.
    Guarantees O(1) time complexity per sample and zero future data leakage.
    """
    
    def __init__(self, cutoff_hz: float = 1.5, fs: float = 10.0, order: int = 2):
        self.cutoff_hz = cutoff_hz
        self.fs = fs
        self.order = order
        self.b, self.a = design_butterworth_lowpass(cutoff_hz, fs, order)
        self.zi = None
        self.initialized = False
        
    def reset(self):
        self.zi = None
        self.initialized = False
        
    def process_sample(self, x: float) -> float:
        """Processes a single float measurement in real time."""
        if not self.initialized:
            self.zi = signal.lfilter_zi(self.b, self.a) * x
            self.initialized = True
            
        y, self.zi = signal.lfilter(self.b, self.a, [x], zi=self.zi)
        return float(y[0])
        
    def process_vector(self, x_arr: np.ndarray) -> np.ndarray:
        """Processes a 1D batch causally preserving filter state."""
        if len(x_arr) == 0:
            return np.array([])
        if not self.initialized:
            self.zi = signal.lfilter_zi(self.b, self.a) * x_arr[0]
            self.initialized = True
            
        y_arr, self.zi = signal.lfilter(self.b, self.a, x_arr, zi=self.zi)
        return y_arr


# ============================================================================
# 5. CUTOFF FREQUENCY SWEEP ANALYSIS
# ============================================================================

def sweep_butterworth_cutoffs(
    signal_data: np.ndarray,
    cutoffs: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
    fs: float = 10.0,
    order: int = 2,
    causal: bool = True,
) -> Dict[float, np.ndarray]:
    """Sweeps multiple candidate cutoff frequencies over a signal."""
    results = {}
    for fc in cutoffs:
        if causal:
            results[fc] = apply_butterworth_causal(signal_data, cutoff_hz=fc, fs=fs, order=order)
        else:
            results[fc] = apply_butterworth_offline(signal_data, cutoff_hz=fc, fs=fs, order=order)
    return results


# ============================================================================
# 6. PLOTTING & VISUALIZATION
# ============================================================================

def plot_filter_frequency_responses(
    cutoffs: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
    fs: float = 10.0,
    order: int = 2,
    output_path: str = "Data_details/outputs/phase3/plots/filter_frequency_responses.png",
) -> None:
    """Plots magnitude and phase frequency response curves for candidate filters."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Phase 3 Task 4: Butterworth Low-Pass Filter Frequency Responses (Order {order}, fs={fs}Hz)",
                 fontsize=13, fontweight="bold")
    
    colors = ["#9b59b6", "#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    
    for i, fc in enumerate(cutoffs):
        b, a = design_butterworth_lowpass(fc, fs, order)
        w, h = signal.freqz(b, a, worN=512, fs=fs)
        
        # Magnitude (dB)
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
        axes[0].plot(w, mag_db, color=colors[i % len(colors)], linewidth=1.5, label=f"fc = {fc:.1f} Hz")
        
        # Phase (degrees)
        phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi
        axes[1].plot(w, phase_deg, color=colors[i % len(colors)], linewidth=1.5, label=f"fc = {fc:.1f} Hz")
        
    axes[0].axhline(-3, color="black", linestyle=":", label="-3 dB Cutoff Line")
    axes[0].axvline(fs / 2, color="gray", linestyle="--", label="Nyquist (5 Hz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_ylim(-50, 5)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Magnitude Attenuation vs Frequency")
    
    axes[1].set_ylabel("Phase Delay (degrees)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Phase Response (Group Delay Characteristic)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Filter frequency response plot saved to {output_path}")


def plot_causal_vs_noncausal_comparison(
    time_s: np.ndarray,
    raw_signal: np.ndarray,
    causal_signal: np.ndarray,
    offline_signal: np.ndarray,
    output_path: str,
    event_title: str = "Braking Maneuver",
) -> None:
    """Plots side-by-side comparison of causal (Mode B) vs offline non-causal (Mode A) filtering."""
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(f"Phase 3 Task 8: Causal (Real-Time) vs Non-Causal (Zero-Phase) Filter Comparison\n{event_title}",
                 fontsize=13, fontweight="bold")
    
    t_rel = time_s - time_s[0]
    ax.plot(t_rel, raw_signal, color="#95a5a6", linewidth=1.0, alpha=0.6, label="Raw Sensor Signal (with Vibration)")
    ax.plot(t_rel, offline_signal, color="#e74c3c", linewidth=2.0, linestyle="--",
            label="Mode A: Offline Zero-Phase (filtfilt — Uses Future Samples)")
    ax.plot(t_rel, causal_signal, color="#2980b9", linewidth=2.2,
            label="Mode B: Real-Time Causal (lfilter — Deployed on Smartphone)")
    
    ax.set_xlabel("Time Relative to Event Start (seconds)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Causal vs non-causal comparison plot saved to {output_path}")
