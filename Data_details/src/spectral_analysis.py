"""
Phase 3 Tasks 2 & 3: Spectral Analysis and Frequency Characterization Module.
Computes FFT magnitude/power spectra, Welch Power Spectral Density (PSD),
identifies dominant frequency peaks, spectral energy distribution,
and segments driving events to produce frequency_characterization.csv.
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


def compute_fft_spectrum(
    signal_data: np.ndarray,
    fs: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes single-sided FFT magnitude and power spectrum.
    
    Parameters:
        signal_data: 1D numpy array
        fs: sampling rate in Hz (default 10 Hz)
        
    Returns:
        freqs: 1D array of frequencies [0, fs/2]
        magnitude: 1D array of single-sided magnitude
        power: 1D array of single-sided power spectrum
    """
    s = signal_data - np.mean(signal_data)  # remove DC component
    n = len(s)
    if n < 2:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
        
    fft_vals = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    
    magnitude = (2.0 / n) * np.abs(fft_vals)
    magnitude[0] = magnitude[0] / 2.0  # DC component adjustment
    power = (magnitude ** 2) / 2.0
    power[0] = magnitude[0] ** 2
    
    return freqs, magnitude, power


def compute_welch_psd(
    signal_data: np.ndarray,
    fs: float = 10.0,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Power Spectral Density (PSD) using Welch's averaged periodogram method.
    """
    s = signal_data - np.mean(signal_data)
    n = len(s)
    if nperseg is None:
        nperseg = min(n, int(fs * 10))  # 10 second window default
    if nperseg < 4:
        nperseg = n
        
    freqs, psd = signal.welch(s, fs=fs, window="hann", nperseg=nperseg, scaling="density")
    return freqs, psd


def extract_spectral_metrics(
    signal_data: np.ndarray,
    fs: float = 10.0,
    hf_cutoff: float = 2.5,
) -> Dict[str, float]:
    """
    Computes spectral summary metrics: dominant frequency, total power,
    and high-frequency energy ratio.
    """
    freqs, mag, pwr = compute_fft_spectrum(signal_data, fs)
    
    if len(freqs) <= 1:
        return {"dominant_freq_hz": 0.0, "total_power": 0.0, "hf_ratio_pct": 0.0}
        
    # Ignore DC bin (index 0) for dominant frequency
    ac_indices = np.arange(1, len(freqs))
    dom_idx = ac_indices[np.argmax(pwr[ac_indices])] if len(ac_indices) > 0 else 0
    dom_freq = float(freqs[dom_idx])
    
    total_power = float(np.sum(pwr))
    hf_mask = freqs >= hf_cutoff
    hf_power = float(np.sum(pwr[hf_mask])) if np.any(hf_mask) else 0.0
    hf_ratio = (hf_power / total_power * 100.0) if total_power > 1e-12 else 0.0
    
    return {
        "dominant_freq_hz": round(dom_freq, 3),
        "total_power": round(total_power, 5),
        "hf_power": round(hf_power, 5),
        "hf_ratio_pct": round(hf_ratio, 2),
    }


def segment_driving_events(df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """
    Automatically detects 7 representative driving event slices from the sequence.
    
    Returns:
        dict mapping event_name -> (start_idx, end_idx)
    """
    n = len(df)
    speed = df["gps_speed_mps"].fillna(0.0).values if "gps_speed_mps" in df.columns else np.zeros(n)
    dt = df["dt"].replace(0, 0.1).values
    
    # Calculate vehicle acceleration from speed
    speed_diff = np.diff(speed, prepend=speed[0])
    accel_long = speed_diff / dt
    
    gyro_z = np.abs(df["gyro_z"].values)
    acc_z = df["acc_z"].values
    
    # 1. Stationary: lowest speed and lowest gyro
    stationary_cand = np.where((speed < 0.2) & (gyro_z < 0.02))[0]
    stat_idx = (stationary_cand[100], stationary_cand[100] + 150) if len(stationary_cand) > 250 else (0, 150)
    
    # 2. Hard Braking: lowest negative longitudinal acceleration
    brake_peak = np.argmin(accel_long[200:-200]) + 200
    brake_idx = (max(0, brake_peak - 40), min(n, brake_peak + 60))
    
    # 3. Hard Acceleration: highest positive acceleration
    accel_peak = np.argmax(accel_long[200:-200]) + 200
    accel_idx = (max(0, accel_peak - 40), min(n, accel_peak + 60))
    
    # 4. Smooth Acceleration: cruising acceleration (0.5 to 1.0 m/s^2)
    smooth_cand = np.where((accel_long > 0.4) & (accel_long < 1.0) & (speed > 5.0) & (gyro_z < 0.03))[0]
    smooth_idx = (smooth_cand[50], smooth_cand[50] + 100) if len(smooth_cand) > 100 else (1000, 1100)
    
    # 5. Turning / Cornering: highest yaw rate
    turn_peak = np.argmax(gyro_z[200:-200]) + 200
    turn_idx = (max(0, turn_peak - 50), min(n, turn_peak + 50))
    
    # 6. Road Bump / Pothole: high vertical acceleration variance
    acc_z_diff = np.abs(np.diff(acc_z, prepend=acc_z[0]))
    bump_peak = np.argmax(acc_z_diff[200:-200]) + 200
    bump_idx = (max(0, bump_peak - 30), min(n, bump_peak + 30))
    
    # 7. High-Vibration Cruising: high speed straight road
    cruise_cand = np.where((speed > 12.0) & (gyro_z < 0.03))[0]
    cruise_idx = (cruise_cand[100], cruise_cand[100] + 150) if len(cruise_cand) > 200 else (3000, 3150)
    
    events = {
        "1_Stationary": stat_idx,
        "2_Smooth_Acceleration": smooth_idx,
        "3_Hard_Acceleration": accel_idx,
        "4_Hard_Braking": brake_idx,
        "5_Sharp_Turn": turn_idx,
        "6_Road_Bump_Pothole": bump_idx,
        "7_High_Vibration_Cruising": cruise_idx,
    }
    
    logger.info("Automatically extracted 7 driving event segments for spectral analysis")
    return events


def build_frequency_characterization_table(
    df: pd.DataFrame,
    events: Dict[str, Tuple[int, int]],
    fs: float = 10.0,
) -> pd.DataFrame:
    """
    Extracts spectral properties across each driving event segment and saves
    frequency_characterization.csv.
    """
    records = []
    time_s = df["time_s"].values
    
    for event_name, (start_idx, end_idx) in events.items():
        sub_df = df.iloc[start_idx:end_idx]
        duration_s = float(time_s[end_idx-1] - time_s[start_idx])
        n_samples = len(sub_df)
        
        # Accelerometer norm (m/s^2)
        a_norm = np.sqrt(sub_df["acc_x"]**2 + sub_df["acc_y"]**2 + sub_df["acc_z"]**2).values
        # Dynamic horizontal acceleration (approximate)
        a_horiz = np.sqrt(sub_df["acc_x"]**2 + sub_df["acc_y"]**2).values
        # Gyroscope norm (rad/s)
        w_norm = np.sqrt(sub_df["gyro_x"]**2 + sub_df["gyro_y"]**2 + sub_df["gyro_z"]**2).values
        
        # Metrics
        spec_a = extract_spectral_metrics(a_norm, fs=fs, hf_cutoff=2.5)
        spec_w = extract_spectral_metrics(w_norm, fs=fs, hf_cutoff=2.5)
        
        rms_acc = float(np.sqrt(np.mean(a_norm**2)))
        peak_acc = float(np.max(np.abs(a_norm)))
        crest_factor = round(peak_acc / rms_acc, 2) if rms_acc > 0 else 1.0
        
        mean_speed = float(sub_df["gps_speed_mps"].mean()) if "gps_speed_mps" in sub_df.columns else 0.0
        
        records.append({
            "event_name": event_name,
            "start_time_s": round(float(time_s[start_idx]), 1),
            "end_time_s": round(float(time_s[end_idx-1]), 1),
            "duration_s": round(duration_s, 1),
            "samples": n_samples,
            "mean_speed_mps": round(mean_speed, 2),
            "accel_dominant_freq_hz": spec_a["dominant_freq_hz"],
            "accel_total_power": spec_a["total_power"],
            "accel_hf_energy_ratio_pct": spec_a["hf_ratio_pct"],
            "gyro_dominant_freq_hz": spec_w["dominant_freq_hz"],
            "gyro_total_power": spec_w["total_power"],
            "gyro_hf_energy_ratio_pct": spec_w["hf_ratio_pct"],
            "peak_accel_mps2": round(peak_acc, 2),
            "rms_accel_mps2": round(rms_acc, 2),
            "crest_factor": crest_factor,
        })
        
    char_df = pd.DataFrame(records)
    logger.info(f"Built frequency characterization table ({len(char_df)} events)")
    return char_df


def plot_spectral_analysis(
    df: pd.DataFrame,
    events: Dict[str, Tuple[int, int]],
    output_fft_path: str,
    output_psd_path: str,
    sequence_name: str = "S1",
    fs: float = 10.0,
) -> None:
    """
    Generates publication-quality FFT and Welch PSD comparison plots across driving events.
    """
    # 1. FFT Comparison Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Phase 3: Fast Fourier Transform (FFT) Spectra — Sequence {sequence_name}",
                 fontsize=14, fontweight="bold")
    
    axes[0].set_title("Full Sequence Accelerometer Magnitude Spectrum", fontsize=11, fontweight="bold")
    axes[1].set_title("Full Sequence Gyroscope Magnitude Spectrum", fontsize=11, fontweight="bold")
    axes[2].set_title("Welch PSD vs Nyquist Boundary (5.0 Hz)", fontsize=11, fontweight="bold")
    
    # Compute FFT for full sequence
    f_ax, m_ax, _ = compute_fft_spectrum(df["acc_x"].values, fs)
    _, m_ay, _ = compute_fft_spectrum(df["acc_y"].values, fs)
    _, m_az, _ = compute_fft_spectrum(df["acc_z"].values, fs)
    
    axes[0].plot(f_ax, m_ax, color="#e74c3c", linewidth=0.8, alpha=0.8, label="Acc X (Lateral)")
    axes[0].plot(f_ax, m_ay, color="#2ecc71", linewidth=0.8, alpha=0.8, label="Acc Y (Forward)")
    axes[0].plot(f_ax, m_az, color="#3498db", linewidth=0.8, alpha=0.8, label="Acc Z (Vertical)")
    axes[0].axvline(2.5, color="gray", linestyle="--", alpha=0.7, label="Mid-Band (2.5 Hz)")
    axes[0].set_ylabel("Magnitude (m/s²)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    f_gx, m_gx, _ = compute_fft_spectrum(df["gyro_x"].values, fs)
    _, m_gy, _ = compute_fft_spectrum(df["gyro_y"].values, fs)
    _, m_gz, _ = compute_fft_spectrum(df["gyro_z"].values, fs)
    
    axes[1].plot(f_gx, m_gx, color="#e74c3c", linewidth=0.8, alpha=0.8, label="Gyro X (Roll)")
    axes[1].plot(f_gx, m_gy, color="#2ecc71", linewidth=0.8, alpha=0.8, label="Gyro Y (Pitch)")
    axes[1].plot(f_gx, m_gz, color="#9b59b6", linewidth=0.8, alpha=0.8, label="Gyro Z (Yaw)")
    axes[1].axvline(2.5, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Magnitude (rad/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Welch PSD full
    f_psd, psd_ay = compute_welch_psd(df["acc_y"].values, fs)
    f_psd, psd_az = compute_welch_psd(df["acc_z"].values, fs)
    axes[2].semilogy(f_psd, psd_ay, color="#2ecc71", linewidth=1.2, label="Acc Y (Longitudinal PSD)")
    axes[2].semilogy(f_psd, psd_az, color="#3498db", linewidth=1.2, label="Acc Z (Vertical PSD)")
    axes[2].axvspan(0.0, 1.8, color="#2ecc71", alpha=0.15, label="Vehicle Dynamics (< 1.8 Hz)")
    axes[2].axvspan(2.5, 5.0, color="#e74c3c", alpha=0.15, label="Vibration/Noise Zone (> 2.5 Hz)")
    axes[2].set_ylabel("PSD ((m/s²)²/Hz)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_fft_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"FFT spectra plot saved to {output_fft_path}")
    
    # 2. Welch PSD Across Events Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Phase 3: Power Spectral Density (PSD) Across Distinct Driving Events — {sequence_name}",
                 fontsize=14, fontweight="bold")
    
    event_colors = {
        "1_Stationary": "#7f8c8d",
        "3_Hard_Acceleration": "#27ae60",
        "4_Hard_Braking": "#e74c3c",
        "5_Sharp_Turn": "#8e44ad",
        "6_Road_Bump_Pothole": "#d35400",
        "7_High_Vibration_Cruising": "#2980b9",
    }
    
    ax_tl = axes[0, 0]  # Forward Accel PSD
    ax_tr = axes[0, 1]  # Vertical Accel PSD
    ax_bl = axes[1, 0]  # Yaw Rate PSD
    ax_br = axes[1, 1]  # Acceleration Magnitude PSD
    
    for ev_name, (s_idx, e_idx) in events.items():
        if ev_name not in event_colors:
            continue
        c = event_colors[ev_name]
        lbl = ev_name.replace("_", " ")
        sub = df.iloc[s_idx:e_idx]
        
        f, p_ay = compute_welch_psd(sub["acc_y"].values, fs, nperseg=min(len(sub), 64))
        ax_tl.semilogy(f, p_ay, color=c, linewidth=1.2, label=lbl)
        
        f, p_az = compute_welch_psd(sub["acc_z"].values, fs, nperseg=min(len(sub), 64))
        ax_tr.semilogy(f, p_az, color=c, linewidth=1.2, label=lbl)
        
        f, p_gz = compute_welch_psd(sub["gyro_z"].values, fs, nperseg=min(len(sub), 64))
        ax_bl.semilogy(f, p_gz, color=c, linewidth=1.2, label=lbl)
        
        a_mag = np.sqrt(sub["acc_x"]**2 + sub["acc_y"]**2 + sub["acc_z"]**2).values
        f, p_mag = compute_welch_psd(a_mag, fs, nperseg=min(len(sub), 64))
        ax_br.semilogy(f, p_mag, color=c, linewidth=1.2, label=lbl)
        
    ax_tl.set_title("Forward Acceleration (Acc Y) PSD", fontsize=11, fontweight="bold")
    ax_tl.set_ylabel("PSD ((m/s²)²/Hz)")
    ax_tl.set_xlabel("Frequency (Hz)")
    ax_tl.legend(fontsize=8)
    ax_tl.grid(True, alpha=0.3)
    
    ax_tr.set_title("Vertical Acceleration (Acc Z) PSD", fontsize=11, fontweight="bold")
    ax_tr.set_ylabel("PSD ((m/s²)²/Hz)")
    ax_tr.set_xlabel("Frequency (Hz)")
    ax_tr.legend(fontsize=8)
    ax_tr.grid(True, alpha=0.3)
    
    ax_bl.set_title("Yaw Rate (Gyro Z) PSD", fontsize=11, fontweight="bold")
    ax_bl.set_ylabel("PSD ((rad/s)²/Hz)")
    ax_bl.set_xlabel("Frequency (Hz)")
    ax_bl.legend(fontsize=8)
    ax_bl.grid(True, alpha=0.3)
    
    ax_br.set_title("Total Acceleration Norm PSD", fontsize=11, fontweight="bold")
    ax_br.set_ylabel("PSD ((m/s²)²/Hz)")
    ax_br.set_xlabel("Frequency (Hz)")
    ax_br.legend(fontsize=8)
    ax_br.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_psd_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Welch PSD event plot saved to {output_psd_path}")
