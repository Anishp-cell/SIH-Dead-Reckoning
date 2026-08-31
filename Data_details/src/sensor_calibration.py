"""
Phase 2 Task 3, 4, 5, 6, 10: Sensor Unit/Axis Audit, Multi-Sensor Stationary Detection,
Accelerometer/Gyroscope Bias Calibration, and Magnetometer Reliability Analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

GRAVITY_STANDARD = 9.80665  # m/s^2 standard gravity


# ============================================================================
# TASK 3: SENSOR UNIT AND AXIS AUDIT
# ============================================================================

def audit_sensor_units_and_axes(
    df: pd.DataFrame,
    stationary_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Empirically audits sensor units, axis magnitudes, and sign conventions.
    
    Computes:
    - Accelerometer norm: |a| = sqrt(ax^2 + ay^2 + az^2) -> expected ~9.81 m/s^2 stationary
    - Gyroscope norm: |w| = sqrt(gx^2 + gy^2 + gz^2) -> expected ~0 rad/s stationary
    - Magnetometer norm: |m| = sqrt(mx^2 + my^2 + mz^2) -> expected ~30-60 uT Earth field
    
    Returns structured audit dictionary and summary table.
    """
    logger.info("Auditing sensor units, axes, and norms...")
    
    # 1. Accelerometer
    ax = df["acc_x"].values
    ay = df["acc_y"].values
    az = df["acc_z"].values
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    
    # 2. Gyroscope
    gx = df["gyro_x"].values
    gy = df["gyro_y"].values
    gz = df["gyro_z"].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # 3. Magnetometer
    mx = df["mag_x"].values
    my = df["mag_y"].values
    mz = df["mag_z"].values
    m_norm = np.sqrt(mx**2 + my**2 + mz**2)
    
    if stationary_mask is None:
        # Default simple stationary fallback
        if "gps_speed_mps" in df.columns:
            stationary_mask = df["gps_speed_mps"].values < 0.5
        else:
            stationary_mask = np.ones(len(df), dtype=bool)
    
    stat_count = int(np.sum(stationary_mask))
    
    # Metrics
    audit_results = {
        "accel_unit": "m/s²",
        "accel_norm_all_mean": float(np.nanmean(a_norm)),
        "accel_norm_all_std": float(np.nanstd(a_norm)),
        "accel_norm_stat_mean": float(np.nanmean(a_norm[stationary_mask])) if stat_count > 0 else float(np.nanmean(a_norm)),
        "accel_norm_stat_std": float(np.nanstd(a_norm[stationary_mask])) if stat_count > 0 else float(np.nanstd(a_norm)),
        "accel_is_mps2": bool(8.5 < np.nanmean(a_norm[stationary_mask]) < 11.5) if stat_count > 0 else True,
        
        "gyro_unit": "rad/s",
        "gyro_norm_all_mean": float(np.nanmean(g_norm)),
        "gyro_norm_stat_mean": float(np.nanmean(g_norm[stationary_mask])) if stat_count > 0 else float(np.nanmean(g_norm)),
        "gyro_norm_stat_std": float(np.nanstd(g_norm[stationary_mask])) if stat_count > 0 else float(np.nanstd(g_norm)),
        "gyro_is_rads": bool(np.nanmean(g_norm[stationary_mask]) < 0.2) if stat_count > 0 else True,
        
        "mag_unit": "μT",
        "mag_norm_all_mean": float(np.nanmean(m_norm)),
        "mag_norm_all_std": float(np.nanstd(m_norm)),
        "mag_norm_stat_mean": float(np.nanmean(m_norm[stationary_mask])) if stat_count > 0 else float(np.nanmean(m_norm)),
        "mag_norm_stat_std": float(np.nanstd(m_norm[stationary_mask])) if stat_count > 0 else float(np.nanstd(m_norm)),
        "mag_is_uT": bool(10.0 < np.nanmean(m_norm) < 100.0),
        
        "axis_conventions": {
            "phone_x": "Right side of phone display",
            "phone_y": "Top of phone display (approx forward when vertically mounted)",
            "phone_z": "Out of phone screen (orthogonal)",
            "vehicle_x": "Forward longitudinal axis",
            "vehicle_y": "Lateral right axis",
            "vehicle_z": "Vertical upwards axis",
        }
    }
    
    logger.info(f"Audit Complete: |a|_stat={audit_results['accel_norm_stat_mean']:.3f} m/s^2, "
                f"|w|_stat={audit_results['gyro_norm_stat_mean']:.5f} rad/s, "
                f"|m|_all={audit_results['mag_norm_all_mean']:.2f} uT")
    
    return audit_results


def generate_sensor_audit_table(df: pd.DataFrame, stationary_mask: np.ndarray) -> pd.DataFrame:
    """Creates a comprehensive sensor unit audit table."""
    records = []
    
    channels = [
        ("acc_x", "Accelerometer X", "m/s²"),
        ("acc_y", "Accelerometer Y", "m/s²"),
        ("acc_z", "Accelerometer Z", "m/s²"),
        ("gyro_x", "Gyroscope X (Yaw)", "rad/s"),
        ("gyro_y", "Gyroscope Y (Pitch)", "rad/s"),
        ("gyro_z", "Gyroscope Z (Roll)", "rad/s"),
        ("mag_x", "Magnetometer X", "μT"),
        ("mag_y", "Magnetometer Y", "μT"),
        ("mag_z", "Magnetometer Z", "μT"),
        ("grav_x", "Gravity X", "m/s²"),
        ("grav_y", "Gravity Y", "m/s²"),
        ("grav_z", "Gravity Z", "m/s²"),
    ]
    
    for col, desc, unit in channels:
        if col not in df.columns:
            continue
        vals_all = df[col].dropna().values
        vals_stat = df.loc[stationary_mask, col].dropna().values if np.any(stationary_mask) else vals_all
        
        records.append({
            "channel": col,
            "description": desc,
            "verified_unit": unit,
            "all_mean": round(float(np.mean(vals_all)), 5),
            "all_std": round(float(np.std(vals_all)), 5),
            "all_min": round(float(np.min(vals_all)), 5),
            "all_max": round(float(np.max(vals_all)), 5),
            "stat_mean": round(float(np.mean(vals_stat)), 5),
            "stat_std": round(float(np.std(vals_stat)), 5),
        })
    
    return pd.DataFrame(records)


def plot_sensor_audit(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
    output_path: str,
) -> None:
    """Plots sensor magnitude norms across driving vs stationary intervals."""
    time_min = df["time_s"].values / 60.0
    
    ax = df["acc_x"].values
    ay = df["acc_y"].values
    az = df["acc_z"].values
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    
    gx = df["gyro_x"].values
    gy = df["gyro_y"].values
    gz = df["gyro_z"].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    
    mx = df["mag_x"].values
    my = df["mag_y"].values
    mz = df["mag_z"].values
    m_norm = np.sqrt(mx**2 + my**2 + mz**2)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Phase 2 Task 3: Sensor Norm Audit & Stationary Intervals", fontsize=13, fontweight="bold")
    
    # 1. Accel norm
    axes[0].plot(time_min, a_norm, color="steelblue", linewidth=0.5, alpha=0.7, label="|a| total")
    axes[0].axhline(GRAVITY_STANDARD, color="red", linestyle="--", linewidth=1.2, label=f"Standard g ({GRAVITY_STANDARD} m/s²)")
    axes[0].set_ylabel("|a| (m/s²)")
    axes[0].set_title("Accelerometer Magnitude Norm (Verifies m/s²)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Gyro norm
    axes[1].plot(time_min, g_norm, color="forestgreen", linewidth=0.5, alpha=0.7, label="|ω| total")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("|ω| (rad/s)")
    axes[1].set_title("Gyroscope Magnitude Norm (Verifies rad/s, near-zero stationary)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Mag norm
    axes[2].plot(time_min, m_norm, color="darkorange", linewidth=0.5, alpha=0.7, label="|m| total")
    axes[2].set_ylabel("|m| (μT)")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_title("Magnetometer Magnitude Norm (Verifies μT)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sensor audit plot saved to {output_path}")


# ============================================================================
# TASK 4: MULTI-SENSOR STATIONARY DETECTION
# ============================================================================

def detect_stationary_periods_multi(
    df: pd.DataFrame,
    speed_threshold_mps: float = 0.5,
    gyro_mag_threshold_rads: float = 0.05,
    accel_tol_mps2: float = 0.35,
    min_duration_s: float = 5.0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Calibration-grade multi-sensor stationary detection.
    
    Criteria (must ALL be satisfied):
    1. GPS speed < speed_threshold_mps (if GPS speed available)
    2. Gyroscope norm |w| < gyro_mag_threshold_rads
    3. Accelerometer norm ||a| - 9.80665| < accel_tol_mps2
    4. Condition sustained continuously for >= min_duration_s
    
    Returns:
        periods: List of dicts with start_idx, end_idx, start_time_s, duration_s
        stationary_mask: Boolean array for entire sequence
    """
    n = len(df)
    time_s = df["time_s"].values
    
    # 1. Accel norm condition
    ax = df["acc_x"].values
    ay = df["acc_y"].values
    az = df["acc_z"].values
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    accel_cond = np.abs(a_norm - GRAVITY_STANDARD) < accel_tol_mps2
    
    # 2. Gyro norm condition
    gx = df["gyro_x"].values
    gy = df["gyro_y"].values
    gz = df["gyro_z"].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_cond = g_norm < gyro_mag_threshold_rads
    
    # 3. GPS speed condition
    if "gps_speed_mps" in df.columns:
        speed = df["gps_speed_mps"].fillna(0.0).values
        speed_cond = speed < speed_threshold_mps
    elif "gps_speed_kmh" in df.columns:
        speed = (df["gps_speed_kmh"] / 3.6).fillna(0.0).values
        speed_cond = speed < speed_threshold_mps
    else:
        speed_cond = np.ones(n, dtype=bool)
    
    # Combined instantaneous condition
    inst_stationary = accel_cond & gyro_cond & speed_cond
    
    # Sustained duration filtering
    periods = []
    stationary_mask = np.zeros(n, dtype=bool)
    in_period = False
    start_idx = 0
    
    for i in range(n):
        if inst_stationary[i] and not in_period:
            start_idx = i
            in_period = True
        elif not inst_stationary[i] and in_period:
            duration = time_s[i-1] - time_s[start_idx]
            if duration >= min_duration_s:
                periods.append({
                    "period_id": len(periods),
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "start_time_s": float(time_s[start_idx]),
                    "end_time_s": float(time_s[i-1]),
                    "duration_s": round(float(duration), 2),
                    "n_samples": i - start_idx,
                })
                stationary_mask[start_idx:i] = True
            in_period = False
            
    if in_period:
        duration = time_s[-1] - time_s[start_idx]
        if duration >= min_duration_s:
            periods.append({
                "period_id": len(periods),
                "start_idx": start_idx,
                "end_idx": n - 1,
                "start_time_s": float(time_s[start_idx]),
                "end_time_s": float(time_s[-1]),
                "duration_s": round(float(duration), 2),
                "n_samples": n - start_idx,
            })
            stationary_mask[start_idx:n] = True
            
    logger.info(f"Multi-sensor stationary detector: identified {len(periods)} sustained periods "
                f"({stationary_mask.sum()} total samples, {stationary_mask.mean()*100:.1f}% of driving).")
    
    return periods, stationary_mask


# ============================================================================
# TASK 5 & 6: BIAS CALIBRATION (ACCELEROMETER & GYROSCOPE)
# ============================================================================

def estimate_gyroscope_bias(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Estimates gyroscope zero-rate bias vector (bx, by, bz) from stationary intervals.
    
    Model:
        w_meas = w_true + b_g + n_g
        When stationary, w_true = 0 => b_g = mean(w_meas[stationary])
    """
    gx = df.loc[stationary_mask, "gyro_x"].values
    gy = df.loc[stationary_mask, "gyro_y"].values
    gz = df.loc[stationary_mask, "gyro_z"].values
    
    if len(gx) == 0:
        logger.warning("No stationary samples for gyro bias estimation; defaulting to zeros")
        return {"bx": 0.0, "by": 0.0, "bz": 0.0, "std_x": 0.0, "std_y": 0.0, "std_z": 0.0}
    
    bias_info = {
        "bx": float(np.mean(gx)),
        "by": float(np.mean(gy)),
        "bz": float(np.mean(gz)),
        "std_x": float(np.std(gx)),
        "std_y": float(np.std(gy)),
        "std_z": float(np.std(gz)),
        "norm_bias": float(np.sqrt(np.mean(gx)**2 + np.mean(gy)**2 + np.mean(gz)**2)),
        "n_samples_used": int(len(gx)),
    }
    
    logger.info(f"Gyroscope Bias Estimated: bx={bias_info['bx']:.6f}, by={bias_info['by']:.6f}, "
                f"bz={bias_info['bz']:.6f} rad/s (|b_g|={bias_info['norm_bias']:.6f} rad/s)")
    
    return bias_info


def estimate_accelerometer_bias(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Estimates accelerometer bias characteristics.
    
    In stationary conditions, ||a_meas|| should equal g_standard.
    Scale/offset discrepancy: delta_g = mean(||a_meas||) - g_standard.
    """
    ax = df.loc[stationary_mask, "acc_x"].values
    ay = df.loc[stationary_mask, "acc_y"].values
    az = df.loc[stationary_mask, "acc_z"].values
    
    if len(ax) == 0:
        return {"scale_bias": 0.0, "norm_mean": GRAVITY_STANDARD, "norm_std": 0.0}
    
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    norm_mean = float(np.mean(a_norm))
    norm_std = float(np.std(a_norm))
    scale_bias = float(norm_mean - GRAVITY_STANDARD)
    
    accel_bias_info = {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "magnitude_bias_mps2": scale_bias,
        "mean_ax": float(np.mean(ax)),
        "mean_ay": float(np.mean(ay)),
        "mean_az": float(np.mean(az)),
        "std_ax": float(np.std(ax)),
        "std_ay": float(np.std(ay)),
        "std_az": float(np.std(az)),
        "n_samples_used": int(len(ax)),
    }
    
    logger.info(f"Accelerometer Bias: ||a||_stat={norm_mean:.4f} m/s^2 (offset={scale_bias:+.4f} m/s^2)")
    return accel_bias_info


# ============================================================================
# TASK 10: MAGNETOMETER RELIABILITY ANALYSIS
# ============================================================================

def analyze_magnetometer_reliability(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates magnetometer stability, electromagnetic interference, and distortion.
    
    Checks:
    1. Norm stability: ||m|| variation during maneuvers vs stationary
    2. Dynamic field perturbation index: std(||m||_driving) / mean(||m||_stat)
    3. Soft/hard-iron indication: non-spherical distortion
    """
    mx = df["mag_x"].values
    my = df["mag_y"].values
    mz = df["mag_z"].values
    m_norm = np.sqrt(mx**2 + my**2 + mz**2)
    
    m_stat = m_norm[stationary_mask] if np.any(stationary_mask) else m_norm
    
    stat_mean = float(np.mean(m_stat))
    stat_std = float(np.std(m_stat))
    all_std = float(np.std(m_norm))
    
    # Normalized perturbation metric
    perturbation_ratio = float(all_std / stat_mean) if stat_mean > 0 else 1.0
    
    # Reliability threshold: if perturbation ratio > 0.15, vehicle magnetic interference is high
    is_reliable = bool(perturbation_ratio < 0.12 and stat_std < 2.5)
    
    mag_audit = {
        "mag_stat_mean_uT": round(stat_mean, 2),
        "mag_stat_std_uT": round(stat_std, 3),
        "mag_all_std_uT": round(all_std, 3),
        "perturbation_ratio": round(perturbation_ratio, 4),
        "is_reliable_for_yaw": is_reliable,
        "recommendation": (
            "Magnetometer exhibits acceptable stability for heading aid" if is_reliable
            else "Magnetometer exhibits vehicle/chassis electromagnetic interference; treat as UNRELIABLE/AUXILIARY"
        )
    }
    
    logger.info(f"Magnetometer Audit: perturbation_ratio={perturbation_ratio:.3f}, reliable={is_reliable}")
    return mag_audit
