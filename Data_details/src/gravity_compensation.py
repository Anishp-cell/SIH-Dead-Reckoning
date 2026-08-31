"""
Phase 2 Task 15: Orientation-Aware World-Frame Gravity Compensation.
Rotates calibrated body acceleration into the Navigation (ENU) frame and removes
true Earth gravity vector [0, 0, g]^T to extract pure dynamic vehicle acceleration.
"""

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.attitude_estimation import quaternion_to_rotation_matrix

logger = logging.getLogger(__name__)

GRAVITY_STANDARD = 9.80665  # m/s^2


def compensate_gravity_world_frame(
    acc_body: np.ndarray,
    quaternions: np.ndarray,
    accel_bias: Optional[np.ndarray] = None,
    gravity_norm: float = GRAVITY_STANDARD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms accelerometer readings to the Navigation (ENU) frame and subtracts
    the vertical gravity vector.
    
    Equations:
        1. Correct bias: a_body_corr = a_body - b_a
        2. Rotate to Nav frame: a_nav[k] = R(q[k]) * a_body_corr[k]
        3. Remove gravity: a_dyn_nav[k] = a_nav[k] - [0, 0, g]^T
    
    Parameters:
        acc_body: shape (N, 3) array of [ax, ay, az] in phone body frame
        quaternions: shape (N, 4) array of unit quaternions [qw, qx, qy, qz]
        accel_bias: optional (3,) array of bias offsets [bx, by, bz]
        gravity_norm: gravitational acceleration magnitude (default 9.80665 m/s^2)
        
    Returns:
        a_dyn_nav: shape (N, 3) dynamic acceleration in ENU frame [a_east, a_north, a_up]
        a_nav_total: shape (N, 3) total acceleration including gravity in ENU frame
    """
    n = len(acc_body)
    a_dyn_nav = np.zeros((n, 3))
    a_nav_total = np.zeros((n, 3))
    
    g_vector_nav = np.array([0.0, 0.0, gravity_norm])
    
    if accel_bias is None:
        accel_bias = np.zeros(3)
        
    a_body_corr = acc_body - accel_bias
    
    for i in range(n):
        R_b2n = quaternion_to_rotation_matrix(quaternions[i])
        a_n = R_b2n @ a_body_corr[i]
        a_nav_total[i] = a_n
        a_dyn_nav[i] = a_n - g_vector_nav
        
    return a_dyn_nav, a_nav_total


def evaluate_gravity_compensation(
    df: pd.DataFrame,
    a_dyn_nav: np.ndarray,
    stationary_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates dynamic acceleration residuals before and after gravity compensation.
    
    When stationary:
    - True dynamic acceleration is identically 0.0 m/s^2
    - Residual norm ||a_dyn|| measures gravity leakage and sensor calibration error.
    """
    # 1. Phase 1 naive linear acceleration norm
    if "grav_x" in df.columns:
        ax_p1 = df["acc_x"] - df["grav_x"]
        ay_p1 = df["acc_y"] - df["grav_y"]
        az_p1 = df["acc_z"] - df["grav_z"]
        p1_norm = np.sqrt(ax_p1**2 + ay_p1**2 + az_p1**2).values
    else:
        p1_norm = np.abs(np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2).values - GRAVITY_STANDARD)
        
    # 2. Phase 2 world-frame dynamic acceleration norm
    p2_norm = np.sqrt(np.sum(a_dyn_nav**2, axis=1))
    
    # 3. Stationary residuals
    p1_stat_mean = float(np.mean(p1_norm[stationary_mask])) if np.any(stationary_mask) else float(np.mean(p1_norm))
    p1_stat_std  = float(np.std(p1_norm[stationary_mask])) if np.any(stationary_mask) else float(np.std(p1_norm))
    p2_stat_mean = float(np.mean(p2_norm[stationary_mask])) if np.any(stationary_mask) else float(np.mean(p2_norm))
    p2_stat_std  = float(np.std(p2_norm[stationary_mask])) if np.any(stationary_mask) else float(np.std(p2_norm))
    
    comp_metrics = {
        "p1_naive_stationary_residual_mps2": round(p1_stat_mean, 4),
        "p1_naive_stationary_std_mps2": round(p1_stat_std, 4),
        "p2_calibrated_stationary_residual_mps2": round(p2_stat_mean, 4),
        "p2_calibrated_stationary_std_mps2": round(p2_stat_std, 4),
        "residual_reduction_pct": round((p1_stat_mean - p2_stat_mean) / p1_stat_mean * 100, 2) if p1_stat_mean > 0 else 0.0,
    }
    
    logger.info(f"Gravity Compensation Residual: P1={p1_stat_mean:.4f} m/s^2 -> P2={p2_stat_mean:.4f} m/s^2 "
                f"({comp_metrics['residual_reduction_pct']}% reduction)")
    
    return comp_metrics


def plot_gravity_compensation(
    df: pd.DataFrame,
    a_dyn_nav: np.ndarray,
    output_path: str,
    sequence_name: str = "S1",
) -> None:
    """Plots before/after gravity compensation time-series and residuals."""
    time_min = df["time_s"].values / 60.0
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Phase 2 Task 15: World-Frame Gravity Compensation — Sequence {sequence_name}",
                 fontsize=13, fontweight="bold")
    
    # 1. Raw Phone Acceleration
    axes[0].plot(time_min, df["acc_x"], color="#e74c3c", linewidth=0.4, alpha=0.6, label="Acc X")
    axes[0].plot(time_min, df["acc_y"], color="#2ecc71", linewidth=0.4, alpha=0.6, label="Acc Y")
    axes[0].plot(time_min, df["acc_z"], color="#3498db", linewidth=0.4, alpha=0.6, label="Acc Z")
    axes[0].set_ylabel("Raw Accel (m/s²)")
    axes[0].set_title("Raw Accelerometer (Phone Body Frame — Dominant Gravity on Screen Z-Axis)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Dynamic World-Frame Acceleration (ENU)
    axes[1].plot(time_min, a_dyn_nav[:, 0], color="#e74c3c", linewidth=0.5, alpha=0.7, label="East Accel")
    axes[1].plot(time_min, a_dyn_nav[:, 1], color="#2ecc71", linewidth=0.5, alpha=0.7, label="North Accel")
    axes[1].plot(time_min, a_dyn_nav[:, 2], color="#3498db", linewidth=0.4, alpha=0.5, label="Up Accel (Dynamic)")
    axes[1].set_ylabel("Dyn Accel (m/s²)")
    axes[1].set_title("Calibrated Dynamic Acceleration in World Frame (ENU — Gravity Removed)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Dynamic Acceleration Norm vs Speed
    a_dyn_norm = np.sqrt(np.sum(a_dyn_nav[:, :2]**2, axis=1))
    axes[2].plot(time_min, a_dyn_norm, color="#8e44ad", linewidth=0.5, alpha=0.8, label="Horizontal Dyn Accel Norm")
    if "gps_speed_kmh" in df.columns:
        ax_twin = axes[2].twinx()
        ax_twin.plot(time_min, df["gps_speed_kmh"], color="#f39c12", linewidth=0.5, alpha=0.5, label="GPS Speed (km/h)")
        ax_twin.set_ylabel("GPS Speed (km/h)", color="#f39c12")
    axes[2].set_ylabel("Horiz Accel (m/s²)")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_title("Horizontal Dynamic Acceleration Magnitude vs Vehicle Speed")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gravity compensation plot saved to {output_path}")
