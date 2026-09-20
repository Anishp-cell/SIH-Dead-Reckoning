"""
Phase 5 Trajectory & Inertial Navigation Evaluation Metrics.
"""

from typing import Dict, Any
import numpy as np


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wraps angles to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_navigation_metrics(
    p_est: np.ndarray,
    p_ref: np.ndarray,
    v_est: np.ndarray,
    v_ref: np.ndarray,
    yaw_est_rad: np.ndarray,
    yaw_ref_rad: np.ndarray,
    traveled_distance_m: float,
) -> Dict[str, float]:
    """
    Computes rigorous trajectory error metrics for position, velocity, and attitude.
    """
    # 2D horizontal position error
    de = p_est[:, 0] - p_ref[:, 0]
    dn = p_est[:, 1] - p_ref[:, 1]
    pos_err_2d = np.sqrt(de ** 2 + dn ** 2)

    pos_rmse_m = float(np.sqrt(np.mean(pos_err_2d ** 2)))
    pos_mae_m = float(np.mean(pos_err_2d))
    pos_max_m = float(np.max(pos_err_2d))
    pos_p95_m = float(np.percentile(pos_err_2d, 95))
    endpoint_error_m = float(pos_err_2d[-1])

    # Drift percentage relative to actual traveled distance
    if traveled_distance_m > 1.0:
        drift_pct = float((endpoint_error_m / traveled_distance_m) * 100.0)
    else:
        drift_pct = 0.0

    # Velocity error
    v_diff = v_est - v_ref
    vel_err_norm = np.sqrt(np.sum(v_diff ** 2, axis=1))
    vel_rmse_mps = float(np.sqrt(np.mean(vel_err_norm ** 2)))
    vel_mae_mps = float(np.mean(vel_err_norm))

    # Heading error
    yaw_err = np.abs(wrap_angle_rad(yaw_est_rad - yaw_ref_rad))
    yaw_rmse_deg = float(np.degrees(np.sqrt(np.mean(yaw_err ** 2))))
    yaw_mae_deg = float(np.degrees(np.mean(yaw_err)))
    final_yaw_err_deg = float(np.degrees(yaw_err[-1]))

    return {
        "traveled_distance_m": round(traveled_distance_m, 2),
        "endpoint_error_m": round(endpoint_error_m, 2),
        "drift_pct": round(drift_pct, 2),
        "pos_rmse_m": round(pos_rmse_m, 2),
        "pos_mae_m": round(pos_mae_m, 2),
        "pos_max_m": round(pos_max_m, 2),
        "pos_p95_m": round(pos_p95_m, 2),
        "vel_rmse_mps": round(vel_rmse_mps, 4),
        "vel_mae_mps": round(vel_mae_mps, 4),
        "yaw_rmse_deg": round(yaw_rmse_deg, 2),
        "yaw_mae_deg": round(yaw_mae_deg, 2),
        "final_yaw_err_deg": round(final_yaw_err_deg, 2),
    }
