"""
Phase 8 Evaluation: Trajectory Error Decomposition and Recovery Continuity Metrics.
Computes along-track, cross-track, horizontal positional errors, outage drift rates,
and verifies zero-teleportation recovery bounds.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def compute_road_aligned_errors(
    est_pos_2d: np.ndarray,      # (N, 2) [East, North]
    ref_pos_2d: np.ndarray,      # (N, 2) [East, North]
    ref_yaw_enu_rad: np.ndarray, # (N,) heading in Cartesian ENU rad
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes 2D horizontal positional error delta_p = p_est - p_ref
    into along-track (tangential) and cross-track (orthogonal) components.
    """
    diff = est_pos_2d - ref_pos_2d  # (N, 2)
    horiz_error = np.linalg.norm(diff, axis=1)

    cos_yaw = np.cos(ref_yaw_enu_rad)
    sin_yaw = np.sin(ref_yaw_enu_rad)

    # Tangent vector t = [cos_yaw, sin_yaw]
    # Normal vector n = [-sin_yaw, cos_yaw]
    along_track = diff[:, 0] * cos_yaw + diff[:, 1] * sin_yaw
    cross_track = -diff[:, 0] * sin_yaw + diff[:, 1] * cos_yaw

    return along_track, cross_track, horiz_error


def compute_benchmark_metrics(
    time_s: np.ndarray,
    est_pos_2d: np.ndarray,
    ref_pos_2d: np.ndarray,
    ref_yaw_enu_rad: np.ndarray,
    outage_start: Optional[float] = None,
    outage_end: Optional[float] = None,
) -> Dict[str, float]:
    """
    Computes summary benchmark metrics for a trajectory.
    """
    along, cross, horiz = compute_road_aligned_errors(est_pos_2d, ref_pos_2d, ref_yaw_enu_rad)

    metrics = {
        "max_error_m": float(np.max(horiz)),
        "mean_error_m": float(np.mean(horiz)),
        "rmse_error_m": float(np.sqrt(np.mean(horiz ** 2))),
        "max_along_track_m": float(np.max(np.abs(along))),
        "max_cross_track_m": float(np.max(np.abs(cross))),
        "final_error_m": float(horiz[-1]),
    }

    # If outage interval specified, compute drift percentage during outage
    if outage_start is not None and outage_end is not None:
        mask_outage = (time_s >= outage_start) & (time_s <= outage_end)
        if np.any(mask_outage):
            idx_start = np.where(mask_outage)[0][0]
            idx_end = np.where(mask_outage)[0][-1]

            e_start = horiz[idx_start]
            e_end = horiz[idx_end]

            # Cumulative distance traveled along reference trajectory during outage
            diffs = np.diff(ref_pos_2d[idx_start:idx_end+1], axis=0)
            d_traveled = float(np.sum(np.linalg.norm(diffs, axis=1)))

            drift_pct = ((e_end - e_start) / max(1.0, d_traveled)) * 100.0
            metrics["outage_d_traveled_m"] = d_traveled
            metrics["outage_drift_pct"] = float(drift_pct)
            metrics["outage_max_error_m"] = float(np.max(horiz[mask_outage]))
            metrics["outage_end_error_m"] = float(e_end)

    return metrics


def evaluate_recovery_continuity(
    time_s: np.ndarray,
    est_pos_2d: np.ndarray,
    est_vel_2d: np.ndarray,
    est_yaw_rad: np.ndarray,
    recovery_time_s: float,
    window_sec: float = 1.0,
) -> Dict[str, Any]:
    """
    Evaluates step continuity at the moment GNSS recovers from an outage.
    """
    idx_rec = np.argmin(np.abs(time_s - recovery_time_s))

    if idx_rec == 0 or idx_rec >= len(time_s) - 1:
        return {
            "p_step_m": 0.0,
            "v_step_mps": 0.0,
            "yaw_step_deg": 0.0,
            "pseudo_accel_mps2": 0.0,
            "passed": True,
        }

    dt = max(1e-3, time_s[idx_rec] - time_s[idx_rec - 1])
    dp = float(np.linalg.norm(est_pos_2d[idx_rec] - est_pos_2d[idx_rec - 1]))
    dv = float(np.linalg.norm(est_vel_2d[idx_rec] - est_vel_2d[idx_rec - 1]))

    dyaw = np.arctan2(
        np.sin(est_yaw_rad[idx_rec] - est_yaw_rad[idx_rec - 1]),
        np.cos(est_yaw_rad[idx_rec] - est_yaw_rad[idx_rec - 1]),
    )
    dyaw_deg = float(np.abs(np.degrees(dyaw)))
    pseudo_a = dv / dt

    passed = (dp <= 3.5 and dv <= 1.0 and dyaw_deg <= 3.0 and pseudo_a <= 2.5)

    return {
        "p_step_m": dp,
        "v_step_mps": dv,
        "yaw_step_deg": dyaw_deg,
        "pseudo_accel_mps2": pseudo_a,
        "passed": passed,
    }
