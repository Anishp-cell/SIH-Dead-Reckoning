"""
Phase 6 Physical Constraint & Diagnostics Metrics.
Computes lateral/vertical body velocity errors, stationary position creep, and residual statistics.
"""

from typing import Dict, Any, List
import numpy as np


def compute_constraint_metrics(
    v_b_traj: np.ndarray,
    p_traj: np.ndarray,
    is_stationary_arr: np.ndarray,
    nhc_nu_list: List[np.ndarray],
    nhc_accepted_list: List[bool],
    zupt_accepted_list: List[bool],
    zaru_accepted_list: List[bool],
    ba_traj: np.ndarray,
    bg_traj: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes specialized Phase 6 physical constraint metrics.
    
    Parameters:
        v_b_traj: (N, 3) Estimated body velocity [v_fwd, v_lat, v_up]
        p_traj: (N, 3) Estimated ENU position
        is_stationary_arr: (N,) Boolean array of stationary status
        nhc_nu_list: List of (2,) NHC innovation residuals
        nhc_accepted_list: List of boolean NHC gating results
        zupt_accepted_list: List of boolean ZUPT gating results
        zaru_accepted_list: List of boolean ZARU gating results
        ba_traj: (N, 3) Estimated accel bias trajectory
        bg_traj: (N, 3) Estimated gyro bias trajectory
    """
    v_lat = v_b_traj[:, 1]
    v_up = v_b_traj[:, 2]

    lat_rmse = float(np.sqrt(np.mean(v_lat ** 2)))
    lat_mae = float(np.mean(np.abs(v_lat)))
    lat_max = float(np.max(np.abs(v_lat)))

    up_rmse = float(np.sqrt(np.mean(v_up ** 2)))
    up_mae = float(np.mean(np.abs(v_up)))
    up_max = float(np.max(np.abs(v_up)))

    # Stationary position creep: evaluate each continuous stationary segment independently
    segment_creeps = []
    in_segment = False
    seg_start_idx = 0

    for idx, is_stat in enumerate(is_stationary_arr):
        if is_stat and not in_segment:
            in_segment = True
            seg_start_idx = idx
        elif not is_stat and in_segment:
            in_segment = False
            # Evaluate creep over [seg_start_idx, idx)
            p_seg = p_traj[seg_start_idx:idx]
            if len(p_seg) > 1:
                disp = np.linalg.norm(p_seg - p_seg[0], axis=1)
                segment_creeps.append(float(np.max(disp)))

    # Handle case where trajectory ends while stationary
    if in_segment:
        p_seg = p_traj[seg_start_idx:]
        if len(p_seg) > 1:
            disp = np.linalg.norm(p_seg - p_seg[0], axis=1)
            segment_creeps.append(float(np.max(disp)))

    num_stationary_segments = len(segment_creeps)
    max_segment_creep_m = float(np.max(segment_creeps)) if num_stationary_segments > 0 else 0.0
    mean_segment_creep_m = float(np.mean(segment_creeps)) if num_stationary_segments > 0 else 0.0
    median_segment_creep_m = float(np.median(segment_creeps)) if num_stationary_segments > 0 else 0.0

    # Retain stationary_creep_m as max_segment_creep_m for backwards compatibility
    stationary_creep_m = max_segment_creep_m

    # NHC residual RMS
    nhc_nu_arr = np.array(nhc_nu_list) if len(nhc_nu_list) > 0 else np.zeros((1, 2))
    nhc_res_rms_lat = float(np.sqrt(np.mean(nhc_nu_arr[:, 0] ** 2)))
    nhc_res_rms_up = float(np.sqrt(np.mean(nhc_nu_arr[:, 1] ** 2)))

    nhc_accept_rate = float(np.mean(nhc_accepted_list) * 100.0) if len(nhc_accepted_list) > 0 else 0.0
    zupt_count = int(np.sum(zupt_accepted_list))
    zaru_count = int(np.sum(zaru_accepted_list))

    # Sensor bias stability (standard deviation over trajectory)
    ba_std = float(np.mean(np.std(ba_traj, axis=0)))
    bg_std = float(np.mean(np.std(bg_traj, axis=0)))

    return {
        "lateral_vel_rmse_mps": round(lat_rmse, 4),
        "lateral_vel_mae_mps": round(lat_mae, 4),
        "lateral_vel_max_mps": round(lat_max, 4),
        "vertical_vel_rmse_mps": round(up_rmse, 4),
        "vertical_vel_mae_mps": round(up_mae, 4),
        "vertical_vel_max_mps": round(up_max, 4),
        "stationary_creep_m": round(stationary_creep_m, 2),
        "max_segment_creep_m": round(max_segment_creep_m, 2),
        "mean_segment_creep_m": round(mean_segment_creep_m, 2),
        "median_segment_creep_m": round(median_segment_creep_m, 2),
        "num_stationary_segments": num_stationary_segments,
        "nhc_residual_lat_rms": round(nhc_res_rms_lat, 4),
        "nhc_residual_up_rms": round(nhc_res_rms_up, 4),
        "nhc_acceptance_pct": round(nhc_accept_rate, 1),
        "zupt_accepted_count": zupt_count,
        "zaru_accepted_count": zaru_count,
        "accel_bias_std_mps2": round(ba_std, 5),
        "gyro_bias_std_radps": round(bg_std, 6),
    }
