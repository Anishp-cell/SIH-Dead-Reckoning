"""
Phase 7 Map Matching & Trajectory Evaluation Metrics:
Computes spatial, heading, matching, filtering, and stability metrics compliant
with Section 37 of the Phase 7 specification.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from Data_details.src.phase5.evaluation.trajectory_metrics import compute_navigation_metrics
from Data_details.src.phase7.map.geometry import wrap_angle_rad


def compute_phase7_metrics(
    p_est: np.ndarray,                  # (N, 3) Estimated positions [East, North, Up]
    p_ref: np.ndarray,                  # (N, 3) Ground truth positions [East, North, Up]
    v_est: np.ndarray,                  # (N, 3) Estimated velocity [East, North, Up]
    v_ref: np.ndarray,                  # (N, 3) Ground truth velocity
    yaw_est_rad: np.ndarray,            # (N,) Estimated Cartesian ENU yaw in rad
    yaw_ref_rad: np.ndarray,            # (N,) Ground truth Cartesian ENU yaw in rad
    traveled_distance_m: float,         # Total traveled distance
    matched_segment_ids: List[Optional[str]], # List of matched segment IDs
    map_confidences: np.ndarray,        # (N,) Map confidence scores
    candidate_counts: np.ndarray,       # (N,) Candidate counts
    cross_track_errors: np.ndarray,     # (N,) Cross-track residuals
    map_ct_accepted: List[bool],        # (N,) Whether cross-track update applied
    map_hd_accepted: List[bool],        # (N,) Whether heading update applied
    ct_nis_list: List[float],           # (N,) Cross-track NIS
    hd_nis_list: List[float],           # (N,) Heading NIS
) -> Dict[str, Any]:
    """
    Computes complete Phase 7 evaluation metric dictionary.
    """
    # 1. Base navigation metrics (endpoint, position RMSE, yaw RMSE, drift %)
    base_m = compute_navigation_metrics(
        p_est, p_ref, v_est, v_ref, yaw_est_rad, yaw_ref_rad, traveled_distance_m
    )

    n = len(p_est)
    # 2. Decompose position error into Cross-track and Along-track relative to reference trajectory
    pos_err = p_est[:, :2] - p_ref[:, :2]
    # Unit tangent of reference path
    ref_tangents = np.zeros((n, 2))
    ref_tangents[:, 0] = np.cos(yaw_ref_rad)
    ref_tangents[:, 1] = np.sin(yaw_ref_rad)
    # Unit normal (left)
    ref_normals = np.zeros((n, 2))
    ref_normals[:, 0] = -np.sin(yaw_ref_rad)
    ref_normals[:, 1] = np.cos(yaw_ref_rad)

    # Along-track error: projection onto reference tangent
    along_track_err = np.sum(pos_err * ref_tangents, axis=1)
    along_track_rmse = float(np.sqrt(np.mean(along_track_err**2)))

    # Cross-track error: projection onto reference normal
    cross_track_err = np.sum(pos_err * ref_normals, axis=1)
    cross_track_rmse = float(np.sqrt(np.mean(cross_track_err**2)))

    # 3. Heading metrics
    yaw_errors_rad = np.array([wrap_angle_rad(y_e - y_r) for y_e, y_r in zip(yaw_est_rad, yaw_ref_rad)])
    yaw_errors_deg = np.degrees(np.abs(yaw_errors_rad))
    final_yaw_error_deg = float(yaw_errors_deg[-1])

    # 4. Map Matching stability metrics
    switches = 0
    valid_matches = 0
    prev_id = None
    for sid in matched_segment_ids:
        if sid is not None:
            valid_matches += 1
            if prev_id is not None and sid != prev_id:
                switches += 1
            prev_id = sid

    match_coverage_pct = (valid_matches / max(1, n)) * 100.0
    mean_candidates = float(np.mean(candidate_counts)) if len(candidate_counts) > 0 else 0.0
    mean_confidence = float(np.mean(map_confidences)) if len(map_confidences) > 0 else 0.0

    # 5. Filtering & Gating statistics
    ct_acc_count = sum(1 for a in map_ct_accepted if a)
    hd_acc_count = sum(1 for a in map_hd_accepted if a)
    ct_acc_pct = (ct_acc_count / max(1, n)) * 100.0
    hd_acc_pct = (hd_acc_count / max(1, n)) * 100.0

    valid_ct_nis = [x for x, a in zip(ct_nis_list, map_ct_accepted) if a and np.isfinite(x)]
    valid_hd_nis = [x for x, a in zip(hd_nis_list, map_hd_accepted) if a and np.isfinite(x)]

    mean_ct_nis = float(np.mean(valid_ct_nis)) if valid_ct_nis else 0.0
    mean_hd_nis = float(np.mean(valid_hd_nis)) if valid_hd_nis else 0.0

    return {
        # Navigation
        "endpoint_error_m": round(base_m["endpoint_error_m"], 2),
        "position_rmse_m": round(base_m.get("pos_rmse_m", base_m.get("position_rmse_m", 0.0)), 2),
        "cross_track_rmse_m": round(cross_track_rmse, 2),
        "along_track_rmse_m": round(along_track_rmse, 2),
        "drift_pct": round(base_m["drift_pct"], 2),
        "yaw_rmse_deg": round(base_m["yaw_rmse_deg"], 2),
        "final_yaw_error_deg": round(final_yaw_error_deg, 2),
        # Map Matching
        "match_coverage_pct": round(match_coverage_pct, 1),
        "road_switch_count": int(switches),
        "mean_candidate_count": round(mean_candidates, 1),
        "mean_map_confidence": round(mean_confidence, 3),
        "map_ct_accepted_pct": round(ct_acc_pct, 1),
        "map_hd_accepted_pct": round(hd_acc_pct, 1),
        "mean_ct_nis": round(mean_ct_nis, 2),
        "mean_hd_nis": round(mean_hd_nis, 2),
    }
