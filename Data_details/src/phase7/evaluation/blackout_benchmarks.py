"""
Phase 7 Blackout Benchmarks Runner:
Executes sequential Phase 7 navigation engine over telemetry slices across outage durations.
Strictly causal, sample-by-sample, zero future buffering.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from ..map.map_database import RoadMapDatabase
from ..map.spatial_index import SpatialIndex
from ..matching.candidate_generator import CandidateGenerator
from ..matching.candidate_scoring import CandidateScorer, ScoringWeights
from ..matching.topology import RoadTopologyGraph, TransitionModel
from ..matching.temporal_matcher import TemporalMapMatcher
from ..matching.map_measurement import MapMeasurementModel, MapMeasurementConfig
from ..streaming import Phase7StreamingEngine
from .map_matching_metrics import compute_phase7_metrics


def run_phase7_trajectory(
    df_slice: pd.DataFrame,
    init_p_enu: np.ndarray,
    init_v_enu: np.ndarray,
    init_q: np.ndarray,
    road_db: RoadMapDatabase,
    spatial_index: SpatialIndex,
    ai_engine: Optional[CausalStreamingInferenceEngine] = None,
    enable_ai_speed: bool = True,
    enable_nhc: bool = True,
    enable_zupt: bool = True,
    enable_zaru: bool = True,
    enable_map_matching: bool = True,
    enable_map_updates: bool = True,
    enable_cross_track_update: bool = True,
    enable_heading_update: bool = True,
    scoring_weights: Optional[ScoringWeights] = None,
    min_confidence_thresh: float = 0.35,
    hard_snapping: bool = False,  # For M1 ablation
    warm_start_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Executes sequential Phase 7 navigation engine over a given telemetry slice.
    """
    # Build matching stack
    if enable_map_matching:
        generator = CandidateGenerator(spatial_index, k_sigma=3.0, d_min_m=20.0)
        scorer = CandidateScorer(weights=scoring_weights)
        topo_graph = RoadTopologyGraph(road_db)
        trans_model = TransitionModel(topo_graph)
        matcher = TemporalMapMatcher(
            generator, scorer, trans_model, min_confidence_threshold=min_confidence_thresh
        )
    else:
        matcher = None

    map_model = MapMeasurementModel()
    eskf = Phase6ESKF(
        enable_nhc=enable_nhc,
        enable_zupt=enable_zupt,
        enable_zaru=enable_zaru,
        enable_disturbance_adaptive=True,
    )

    engine = Phase7StreamingEngine(
        eskf=eskf,
        ai_engine=ai_engine,
        map_matcher=matcher,
        map_model=map_model,
        enable_map_updates=enable_map_updates,
        enable_cross_track_update=enable_cross_track_update,
        enable_heading_update=enable_heading_update,
    )

    engine.initialize(
        p0=init_p_enu,
        v0=init_v_enu,
        q0=init_q,
        initial_timestamp=float(df_slice["time_s"].iloc[0]),
    )

    n = len(df_slice)
    p_traj = np.zeros((n, 3))
    v_traj = np.zeros((n, 3))
    q_traj = np.zeros((n, 4))
    v_b_traj = np.zeros((n, 3))
    yaw_traj = np.zeros(n)
    p_map_traj = np.zeros((n, 2))
    matched_ids: List[Optional[str]] = []
    map_conf_arr = np.zeros(n)
    cand_counts = np.zeros(n)
    ct_errors = np.zeros(n)
    ct_accepted_list: List[bool] = []
    hd_accepted_list: List[bool] = []
    ct_nis_list: List[float] = []
    hd_nis_list: List[float] = []
    cov_diags = np.zeros((n, 15))

    # Prefer raw unfiltered vehicle-frame signals (matching Phase 4 neural network training)
    feature_cols = [
        "acc_fwd_veh", "acc_lat_veh", "acc_up_veh",
        "gyro_roll_veh", "gyro_pitch_veh", "gyro_yaw_veh",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    for idx_c, col_name in enumerate(feature_cols):
        if col_name not in df_slice.columns:
            fallback = col_name + "_filtered"
            if fallback in df_slice.columns:
                feature_cols[idx_c] = fallback

    # Causal warm-start buffer pre-filling (t < t_0)
    if ai_engine is not None and warm_start_df is not None and len(warm_start_df) > 0:
        warm_matrix = warm_start_df[feature_cols].values.astype(np.float32)
        ai_engine.warm_up(warm_matrix)

    features_matrix = df_slice[feature_cols].values.astype(np.float32)
    time_arr = df_slice["time_s"].values.astype(float)
    dt_arr = df_slice["dt"].values.astype(float)

    for i in range(n):
        sample = features_matrix[i]
        t = time_arr[i]
        dt = dt_arr[i]

        est = engine.step(
            sample,
            timestamp=t,
            dt_override=dt,
            enable_ai_speed=enable_ai_speed,
            enable_nhc=enable_nhc,
            enable_zupt=enable_zupt,
            enable_zaru=enable_zaru,
        )

        # M1 Ablation: Hard snapping test override
        if hard_snapping and est.matched_segment_id is not None and not np.isnan(est.p_map_matched[0]):
            p_pos = np.array([est.p_map_matched[0], est.p_map_matched[1], est.p_eskf[2]])
            engine.eskf.state.p = p_pos.copy()
            p_traj[i] = p_pos
        else:
            p_traj[i] = est.p_eskf

        v_traj[i] = est.v_eskf
        q_traj[i] = est.q_eskf
        v_b_traj[i] = est.v_b
        yaw_traj[i] = est.yaw_eskf_rad
        p_map_traj[i] = est.p_map_matched
        matched_ids.append(est.matched_segment_id)
        map_conf_arr[i] = est.map_confidence
        cand_counts[i] = est.candidate_count
        ct_errors[i] = est.cross_track_error_m
        ct_accepted_list.append(est.map_cross_track_accepted)
        hd_accepted_list.append(est.map_heading_accepted)
        ct_nis_list.append(est.cross_track_nis)
        hd_nis_list.append(est.heading_nis)
        cov_diags[i] = est.cov_diagonal

    return {
        "p_est": p_traj,
        "v_est": v_traj,
        "q_est": q_traj,
        "v_b_est": v_b_traj,
        "yaw_est": yaw_traj,
        "p_map_est": p_map_traj,
        "matched_ids": matched_ids,
        "map_confidences": map_conf_arr,
        "candidate_counts": cand_counts,
        "cross_track_errors": ct_errors,
        "map_ct_accepted": ct_accepted_list,
        "map_hd_accepted": hd_accepted_list,
        "ct_nis_list": ct_nis_list,
        "hd_nis_list": hd_nis_list,
        "cov_diags": cov_diags,
        "time": time_arr,
    }
