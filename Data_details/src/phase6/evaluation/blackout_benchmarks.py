"""
Phase 6 Blackout Benchmarks & Ablation Runner:
Evaluates E2 (Phase 5 Full) vs E3 (+NHC) vs E4 (+ZUPT) vs E5 (+ZARU) vs E6 (Full Phase 6).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase5.evaluation.trajectory_metrics import compute_navigation_metrics
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase6.streaming import Phase6StreamingEngine
from Data_details.src.phase6.evaluation.constraint_metrics import compute_constraint_metrics
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine


def run_phase6_trajectory(
    df_slice: pd.DataFrame,
    init_p_enu: np.ndarray,
    init_v_enu: np.ndarray,
    init_q: np.ndarray,
    ai_engine: Optional[CausalStreamingInferenceEngine] = None,
    enable_ai_speed: bool = True,
    enable_nhc: bool = True,
    enable_zupt: bool = True,
    enable_zaru: bool = True,
    enable_disturbance_adaptive: bool = True,
    fixed_sigma_override: Optional[float] = None,
    warm_start_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Executes sequential Phase 6 navigation engine over a given telemetry slice.
    """
    eskf = Phase6ESKF(
        enable_nhc=enable_nhc,
        enable_zupt=enable_zupt,
        enable_zaru=enable_zaru,
        enable_disturbance_adaptive=enable_disturbance_adaptive,
    )
    engine = Phase6StreamingEngine(eskf=eskf, ai_engine=ai_engine)
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
    ba_traj = np.zeros((n, 3))
    bg_traj = np.zeros((n, 3))
    cov_diags = np.zeros((n, 15))

    nhc_nu_list = []
    nhc_accepted_list = []
    zupt_accepted_list = []
    zaru_accepted_list = []
    is_stat_list = []
    nhc_conf_list = []

    # Prefer raw unfiltered vehicle-frame signals (matching Phase 4 neural network training)
    feature_cols = [
        "acc_fwd_veh", "acc_lat_veh", "acc_up_veh",
        "gyro_roll_veh", "gyro_pitch_veh", "gyro_yaw_veh",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    for i, c in enumerate(feature_cols):
        if c not in df_slice.columns:
            fallback = c + "_filtered"
            if fallback in df_slice.columns:
                feature_cols[i] = fallback

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
            fixed_sigma_override=fixed_sigma_override,
        )

        p_traj[i] = est.p
        v_traj[i] = est.v
        q_traj[i] = est.q
        v_b_traj[i] = est.v_b
        ba_traj[i] = est.ba
        bg_traj[i] = est.bg
        cov_diags[i] = est.cov_diagonal
        _, _, yaw = quaternion_to_euler_zyx(est.q)
        yaw_traj[i] = yaw

        nhc_nu_list.append(est.nhc_nu)
        nhc_accepted_list.append(est.nhc_accepted)
        zupt_accepted_list.append(est.zupt_accepted)
        zaru_accepted_list.append(est.zaru_accepted)
        is_stat_list.append(est.is_stationary)
        nhc_conf_list.append(est.nhc_confidence)

    return {
        "p_est": p_traj,
        "v_est": v_traj,
        "q_est": q_traj,
        "v_b_est": v_b_traj,
        "yaw_est": yaw_traj,
        "ba_est": ba_traj,
        "bg_est": bg_traj,
        "cov_diags": cov_diags,
        "nhc_nu_list": nhc_nu_list,
        "nhc_accepted_list": nhc_accepted_list,
        "zupt_accepted_list": zupt_accepted_list,
        "zaru_accepted_list": zaru_accepted_list,
        "is_stat_list": np.array(is_stat_list),
        "nhc_conf_list": np.array(nhc_conf_list),
    }
