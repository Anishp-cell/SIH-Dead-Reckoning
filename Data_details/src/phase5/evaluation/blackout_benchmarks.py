"""
Phase 5 Blackout Benchmarks: E0 (Raw), E1 (ESKF-no-AI), E2 (ESKF+AI) Comparison.
Strictly executed on held-out test partition (t >= 4403.1 s).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.inertial_baseline import integrate_dead_reckoning, phone_to_enu_simple, gravity_removal_simple
from Data_details.src.phase5.core.eskf import ESKF15State
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx
from Data_details.src.phase5.evaluation.trajectory_metrics import compute_navigation_metrics
from Data_details.src.phase5.evaluation.consistency import compute_nis_statistics, compute_nees_statistics
from Data_details.src.phase5.streaming import StreamingNavigationEngine
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from Data_details.src.phase4.models import HeteroscedasticSpeedModel


def run_e0_raw_baseline(
    df_slice: pd.DataFrame,
    init_p_enu: np.ndarray,
    init_v_enu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs classical raw IMU double integration (E0 Baseline)."""
    acc_x = df_slice["acc_x"].values.astype(float)
    acc_y = df_slice["acc_y"].values.astype(float)
    acc_z = df_slice["acc_z"].values.astype(float)
    grav_x = df_slice["grav_x"].values.astype(float)
    grav_y = df_slice["grav_y"].values.astype(float)
    grav_z = df_slice["grav_z"].values.astype(float)
    yaw_deg = df_slice["ori_yaw_deg"].values.astype(float)
    pitch_deg = df_slice["ori_pitch_deg"].values.astype(float)
    roll_deg = df_slice["ori_roll_deg"].values.astype(float)
    dt_arr = df_slice["dt"].values.astype(float)

    lin_x, lin_y, lin_z = gravity_removal_simple(acc_x, acc_y, acc_z, grav_x, grav_y, grav_z)
    acc_east, acc_north = phone_to_enu_simple(lin_x, lin_y, lin_z, yaw_deg, pitch_deg, roll_deg)

    vel_e, vel_n, pos_e, pos_n = integrate_dead_reckoning(
        acc_east, acc_north, dt_arr,
        initial_vel_east=init_v_enu[0],
        initial_vel_north=init_v_enu[1],
        initial_pos_east=init_p_enu[0],
        initial_pos_north=init_p_enu[1],
    )
    p_est = np.column_stack([pos_e, pos_n, np.zeros_like(pos_e)])
    v_est = np.column_stack([vel_e, vel_n, np.zeros_like(vel_e)])
    return p_est, v_est


def run_eskf_trajectory(
    df_slice: pd.DataFrame,
    init_p_enu: np.ndarray,
    init_v_enu: np.ndarray,
    init_q: np.ndarray,
    ai_engine: Optional[CausalStreamingInferenceEngine] = None,
    enable_ai_speed: bool = True,
    fixed_sigma_override: Optional[float] = None,
    nis_threshold: float = 9.0,
) -> Dict[str, Any]:
    """Runs sequential 15-state ESKF over slice."""
    eskf = ESKF15State(nis_threshold=nis_threshold)
    engine = StreamingNavigationEngine(eskf=eskf, ai_engine=ai_engine)

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
    cov_diags = np.zeros((n, 15))
    yaw_traj = np.zeros(n)
    nu_list = []
    S_list = []
    speed_pred_list = []
    speed_meas_list = []
    accepted_flags = []

    # 12-channel column names from Phase 3/4 interface
    feature_cols = [
        "acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered",
        "gyro_roll_veh_filtered", "gyro_pitch_veh_filtered", "gyro_yaw_veh_filtered",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    # Fallback to unconditioned vehicle cols if filtered missing
    for i, c in enumerate(feature_cols):
        if c not in df_slice.columns:
            fallback = c.replace("_filtered", "")
            if fallback in df_slice.columns:
                feature_cols[i] = fallback

    features_matrix = df_slice[feature_cols].values.astype(np.float32)
    time_arr = df_slice["time_s"].values.astype(float)
    dt_arr = df_slice["dt"].values.astype(float)

    for i in range(n):
        sample = features_matrix[i]
        t = time_arr[i]
        dt = dt_arr[i]

        # 1. Prediction step
        f_meas = np.array([sample[0], sample[1], -sample[2]], dtype=np.float64)
        omega_meas = sample[3:6]
        eskf.predict(f_meas, omega_meas, dt)

        # 2. AI forward speed update (if enabled)
        if enable_ai_speed and ai_engine is not None:
            motion = ai_engine.update(sample, timestamp=t)
            speed_val = motion.forward_speed_mps
            sigma_val = fixed_sigma_override if fixed_sigma_override is not None else motion.speed_uncertainty

            eskf.update_speed(speed_val, sigma_val)

            nu_list.append(eskf.last_nu)
            S_list.append(eskf.last_S)
            speed_pred_list.append(motion.forward_speed_mps - eskf.last_nu)
            speed_meas_list.append(motion.forward_speed_mps)
            accepted_flags.append(eskf.last_accepted)

        est = eskf.get_estimate(timestamp=t)
        p_traj[i] = est.p
        v_traj[i] = est.v
        q_traj[i] = est.q
        cov_diags[i] = est.cov_diagonal
        _, _, yaw = quaternion_to_euler_zyx(est.q)
        yaw_traj[i] = yaw

    return {
        "p_est": p_traj,
        "v_est": v_traj,
        "q_est": q_traj,
        "yaw_est": yaw_traj,
        "cov_diags": cov_diags,
        "nu_list": nu_list,
        "S_list": S_list,
        "speed_pred": np.array(speed_pred_list),
        "speed_meas": np.array(speed_meas_list),
        "accepted_flags": accepted_flags,
    }
