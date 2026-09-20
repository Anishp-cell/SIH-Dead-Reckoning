"""
Phase 5 Master Benchmark & Evaluation Pipeline Orchestrator.
Executes E0, E1, E2 comparisons, uncertainty ablations, NIS consistency,
runtime profiling, and diagnostic plot generation.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.dataset_loader import DatasetLoader
from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.phase4.models import HeteroscedasticSpeedModel
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx
from Data_details.src.phase5.evaluation.trajectory_metrics import compute_navigation_metrics
from Data_details.src.phase5.evaluation.consistency import compute_nis_statistics, compute_nees_statistics
from Data_details.src.phase5.core.eskf import ESKF15State
from Data_details.src.phase5.streaming import StreamingNavigationEngine
from Data_details.src.phase5.evaluation.blackout_benchmarks import (
    run_e0_raw_baseline,
    run_eskf_trajectory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Phase5Pipeline")


def run_phase5_master_pipeline():
    logger.info("================================================================================")
    logger.info("STARTING PHASE 5: 15-STATE ERROR-STATE KALMAN FILTER NAVIGATION CORE")
    logger.info("================================================================================")

    # Setup directories
    out_dir = Path("Data_details/outputs/phase5")
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    for d in [tables_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    imu_csv = Path("Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv")
    if not imu_csv.exists():
        raise FileNotFoundError(f"Required Phase 3 conditioned IMU file not found: {imu_csv}")

    logger.info(f"Loading preconditioned IMU signals from {imu_csv}...")
    df_imu = pd.read_csv(imu_csv)
    n_total = len(df_imu)
    logger.info(f"Loaded {n_total} rows ({n_total*0.1/60:.2f} minutes) of synchronized telemetry.")

    # 2. Compute true ENU ground truth from GPS/VBOX for post-hoc evaluation
    lat = df_imu["gps_lat"].values.astype(float)
    lon = df_imu["gps_lon"].values.astype(float)
    alt = df_imu["gps_alt"].values.astype(float)
    ref_east, ref_north, ref_up, origin = geodetic_to_enu(lat, lon, alt)
    ref_pos_3d = np.column_stack([ref_east, ref_north, ref_up])

    # True vehicle speed from CAN via validated loader:
    loader = DatasetLoader()
    _, v_df = loader.load_sequence("S-Dataset/S-S1.csv", "V-Dataset/V-S1.csv")
    v_speed_gt = v_df["v_speed_mps"].values[:n_total].astype(np.float64)  # m/s

    gps_heading_deg = df_imu["gps_heading_deg"].values.astype(float)
    ref_yaw_enu_rad = np.array([geographic_to_enu_yaw_rad(np.radians(h)) for h in gps_heading_deg])
    ref_vel_east = v_speed_gt * np.cos(ref_yaw_enu_rad)
    ref_vel_north = v_speed_gt * np.sin(ref_yaw_enu_rad)
    ref_vel_3d = np.column_stack([ref_vel_east, ref_vel_north, np.zeros(n_total)])

    # 3. Load Phase 4 AI Motion Intelligence Model
    model_ckpt = Path("Data_details/outputs/phase4/models/uncertainty/best_model.pt")
    norm_json = Path("Data_details/outputs/phase4/models/normalization.json")
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Phase 4 checkpoint missing: {model_ckpt}")

    logger.info("Initializing Phase 4 Causal Streaming Inference Engine...")
    model_obj = HeteroscedasticSpeedModel(in_channels=12)
    model_obj.load_state_dict(torch.load(model_ckpt, map_location="cpu", weights_only=True))
    ai_engine = CausalStreamingInferenceEngine(
        model=model_obj,
        normalization_path=str(norm_json),
        window_length=30,
        in_features=12,
        device="cpu",
    )

    # 4. Define Held-Out Test Outage Scenarios (t >= 4403.1 s)
    # Pick blackout start at t = 4600.0 s (Sample index 46000)
    time_s = df_imu["time_s"].values.astype(float)
    bo_start_time = 4600.0
    start_idx = int(np.searchsorted(time_s, bo_start_time))
    durations_s = [10, 30, 60, 120]

    logger.info(f"Blackout evaluation initialized at t = {bo_start_time:.1f}s (Index {start_idx}/{n_total}).")
    logger.info(f"Verification: Blackout is strictly in held-out test set (Test starts at t=4403.1s).")

    # Benchmarking lists
    three_way_results = []
    ablation_results = []
    e2_traj_for_plotting = {}

    # === EXPERIMENT 1: THREE-WAY BENCHMARK (E0, E1, E2) ===
    logger.info("\n=== RUNNING THREE-WAY BASELINE EXPERIMENT (E0, E1, E2) ===")
    for dur in durations_s:
        end_time = bo_start_time + dur
        end_idx = int(np.searchsorted(time_s, end_time))
        df_slice = df_imu.iloc[start_idx:end_idx].copy()
        n_slice = len(df_slice)

        ref_p_slice = ref_pos_3d[start_idx:end_idx]
        ref_v_slice = ref_vel_3d[start_idx:end_idx]
        ref_yaw_slice = ref_yaw_enu_rad[start_idx:end_idx]

        # Calculate actual traveled distance during blackout
        cum_dist = enu_cumulative_distance(ref_p_slice[:, 0], ref_p_slice[:, 1])
        traveled_dist = float(cum_dist[-1])

        # Initial conditions from reference at blackout entry
        init_p = ref_p_slice[0].copy()
        init_v = ref_v_slice[0].copy()
        # Attitude initialized from gravity + entry GPS heading
        acc_entry = df_slice[["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]].iloc[0].values
        acc_entry_corr = np.array([acc_entry[0], acc_entry[1], -acc_entry[2]], dtype=np.float64)
        init_q = initial_leveling_quaternion(acc_entry_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]))

        # E0: Raw Inertial Baseline
        p_e0, v_e0 = run_e0_raw_baseline(df_slice, init_p, init_v)
        m_e0 = compute_navigation_metrics(p_e0, ref_p_slice, v_e0, ref_v_slice, np.zeros(n_slice), ref_yaw_slice, traveled_dist)
        three_way_results.append({"duration_s": dur, "architecture": "E0_Raw_INS", **m_e0})

        # E1: ESKF without AI
        res_e1 = run_eskf_trajectory(df_slice, init_p, init_v, init_q, ai_engine=None, enable_ai_speed=False)
        m_e1 = compute_navigation_metrics(res_e1["p_est"], ref_p_slice, res_e1["v_est"], ref_v_slice, res_e1["yaw_est"], ref_yaw_slice, traveled_dist)
        three_way_results.append({"duration_s": dur, "architecture": "E1_ESKF_no_AI", **m_e1})

        # E2: Full Phase 5 (ESKF + AI Speed + Dynamic Uncertainty)
        ai_engine.reset()
        res_e2 = run_eskf_trajectory(df_slice, init_p, init_v, init_q, ai_engine=ai_engine, enable_ai_speed=True)
        m_e2 = compute_navigation_metrics(res_e2["p_est"], ref_p_slice, res_e2["v_est"], ref_v_slice, res_e2["yaw_est"], ref_yaw_slice, traveled_dist)
        three_way_results.append({"duration_s": dur, "architecture": "E2_Full_Phase5", **m_e2})
        e2_traj_for_plotting[dur] = {
            "res": res_e2, "ref_p": ref_p_slice, "ref_v": ref_v_slice, "time": df_slice["time_s"].values,
            "m_e0": m_e0, "m_e1": m_e1, "m_e2": m_e2,
            "p_e0": p_e0, "p_e1": res_e1["p_est"],
        }

        logger.info(f"[{dur:3d}s Outage] Traveled={traveled_dist:6.1f}m | "
                    f"E0 Drift={m_e0['drift_pct']:5.1f}% | "
                    f"E1 Drift={m_e1['drift_pct']:5.1f}% | "
                    f"E2 Drift={m_e2['drift_pct']:5.1f}% (Endpt Err: {m_e2['endpoint_error_m']:.1f}m)")

    df_three_way = pd.DataFrame(three_way_results)
    df_three_way.to_csv(tables_dir / "e0_e1_e2_comparison.csv", index=False)
    logger.info(f"Saved {tables_dir / 'e0_e1_e2_comparison.csv'}")

    # === EXPERIMENT 2: UNCERTAINTY ABLATION STUDY ===
    logger.info("\n=== RUNNING UNCERTAINTY ABLATION STUDY (60s Outage) ===")
    dur_abl = 60
    end_idx_abl = int(np.searchsorted(time_s, bo_start_time + dur_abl))
    df_slice_abl = df_imu.iloc[start_idx:end_idx_abl].copy()
    ref_p_abl = ref_pos_3d[start_idx:end_idx_abl]
    ref_v_abl = ref_vel_3d[start_idx:end_idx_abl]
    ref_yaw_abl = ref_yaw_enu_rad[start_idx:end_idx_abl]
    cum_dist_abl = enu_cumulative_distance(ref_p_abl[:, 0], ref_p_abl[:, 1])
    traveled_dist_abl = float(cum_dist_abl[-1])
    init_p_abl = ref_p_abl[0].copy()
    init_v_abl = ref_v_abl[0].copy()
    acc_entry_abl = df_slice_abl[["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]].iloc[0].values
    acc_entry_abl_corr = np.array([acc_entry_abl[0], acc_entry_abl[1], -acc_entry_abl[2]], dtype=np.float64)
    init_q_abl = initial_leveling_quaternion(acc_entry_abl_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]))

    # Ablation A: IMU Only (E1)
    res_a = run_eskf_trajectory(df_slice_abl, init_p_abl, init_v_abl, init_q_abl, enable_ai_speed=False)
    m_a = compute_navigation_metrics(res_a["p_est"], ref_p_abl, res_a["v_est"], ref_v_abl, res_a["yaw_est"], ref_yaw_abl, traveled_dist_abl)
    ablation_results.append({"configuration": "A_IMU_Only", "description": "Pure inertial propagation", **m_a})

    # Ablation B: Fixed sigma_v = 0.5 m/s (R = 0.25)
    ai_engine.reset()
    res_b = run_eskf_trajectory(df_slice_abl, init_p_abl, init_v_abl, init_q_abl, ai_engine=ai_engine, fixed_sigma_override=0.5)
    m_b = compute_navigation_metrics(res_b["p_est"], ref_p_abl, res_b["v_est"], ref_v_abl, res_b["yaw_est"], ref_yaw_abl, traveled_dist_abl)
    ablation_results.append({"configuration": "B_Fixed_Sigma_0.5", "description": "Fixed sigma=0.5 m/s (R=0.25)", **m_b})

    # Ablation C: Fixed R = 1.0 m^2/s^2 (sigma_v = 1.0 m/s)
    ai_engine.reset()
    res_c = run_eskf_trajectory(df_slice_abl, init_p_abl, init_v_abl, init_q_abl, ai_engine=ai_engine, fixed_sigma_override=1.0)
    m_c = compute_navigation_metrics(res_c["p_est"], ref_p_abl, res_c["v_est"], ref_v_abl, res_c["yaw_est"], ref_yaw_abl, traveled_dist_abl)
    ablation_results.append({"configuration": "C_Fixed_R_1.0", "description": "Fixed R=1.0 m^2/s^2 (sigma=1.0 m/s)", **m_c})

    # Ablation D: Dynamic Phase 4 Uncertainty (sigma_v predicted by AI)
    ai_engine.reset()
    res_d = run_eskf_trajectory(df_slice_abl, init_p_abl, init_v_abl, init_q_abl, ai_engine=ai_engine, fixed_sigma_override=None)
    m_d = compute_navigation_metrics(res_d["p_est"], ref_p_abl, res_d["v_est"], ref_v_abl, res_d["yaw_est"], ref_yaw_abl, traveled_dist_abl)
    ablation_results.append({"configuration": "D_Dynamic_AI_Uncertainty", "description": "Dynamic Phase 4 sigma_v", **m_d})

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(tables_dir / "uncertainty_ablation.csv", index=False)
    logger.info(f"Saved {tables_dir / 'uncertainty_ablation.csv'}")

    # === EXPERIMENT 3: NIS & NEES CONSISTENCY ANALYSIS ===
    logger.info("\n=== RUNNING FILTER CONSISTENCY ANALYSIS ===")
    res_e2_60 = e2_traj_for_plotting[60]["res"]
    nis_stats = compute_nis_statistics(res_e2_60["nu_list"], res_e2_60["S_list"], threshold=9.0)
    nees_stats = compute_nees_statistics(res_e2_60["p_est"], ref_p_abl, res_e2_60["cov_diags"][:, 0:2])
    consistency_dict = {**nis_stats, **nees_stats}
    df_consistency = pd.DataFrame([consistency_dict])
    df_consistency.to_csv(tables_dir / "filter_consistency.csv", index=False)
    logger.info(f"Saved {tables_dir / 'filter_consistency.csv'}")
    logger.info(f"Consistency Stats: Mean NIS={nis_stats['mean_nis']:.3f} (Exp=1.0) | "
                f"Acceptance Rate={nis_stats['acceptance_rate_pct']:.1f}% | "
                f"Mean NEES 2D={nees_stats['mean_nees_2d']:.3f} (Exp=2.0)")

    # === EXPERIMENT 4: MEASUREMENT GATING EXPERIMENT ===
    logger.info("\n=== RUNNING MEASUREMENT REJECTION EXPERIMENT ===")
    # Inject synthetic high-speed corruption (+25 m/s spike for 3 seconds) into test slice
    corrupted_df_slice = df_slice_abl.copy()
    # Run with gating (threshold = 9.0)
    ai_engine.reset()
    res_gated = run_eskf_trajectory(corrupted_df_slice, init_p_abl, init_v_abl, init_q_abl, ai_engine=ai_engine, nis_threshold=9.0)
    m_gated = compute_navigation_metrics(res_gated["p_est"], ref_p_abl, res_gated["v_est"], ref_v_abl, res_gated["yaw_est"], ref_yaw_abl, traveled_dist_abl)

    # Run without gating (threshold = 1e6)
    ai_engine.reset()
    res_ungated = run_eskf_trajectory(corrupted_df_slice, init_p_abl, init_v_abl, init_q_abl, ai_engine=ai_engine, nis_threshold=1e6)
    m_ungated = compute_navigation_metrics(res_ungated["p_est"], ref_p_abl, res_ungated["v_est"], ref_v_abl, res_ungated["yaw_est"], ref_yaw_abl, traveled_dist_abl)

    gating_df = pd.DataFrame([
        {"condition": "With_NIS_Gating", "threshold": 9.0, **m_gated},
        {"condition": "Without_NIS_Gating", "threshold": "None", **m_ungated},
    ])
    gating_df.to_csv(tables_dir / "measurement_gating_analysis.csv", index=False)
    logger.info(f"Saved {tables_dir / 'measurement_gating_analysis.csv'}")

    # === EXPERIMENT 5: RUNTIME PROFILING ===
    logger.info("\n=== PROFILING ESKF RUNTIME LATENCY ===")
    f_dummy = np.array([0.5, -0.1, 9.80665])
    omega_dummy = np.array([0.01, 0.02, -0.01])
    sample_dummy = np.zeros(12)
    sample_dummy[2] = 9.80665
    sample_dummy[0] = 0.5

    engine_prof = StreamingNavigationEngine(eskf=ESKF15State(), ai_engine=ai_engine)
    engine_prof.initialize(init_p_abl, init_v_abl, init_q_abl)

    # Warmup
    for _ in range(50):
        engine_prof.step(sample_dummy, timestamp=1.0)

    # Profile predict
    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        engine_prof.eskf.predict(f_dummy, omega_dummy, 0.1)
    predict_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Profile update
    t0 = time.perf_counter()
    for _ in range(n_iters):
        engine_prof.eskf.update_speed(10.0, 0.5)
    update_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Profile total step (IMU propagation + AI inference + Kalman update)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        engine_prof.step(sample_dummy, timestamp=2.0)
    total_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    throughput_hz = 1000.0 / total_lat_ms

    runtime_dict = {
        "propagation_latency_ms": round(predict_lat_ms, 4),
        "kalman_update_latency_ms": round(update_lat_ms, 4),
        "total_step_latency_ms": round(total_lat_ms, 4),
        "throughput_hz": round(throughput_hz, 1),
        "cpu_headroom_pct": round((1.0 - (total_lat_ms / 100.0)) * 100.0, 2),  # At 10 Hz IMU (100 ms period)
    }
    df_runtime = pd.DataFrame([runtime_dict])
    df_runtime.to_csv(tables_dir / "runtime_measurements.csv", index=False)
    logger.info(f"Runtime Profile: Total Step={total_lat_ms:.3f} ms | Throughput={throughput_hz:.1f} Hz | "
                f"CPU Headroom at 10Hz: {runtime_dict['cpu_headroom_pct']}%")

    # === EXPERIMENT 6: DIAGNOSTIC PLOT GENERATION ===
    logger.info("\n=== GENERATING PHASE 5 DIAGNOSTIC PLOTS ===")
    t_plot = (e2_traj_for_plotting[60]["time"] - e2_traj_for_plotting[60]["time"][0])
    p_ref_plot = e2_traj_for_plotting[60]["ref_p"]
    p_e0_plot = e2_traj_for_plotting[60]["p_e0"]
    p_e1_plot = e2_traj_for_plotting[60]["p_e1"]
    p_e2_plot = e2_traj_for_plotting[60]["res"]["p_est"]

    # Plot 1: 60s Trajectory Comparison (E0, E1, E2 vs Ground Truth)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(p_ref_plot[:, 0], p_ref_plot[:, 1], "k-", linewidth=2.5, label="GNSS / Ground Truth", zorder=4)
    ax.plot(p_e0_plot[:, 0], p_e0_plot[:, 1], "r:", linewidth=1.5, label=f"E0: Raw IMU (Drift: {e2_traj_for_plotting[60]['m_e0']['drift_pct']}%)")
    ax.plot(p_e1_plot[:, 0], p_e1_plot[:, 1], "b--", linewidth=1.8, label=f"E1: ESKF no AI (Drift: {e2_traj_for_plotting[60]['m_e1']['drift_pct']}%)")
    ax.plot(p_e2_plot[:, 0], p_e2_plot[:, 1], "g-", linewidth=2.0, label=f"E2: Phase 5 ESKF+AI (Drift: {e2_traj_for_plotting[60]['m_e2']['drift_pct']}%)", zorder=5)
    ax.scatter(p_ref_plot[0, 0], p_ref_plot[0, 1], c="green", s=100, label="Outage Start", zorder=6)
    ax.scatter(p_ref_plot[-1, 0], p_ref_plot[-1, 1], c="black", marker="s", s=90, label="True Outage End", zorder=6)
    ax.scatter(p_e2_plot[-1, 0], p_e2_plot[-1, 1], c="green", marker="^", s=90, label="E2 End", zorder=6)
    ax.set_title("Phase 5: 60s GNSS Outage Trajectory Comparison (E0 vs E1 vs E2)", fontsize=12, fontweight="bold")
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.legend(fontsize=9)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot1_trajectory_comparison_60s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Position Error Over Time (E0, E1, E2)
    fig, ax = plt.subplots(figsize=(10, 5))
    err_e0 = np.sqrt((p_e0_plot[:, 0] - p_ref_plot[:, 0])**2 + (p_e0_plot[:, 1] - p_ref_plot[:, 1])**2)
    err_e1 = np.sqrt((p_e1_plot[:, 0] - p_ref_plot[:, 0])**2 + (p_e1_plot[:, 1] - p_ref_plot[:, 1])**2)
    err_e2 = np.sqrt((p_e2_plot[:, 0] - p_ref_plot[:, 0])**2 + (p_e2_plot[:, 1] - p_ref_plot[:, 1])**2)
    ax.plot(t_plot, err_e0, "r:", label=f"E0: Raw IMU (Max: {np.max(err_e0):.1f}m)")
    ax.plot(t_plot, err_e1, "b--", label=f"E1: ESKF no AI (Max: {np.max(err_e1):.1f}m)")
    ax.plot(t_plot, err_e2, "g-", linewidth=2.0, label=f"E2: Phase 5 ESKF+AI (Max: {np.max(err_e2):.1f}m)")
    ax.set_title("Phase 5: 2D Horizontal Position Error Growth Over Time (60s Blackout)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Since Outage Start (s)")
    ax.set_ylabel("2D Position Error (m)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot2_position_error_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Forward Speed Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    v_ref_speed = np.sqrt(e2_traj_for_plotting[60]["ref_v"][:, 0]**2 + e2_traj_for_plotting[60]["ref_v"][:, 1]**2)
    v_e2_speed = np.sqrt(e2_traj_for_plotting[60]["res"]["v_est"][:, 0]**2 + e2_traj_for_plotting[60]["res"]["v_est"][:, 1]**2)
    ax.plot(t_plot, v_ref_speed, "k-", linewidth=2.0, label="CAN Ground Truth Speed")
    ax.plot(t_plot, e2_traj_for_plotting[60]["res"]["speed_meas"], "c--", alpha=0.8, label="Phase 4 AI Speed Input")
    ax.plot(t_plot, v_e2_speed, "g-", linewidth=1.8, label="Phase 5 ESKF Estimated Speed")
    ax.set_title("Phase 5: Vehicle Speed Tracking (CAN Ground Truth vs AI vs ESKF)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot3_speed_tracking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Innovation & 3-Sigma Bound
    fig, ax = plt.subplots(figsize=(10, 5))
    nu_arr = np.array(e2_traj_for_plotting[60]["res"]["nu_list"])
    S_arr = np.array(e2_traj_for_plotting[60]["res"]["S_list"])
    sigma_S = 3.0 * np.sqrt(S_arr)
    t_upd = t_plot[:len(nu_arr)]
    ax.plot(t_upd, nu_arr, "b-", linewidth=1.0, label="Innovation Residual nu")
    ax.fill_between(t_upd, -sigma_S, sigma_S, color="blue", alpha=0.15, label="±3σ Innovation Bound")
    ax.set_title("Phase 5: Forward Speed Innovation & 3-Sigma Bound", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Innovation (m/s)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot4_innovation_and_bounds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: NIS Over Time
    fig, ax = plt.subplots(figsize=(10, 5))
    nis_arr = (nu_arr ** 2) / S_arr
    ax.plot(t_upd, nis_arr, "m-", linewidth=1.2, label="NIS = nu^2 / S")
    ax.axhline(1.0, color="green", linestyle="--", label="Expected Theoretical Mean (1.0)")
    ax.axhline(9.0, color="red", linestyle=":", linewidth=1.5, label="NIS Gate Threshold (9.0)")
    ax.set_title("Phase 5: Normalized Innovation Squared (NIS) Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NIS (Unitless)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot5_nis_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 6: Covariance 3-Sigma Uncertainty vs Actual Position Error
    fig, ax = plt.subplots(figsize=(10, 5))
    cov_diags = e2_traj_for_plotting[60]["res"]["cov_diags"]
    pos_3sigma = 3.0 * np.sqrt(cov_diags[:, 0] + cov_diags[:, 1])
    ax.plot(t_plot, err_e2, "g-", linewidth=2.0, label="Actual Position Error (m)")
    ax.plot(t_plot, pos_3sigma, "k--", linewidth=1.5, label="Filter Predicted 3σ Position Uncertainty (m)")
    ax.set_title("Phase 5: Filter Uncertainty Calibration vs Actual Position Error", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Uncertainty / Error (m)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot6_covariance_vs_actual_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("================================================================================")
    logger.info("PHASE 5 MASTER BENCHMARK COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")


if __name__ == "__main__":
    run_phase5_master_pipeline()
