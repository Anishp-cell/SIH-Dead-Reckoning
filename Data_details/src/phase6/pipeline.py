"""
Phase 6 Master Evaluation Pipeline:
Executes Phase 5 vs Phase 6 Benchmarks, E2-E6 Ablation, Physics Metrics, Runtime Profiling, and Plot Generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import time
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.dataset_loader import DatasetLoader
from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase5.evaluation.trajectory_metrics import compute_navigation_metrics
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from Data_details.src.phase4.models import HeteroscedasticSpeedModel
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase6.streaming import Phase6StreamingEngine
from Data_details.src.phase6.evaluation.blackout_benchmarks import run_phase6_trajectory
from Data_details.src.phase6.evaluation.constraint_metrics import compute_constraint_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase6Pipeline")


def run_phase6_master_pipeline():
    logger.info("================================================================================")
    logger.info("STARTING PHASE 6: VEHICLE PHYSICS CONSTRAINTS, ZUPT/ZARU, AND DISTURBANCE-AWARE ESKF")
    logger.info("================================================================================")

    # Output directories
    out_dir = Path("Data_details/outputs/phase6")
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    for d in [tables_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 3 preconditioned IMU signals
    imu_csv = Path("Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv")
    if not imu_csv.exists():
        raise FileNotFoundError(f"Missing required conditioned IMU: {imu_csv}")

    logger.info(f"Loading preconditioned IMU signals from {imu_csv}...")
    df_imu = pd.read_csv(imu_csv)
    n_total = len(df_imu)
    logger.info(f"Loaded {n_total} samples ({n_total*0.1/60:.2f} minutes).")

    # 2. Compute reference ground truth in local ENU
    lat = df_imu["gps_lat"].values.astype(float)
    lon = df_imu["gps_lon"].values.astype(float)
    alt = df_imu["gps_alt"].values.astype(float)
    ref_east, ref_north, ref_up, _ = geodetic_to_enu(lat, lon, alt)
    ref_pos_3d = np.column_stack([ref_east, ref_north, ref_up])

    loader = DatasetLoader()
    _, v_df = loader.load_sequence("S-Dataset/S-S1.csv", "V-Dataset/V-S1.csv")
    v_speed_gt = v_df["v_speed_mps"].values[:n_total].astype(np.float64)

    gps_heading_deg = df_imu["gps_heading_deg"].values.astype(float)
    ref_yaw_enu_rad = np.array([geographic_to_enu_yaw_rad(np.radians(h)) for h in gps_heading_deg])
    ref_vel_east = v_speed_gt * np.cos(ref_yaw_enu_rad)
    ref_vel_north = v_speed_gt * np.sin(ref_yaw_enu_rad)
    ref_vel_3d = np.column_stack([ref_vel_east, ref_vel_north, np.zeros(n_total)])

    # 3. Load Phase 4 AI model
    model_ckpt = Path("Data_details/outputs/phase4/models/uncertainty/best_model.pt")
    norm_json = Path("Data_details/outputs/phase4/models/normalization.json")
    model = HeteroscedasticSpeedModel(in_channels=12)
    state_dict = torch_load_weights(model_ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    ai_engine = CausalStreamingInferenceEngine(
        model=model,
        normalization_path=str(norm_json),
        window_length=30,
        in_features=12,
        device="cpu",
    )

    time_s = df_imu["time_s"].values.astype(float)

    # === EXPERIMENT 1: PHASE 5 (E2) VS PHASE 6 (E6) COMPARISON ACROSS DURATIONS ===
    logger.info("\n=== EXPERIMENT 1: PHASE 5 (E2) VS PHASE 6 (E6) BENCHMARK ===")
    bo_start_time = 4600.0  # Exact Phase 5 evaluated start point (includes traffic standstill)
    start_idx = int(np.searchsorted(time_s, bo_start_time))
    durations_s = [10, 30, 60, 120]

    p5_vs_p6_results = []
    traj_records = {}

    for dur in durations_s:
        end_time = bo_start_time + dur
        end_idx = int(np.searchsorted(time_s, end_time))
        df_slice = df_imu.iloc[start_idx:end_idx].copy()

        ref_p_slice = ref_pos_3d[start_idx:end_idx]
        ref_v_slice = ref_vel_3d[start_idx:end_idx]
        ref_yaw_slice = ref_yaw_enu_rad[start_idx:end_idx]

        cum_dist = enu_cumulative_distance(ref_p_slice[:, 0], ref_p_slice[:, 1])
        traveled_dist = float(cum_dist[-1])

        init_p = ref_p_slice[0].copy()
        init_v = ref_v_slice[0].copy()
        acc_entry = df_slice[["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]].iloc[0].values
        acc_entry_corr = np.array([acc_entry[0], acc_entry[1], -acc_entry[2]], dtype=np.float64)
        init_q = initial_leveling_quaternion(acc_entry_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]))

        # E2: Full Phase 5 (ESKF + AI Speed, NO vehicle physics constraints)
        ai_engine.reset()
        res_e2 = run_phase6_trajectory(
            df_slice, init_p, init_v, init_q, ai_engine=ai_engine,
            enable_ai_speed=True, enable_nhc=False, enable_zupt=False, enable_zaru=False
        )
        m_e2 = compute_navigation_metrics(res_e2["p_est"], ref_p_slice, res_e2["v_est"], ref_v_slice, res_e2["yaw_est"], ref_yaw_slice, traveled_dist)
        c_e2 = compute_constraint_metrics(
            res_e2["v_b_est"], res_e2["p_est"], res_e2["is_stat_list"],
            res_e2["nhc_nu_list"], res_e2["nhc_accepted_list"], res_e2["zupt_accepted_list"], res_e2["zaru_accepted_list"],
            res_e2["ba_est"], res_e2["bg_est"],
        )
        p5_vs_p6_results.append({"duration_s": dur, "architecture": "Phase5_Full_E2", **m_e2, **c_e2})

        # E6: Full Phase 6 (ESKF + AI Speed + NHC + ZUPT + ZARU + Adaptive Weighting)
        ai_engine.reset()
        res_e6 = run_phase6_trajectory(
            df_slice, init_p, init_v, init_q, ai_engine=ai_engine,
            enable_ai_speed=True, enable_nhc=True, enable_zupt=True, enable_zaru=True, enable_disturbance_adaptive=True
        )
        m_e6 = compute_navigation_metrics(res_e6["p_est"], ref_p_slice, res_e6["v_est"], ref_v_slice, res_e6["yaw_est"], ref_yaw_slice, traveled_dist)
        c_e6 = compute_constraint_metrics(
            res_e6["v_b_est"], res_e6["p_est"], res_e6["is_stat_list"],
            res_e6["nhc_nu_list"], res_e6["nhc_accepted_list"], res_e6["zupt_accepted_list"], res_e6["zaru_accepted_list"],
            res_e6["ba_est"], res_e6["bg_est"],
        )
        p5_vs_p6_results.append({"duration_s": dur, "architecture": "Phase6_Full_E6", **m_e6, **c_e6})

        traj_records[dur] = {
            "res_e2": res_e2, "res_e6": res_e6,
            "ref_p": ref_p_slice, "ref_v": ref_v_slice, "ref_yaw": ref_yaw_slice,
            "time": df_slice["time_s"].values, "m_e2": m_e2, "m_e6": m_e6, "c_e2": c_e2, "c_e6": c_e6,
        }

        logger.info(f"[{dur:3d}s Outage] Traveled={traveled_dist:6.1f}m | "
                    f"Phase 5 Drift: {m_e2['drift_pct']:5.1f}% (Endpt: {m_e2['endpoint_error_m']:.1f}m) | "
                    f"Phase 6 Drift: {m_e6['drift_pct']:5.1f}% (Endpt: {m_e6['endpoint_error_m']:.1f}m)")

    df_p5_vs_p6 = pd.DataFrame(p5_vs_p6_results)
    df_p5_vs_p6.to_csv(tables_dir / "phase5_vs_phase6_comparison.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase5_vs_phase6_comparison.csv'}")

    # === EXPERIMENT 2: STEP-BY-STEP COMPONENT ABLATION STUDY (60s Outage) ===
    logger.info("\n=== EXPERIMENT 2: COMPONENT ABLATION STUDY (E2 -> E3 -> E4 -> E5 -> E6) ===")
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

    ablation_runs = [
        ("E2_Phase5_Full", "AI Speed only", True, False, False, False, False),
        ("E3_Plus_NHC", "E2 + NHC (Fixed R)", True, True, False, False, False),
        ("E4_Plus_ZUPT", "E3 + ZUPT", True, True, True, False, False),
        ("E5_Plus_ZARU", "E4 + ZARU", True, True, True, True, False),
        ("E6_Full_Phase6", "E5 + Disturbance Adaptive", True, True, True, True, True),
    ]

    ablation_results = []
    for name, desc, en_ai, en_nhc, en_zupt, en_zaru, en_adapt in ablation_runs:
        ai_engine.reset()
        res = run_phase6_trajectory(
            df_slice_abl, init_p_abl, init_v_abl, init_q_abl, ai_engine=ai_engine,
            enable_ai_speed=en_ai, enable_nhc=en_nhc, enable_zupt=en_zupt, enable_zaru=en_zaru,
            enable_disturbance_adaptive=en_adapt,
        )
        m = compute_navigation_metrics(res["p_est"], ref_p_abl, res["v_est"], ref_v_abl, res["yaw_est"], ref_yaw_abl, traveled_dist_abl)
        c = compute_constraint_metrics(
            res["v_b_est"], res["p_est"], res["is_stat_list"],
            res["nhc_nu_list"], res["nhc_accepted_list"], res["zupt_accepted_list"], res["zaru_accepted_list"],
            res["ba_est"], res["bg_est"],
        )
        ablation_results.append({"configuration": name, "description": desc, **m, **c})
        logger.info(f"[{name}] Drift={m['drift_pct']:5.1f}% | Endpt={m['endpoint_error_m']:6.1f}m | "
                    f"LatVel RMSE={c['lateral_vel_rmse_mps']:.3f}m/s | StatCreep={c['stationary_creep_m']:.1f}m")

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(tables_dir / "phase6_ablation.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase6_ablation.csv'}")

    # === EXPERIMENT 3: RESIDUAL & CONSISTENCY ANALYSIS ===
    logger.info("\n=== EXPERIMENT 3: MULTI-CONSTRAINT CONSISTENCY ANALYSIS ===")
    res_e6_60 = traj_records[60]["res_e6"]
    nhc_nu_arr = np.array(res_e6_60["nhc_nu_list"])
    nhc_acc_rate = float(np.mean(res_e6_60["nhc_accepted_list"]) * 100.0)
    zupt_acc_rate = float(np.sum(res_e6_60["zupt_accepted_list"]))
    zaru_acc_rate = float(np.sum(res_e6_60["zaru_accepted_list"]))

    consistency_df = pd.DataFrame([{
        "nhc_accepted_pct": round(nhc_acc_rate, 2),
        "nhc_lat_residual_mean_mps": round(float(np.mean(nhc_nu_arr[:, 0])), 4),
        "nhc_lat_residual_rms_mps": round(float(np.sqrt(np.mean(nhc_nu_arr[:, 0]**2))), 4),
        "nhc_up_residual_mean_mps": round(float(np.mean(nhc_nu_arr[:, 1])), 4),
        "nhc_up_residual_rms_mps": round(float(np.sqrt(np.mean(nhc_nu_arr[:, 1]**2))), 4),
        "zupt_updates_applied": zupt_acc_rate,
        "zaru_updates_applied": zaru_acc_rate,
    }])
    consistency_df.to_csv(tables_dir / "phase6_consistency.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase6_consistency.csv'}")

    # === EXPERIMENT 4: RUNTIME PROFILING ===
    logger.info("\n=== EXPERIMENT 4: CPU RUNTIME LATENCY PROFILING ===")
    sample_dummy = np.zeros(12)
    sample_dummy[2] = -9.80665
    sample_dummy[0] = 5.0
    f_dummy = np.array([5.0, 0.0, 9.80665])
    omega_dummy = np.array([0.01, 0.01, 0.01])

    eskf_prof = Phase6ESKF()
    eskf_prof.initialize(init_p_abl, init_v_abl, init_q_abl)
    engine_prof = Phase6StreamingEngine(eskf=eskf_prof, ai_engine=ai_engine)
    engine_prof.initialize(init_p_abl, init_v_abl, init_q_abl)

    # Warmup
    for _ in range(50):
        engine_prof.step(sample_dummy, timestamp=1.0)

    n_iters = 500
    # INS predict latency
    t0 = time.perf_counter()
    for _ in range(n_iters):
        eskf_prof.predict(f_dummy, omega_dummy, 0.1)
    ins_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # NHC update latency
    t0 = time.perf_counter()
    for _ in range(n_iters):
        eskf_prof.update_nhc(0.0, 0.0)
    nhc_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # ZUPT & ZARU latency
    t0 = time.perf_counter()
    for _ in range(n_iters):
        eskf_prof.update_stationary_constraints(omega_dummy)
    stat_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Total streaming step latency
    t0 = time.perf_counter()
    for _ in range(n_iters):
        engine_prof.step(sample_dummy, timestamp=2.0)
    total_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    throughput_hz = 1000.0 / total_lat_ms
    runtime_dict = {
        "ins_prediction_latency_ms": round(ins_lat_ms, 4),
        "nhc_update_latency_ms": round(nhc_lat_ms, 4),
        "zupt_zaru_latency_ms": round(stat_lat_ms, 4),
        "total_step_latency_ms": round(total_lat_ms, 4),
        "throughput_hz": round(throughput_hz, 1),
        "cpu_headroom_pct": round((1.0 - (total_lat_ms / 100.0)) * 100.0, 2),
    }
    df_runtime = pd.DataFrame([runtime_dict])
    df_runtime.to_csv(tables_dir / "phase6_runtime.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase6_runtime.csv'}")
    logger.info(f"Runtime: Step={total_lat_ms:.3f} ms | Throughput={throughput_hz:.1f} Hz | "
                f"Headroom: {runtime_dict['cpu_headroom_pct']}% at 10 Hz")

    # === EXPERIMENT 5: GENERATING DIAGNOSTIC VISUALIZATION PLOTS ===
    logger.info("\n=== EXPERIMENT 5: GENERATING DIAGNOSTIC PLOTS ===")
    rec60 = traj_records[60]
    t_plot = rec60["time"] - rec60["time"][0]
    p_ref = rec60["ref_p"]
    p_e2 = rec60["res_e2"]["p_est"]
    p_e6 = rec60["res_e6"]["p_est"]

    # Plot 1: 60s Trajectory Comparison (Reference vs Phase 5 E2 vs Phase 6 E6)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(p_ref[:, 0], p_ref[:, 1], "k-", linewidth=2.5, label="GNSS Reference Ground Truth", zorder=4)
    ax.plot(p_e2[:, 0], p_e2[:, 1], "r--", linewidth=1.8, label=f"Phase 5 (E2: No Physics Constraints, Drift: {rec60['m_e2']['drift_pct']}%)")
    ax.plot(p_e6[:, 0], p_e6[:, 1], "g-", linewidth=2.2, label=f"Phase 6 (E6: Full Vehicle Physics ESKF, Drift: {rec60['m_e6']['drift_pct']}%)", zorder=5)
    ax.scatter(p_ref[0, 0], p_ref[0, 1], c="green", s=100, label="Blackout Entry", zorder=6)
    ax.scatter(p_ref[-1, 0], p_ref[-1, 1], c="black", marker="s", s=90, label="True Blackout End", zorder=6)
    ax.scatter(p_e6[-1, 0], p_e6[-1, 1], c="green", marker="^", s=90, label="Phase 6 End", zorder=6)
    ax.scatter(p_e2[-1, 0], p_e2[-1, 1], c="red", marker="x", s=90, label="Phase 5 End", zorder=6)
    ax.set_title("Phase 6: 60s Trajectory Comparison (Ground Truth vs Phase 5 vs Phase 6)", fontsize=12, fontweight="bold")
    ax.set_xlabel("East Position (m)")
    ax.set_ylabel("North Position (m)")
    ax.legend(fontsize=9)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot1_phase5_vs_phase6_trajectory_60s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Lateral Velocity Comparison Over Time
    fig, ax = plt.subplots(figsize=(10, 5))
    v_lat_e2 = rec60["res_e2"]["v_b_est"][:, 1]
    v_lat_e6 = rec60["res_e6"]["v_b_est"][:, 1]
    ax.plot(t_plot, v_lat_e2, "r--", label=f"Phase 5 E2 Lateral Velocity (RMSE: {rec60['c_e2']['lateral_vel_rmse_mps']:.2f} m/s)")
    ax.plot(t_plot, v_lat_e6, "g-", linewidth=1.8, label=f"Phase 6 E6 Lateral Velocity with NHC (RMSE: {rec60['c_e6']['lateral_vel_rmse_mps']:.2f} m/s)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title("Phase 6: Body-Frame Lateral Velocity (v_lat) Constraint Enforcement", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Since Outage Start (s)")
    ax.set_ylabel("Lateral Velocity v_lat (m/s)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot2_lateral_velocity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: 2D Horizontal Position Error Growth Over Time
    fig, ax = plt.subplots(figsize=(10, 5))
    err_e2 = np.sqrt((p_e2[:, 0] - p_ref[:, 0])**2 + (p_e2[:, 1] - p_ref[:, 1])**2)
    err_e6 = np.sqrt((p_e6[:, 0] - p_ref[:, 0])**2 + (p_e6[:, 1] - p_ref[:, 1])**2)
    ax.plot(t_plot, err_e2, "r--", linewidth=1.8, label=f"Phase 5 (E2: Max Error = {np.max(err_e2):.1f} m)")
    ax.plot(t_plot, err_e6, "g-", linewidth=2.0, label=f"Phase 6 (E6: Max Error = {np.max(err_e6):.1f} m)")
    ax.set_title("Phase 6: 2D Position Error Growth Over Time (60s Outage)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Since Outage Start (s)")
    ax.set_ylabel("2D Position Error (m)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot3_position_error_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Stationary Creep Suppression Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    is_stat_plot = rec60["res_e6"]["is_stat_list"]
    ax.plot(t_plot, err_e2, "r--", label="Phase 5 Error (Creeps during stop)")
    ax.plot(t_plot, err_e6, "g-", label="Phase 6 Error with ZUPT/ZARU (Zero Creep)")
    ax.fill_between(t_plot, 0, np.max(err_e2), where=is_stat_plot, color="yellow", alpha=0.25, label="Verified Vehicle Standstill (ZUPT Active)")
    ax.set_title("Phase 6: Stationary Position Creep Elimination via ZUPT", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot4_stationary_creep_suppression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: Yaw Angle Error Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    yaw_ref = np.degrees(rec60["ref_yaw"])
    yaw_e2 = np.degrees(rec60["res_e2"]["yaw_est"])
    yaw_e6 = np.degrees(rec60["res_e6"]["yaw_est"])
    # Wrapped error
    err_yaw_e2 = (yaw_e2 - yaw_ref + 180.0) % 360.0 - 180.0
    err_yaw_e6 = (yaw_e6 - yaw_ref + 180.0) % 360.0 - 180.0
    ax.plot(t_plot, np.abs(err_yaw_e2), "r--", label=f"Phase 5 Yaw Error (MAE: {np.mean(np.abs(err_yaw_e2)):.1f}°)")
    ax.plot(t_plot, np.abs(err_yaw_e6), "g-", linewidth=1.8, label=f"Phase 6 Yaw Error (MAE: {np.mean(np.abs(err_yaw_e6)):.1f}°)")
    ax.set_title("Phase 6: Heading (Yaw) Error Comparison Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Yaw Error (deg)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot5_yaw_error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 6: NHC Residuals and Adaptive Disturbance Confidence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(t_plot, nhc_nu_arr[:, 0], "b-", linewidth=1.0, label="Lateral NHC Residual v_lat (m/s)")
    ax1.plot(t_plot, nhc_nu_arr[:, 1], "c--", linewidth=1.0, label="Vertical NHC Residual v_up (m/s)")
    ax1.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
    ax1.set_title("Phase 6: NHC Innovation Residuals & Adaptive Confidence", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Residual (m/s)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    c_nhc_plot = rec60["res_e6"]["nhc_conf_list"]
    ax2.plot(t_plot, c_nhc_plot, "m-", linewidth=1.5, label="Causal NHC Confidence Score c_NHC")
    ax2.set_title("Adaptive Disturbance Weighting (Downweights during turns/shocks)", fontsize=10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Confidence [0, 1]")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "plot6_nhc_residuals_and_confidence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("================================================================================")
    logger.info("PHASE 6 MASTER EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("================================================================================")


def torch_load_weights(ckpt_path: Path) -> Dict[str, Any]:
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


if __name__ == "__main__":
    run_phase6_master_pipeline()
