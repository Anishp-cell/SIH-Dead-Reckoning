"""
Pre-Phase-8 Master Re-Benchmarking Engine:
Executes standardized evaluation of Phase 5, Phase 6, and Phase 7 navigation stacks across:
- Modes: Mode A (WARM_START_OPERATIONAL), Mode B (WARM_START_IDEAL), Mode C (COLD_START)
- Outage Durations: 10s, 30s, 60s, 120s
- Diverse Operational Scenarios: Stop-and-Go, Cruising, Cornering, Highway Straight
Computes all 15 scientific benchmark metrics with zero future lookahead.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from Data_details.src.dataset_loader import DatasetLoader
from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.phase4.models import HeteroscedasticSpeedModel
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase7.map.map_database import RoadMapDatabase
from Data_details.src.phase7.map.spatial_index import SpatialIndex
from Data_details.src.phase7.map.osm_loader import OSMLoader
from Data_details.src.phase7.map.coordinate_mapper import CoordinateMapper
from Data_details.src.phase7.evaluation.blackout_benchmarks import run_phase7_trajectory
from Data_details.src.phase7.evaluation.map_matching_metrics import compute_phase7_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Rebenchmark")


def torch_load_weights(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def main():
    logger.info("==================================================================")
    logger.info("PRE-PHASE-8 RE-BENCHMARKING & VERIFICATION SUITE")
    logger.info("Modes: Operational (A), Ideal Warm (B), Cold Start (C)")
    logger.info("==================================================================")

    out_dir = Path("Data_details/outputs/correction_pass")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load preconditioned IMU signals
    imu_csv = Path("Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv")
    if not imu_csv.exists():
        raise FileNotFoundError(f"Missing required conditioned IMU: {imu_csv}")
    df_imu = pd.read_csv(imu_csv)
    n_total = len(df_imu)
    time_s = df_imu["time_s"].values.astype(float)
    logger.info(f"Loaded {n_total} samples ({n_total * 0.1 / 60:.2f} minutes).")

    # 2. Reference ground truth in local ENU
    lat = df_imu["gps_lat"].values.astype(float)
    lon = df_imu["gps_lon"].values.astype(float)
    alt = df_imu["gps_alt"].values.astype(float)
    ref_east, ref_north, ref_up, origin_info = geodetic_to_enu(lat, lon, alt)
    ref_pos_3d = np.column_stack([ref_east, ref_north, ref_up])

    loader = DatasetLoader()
    _, v_df = loader.load_sequence("S-Dataset/S-S1.csv", "V-Dataset/V-S1.csv")
    v_speed_gt = v_df["v_speed_mps"].values[:n_total].astype(np.float64)

    gps_heading_deg = df_imu["gps_heading_deg"].values.astype(float)
    ref_yaw_enu_rad = np.array([geographic_to_enu_yaw_rad(np.radians(h)) for h in gps_heading_deg])
    ref_vel_east = v_speed_gt * np.cos(ref_yaw_enu_rad)
    ref_vel_north = v_speed_gt * np.sin(ref_yaw_enu_rad)
    ref_vel_3d = np.column_stack([ref_vel_east, ref_vel_north, np.zeros(n_total)])

    # 3. Phase 4 AI model
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

    # 4. OSM Road Database & SpatialIndex
    osm_json = Path("Data_details/data/osm/coventry_s1_osm.json")
    mapper = CoordinateMapper(
        lat0_deg=origin_info["lat0_deg"],
        lon0_deg=origin_info["lon0_deg"],
        alt0_m=origin_info["alt0_m"],
    )
    osm_loader = OSMLoader(coordinate_mapper=mapper)
    road_db = osm_loader.load_from_json(osm_json)
    spatial_index = SpatialIndex(road_db, sampling_interval_m=10.0)

    # Benchmarking configurations
    durations_s = [10, 30, 60, 120]
    bo_start_time = 4600.0
    start_idx = int(np.searchsorted(time_s, bo_start_time))

    benchmark_records = []
    mode_comparison_records = []

    modes = [
        ("MODE_B_WARM_START_IDEAL", True, False),
        ("MODE_C_COLD_START", False, False),
        ("MODE_A_WARM_START_OPERATIONAL", True, True),
    ]

    for mode_name, is_warm, is_operational in modes:
        logger.info(f"\n=======================================================")
        logger.info(f"EVALUATING BENCHMARK: {mode_name}")
        logger.info(f"=======================================================")

        for dur in durations_s:
            end_time = bo_start_time + dur
            end_idx = int(np.searchsorted(time_s, end_time))
            df_slice = df_imu.iloc[start_idx:end_idx].copy()

            ref_p_slice = ref_pos_3d[start_idx:end_idx]
            ref_v_slice = ref_vel_3d[start_idx:end_idx]
            ref_yaw_slice = ref_yaw_enu_rad[start_idx:end_idx]
            cum_dist = enu_cumulative_distance(ref_p_slice[:, 0], ref_p_slice[:, 1])
            traveled_dist = float(cum_dist[-1])

            # Initial entry state
            if is_operational:
                # Operational GNSS hand-off: add realistic GNSS noise offset at t_0 (0.8m position, 0.05m/s vel, 0.4 deg yaw)
                np.random.seed(42 + dur)
                init_p = ref_p_slice[0].copy() + np.random.normal(0, 0.5, 3)
                init_v = ref_v_slice[0].copy() + np.random.normal(0, 0.05, 3)
                heading_err = np.radians(np.random.normal(0, 0.4))
                acc_cols = ["acc_fwd_veh", "acc_lat_veh", "acc_up_veh"] if "acc_fwd_veh" in df_slice.columns else ["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]
                acc_entry = df_slice[acc_cols].iloc[0].values
                acc_entry_corr = np.array([acc_entry[0], acc_entry[1], -acc_entry[2]], dtype=np.float64)
                init_q = initial_leveling_quaternion(acc_entry_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]) + np.degrees(heading_err))
            else:
                init_p = ref_p_slice[0].copy()
                init_v = ref_v_slice[0].copy()
                acc_cols = ["acc_fwd_veh", "acc_lat_veh", "acc_up_veh"] if "acc_fwd_veh" in df_slice.columns else ["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]
                acc_entry = df_slice[acc_cols].iloc[0].values
                acc_entry_corr = np.array([acc_entry[0], acc_entry[1], -acc_entry[2]], dtype=np.float64)
                init_q = initial_leveling_quaternion(acc_entry_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]))

            warm_df = df_imu.iloc[max(0, start_idx - 30):start_idx].copy() if is_warm else None

            # 1. Phase 5 (ESKF + AI Speed only)
            ai_engine.reset()
            res_p5 = run_phase7_trajectory(
                df_slice, init_p, init_v, init_q, road_db, spatial_index,
                ai_engine=ai_engine, enable_nhc=False, enable_zupt=False, enable_zaru=False,
                enable_map_matching=False, enable_map_updates=False,
                warm_start_df=warm_df,
            )
            m_p5 = compute_phase7_metrics(
                res_p5["p_est"], ref_p_slice, res_p5["v_est"], ref_v_slice, res_p5["yaw_est"], ref_yaw_slice,
                traveled_dist, res_p5["matched_ids"], res_p5["map_confidences"], res_p5["candidate_counts"],
                res_p5["cross_track_errors"], res_p5["map_ct_accepted"], res_p5["map_hd_accepted"],
                res_p5["ct_nis_list"], res_p5["hd_nis_list"],
            )

            # 2. Phase 6 (Vehicle Physics / NHC / ZUPT / ZARU)
            ai_engine.reset()
            res_p6 = run_phase7_trajectory(
                df_slice, init_p, init_v, init_q, road_db, spatial_index,
                ai_engine=ai_engine, enable_nhc=True, enable_zupt=True, enable_zaru=True,
                enable_map_matching=False, enable_map_updates=False,
                warm_start_df=warm_df,
            )
            m_p6 = compute_phase7_metrics(
                res_p6["p_est"], ref_p_slice, res_p6["v_est"], ref_v_slice, res_p6["yaw_est"], ref_yaw_slice,
                traveled_dist, res_p6["matched_ids"], res_p6["map_confidences"], res_p6["candidate_counts"],
                res_p6["cross_track_errors"], res_p6["map_ct_accepted"], res_p6["map_hd_accepted"],
                res_p6["ct_nis_list"], res_p6["hd_nis_list"],
            )

            # 3. Phase 7 (Map Constrained / M5)
            ai_engine.reset()
            res_p7 = run_phase7_trajectory(
                df_slice, init_p, init_v, init_q, road_db, spatial_index,
                ai_engine=ai_engine, enable_nhc=True, enable_zupt=True, enable_zaru=True,
                enable_map_matching=True, enable_map_updates=True,
                enable_cross_track_update=True, enable_heading_update=True,
                warm_start_df=warm_df,
            )
            m_p7 = compute_phase7_metrics(
                res_p7["p_est"], ref_p_slice, res_p7["v_est"], ref_v_slice, res_p7["yaw_est"], ref_yaw_slice,
                traveled_dist, res_p7["matched_ids"], res_p7["map_confidences"], res_p7["candidate_counts"],
                res_p7["cross_track_errors"], res_p7["map_ct_accepted"], res_p7["map_hd_accepted"],
                res_p7["ct_nis_list"], res_p7["hd_nis_list"],
            )

            for arch, m in [("Phase_5_ESKF", m_p5), ("Phase_6_Physics", m_p6), ("Phase_7_Map_Constrained", m_p7)]:
                benchmark_records.append({
                    "mode": mode_name,
                    "duration_s": dur,
                    "architecture": arch,
                    **m,
                })

            mode_comparison_records.append({
                "mode": mode_name,
                "duration_s": dur,
                "traveled_dist_m": round(traveled_dist, 1),
                "p5_drift_pct": m_p5["drift_pct"],
                "p6_drift_pct": m_p6["drift_pct"],
                "p7_drift_pct": m_p7["drift_pct"],
                "p5_endpt_m": m_p5["endpoint_error_m"],
                "p6_endpt_m": m_p6["endpoint_error_m"],
                "p7_endpt_m": m_p7["endpoint_error_m"],
                "p5_cross_track_m": m_p5["cross_track_rmse_m"],
                "p6_cross_track_m": m_p6["cross_track_rmse_m"],
                "p7_cross_track_m": m_p7["cross_track_rmse_m"],
                "p5_yaw_rmse_deg": m_p5["yaw_rmse_deg"],
                "p6_yaw_rmse_deg": m_p6["yaw_rmse_deg"],
                "p7_yaw_rmse_deg": m_p7["yaw_rmse_deg"],
            })

            logger.info(f"[{dur:3d}s] Dist={traveled_dist:5.1f}m | Drift: P5={m_p5['drift_pct']:5.1f}%, P6={m_p6['drift_pct']:5.1f}%, P7={m_p7['drift_pct']:5.1f}% | "
                        f"Endpt: P5={m_p5['endpoint_error_m']:5.1f}m, P6={m_p6['endpoint_error_m']:5.1f}m, P7={m_p7['endpoint_error_m']:5.1f}m | "
                        f"CrossTrack: P6={m_p6['cross_track_rmse_m']:5.1f}m, P7={m_p7['cross_track_rmse_m']:5.1f}m")

    df_benchmarks = pd.DataFrame(benchmark_records)
    df_benchmarks.to_csv(out_dir / "benchmark_results.csv", index=False)
    logger.info(f"Saved {out_dir / 'benchmark_results.csv'}")

    df_modes = pd.DataFrame(mode_comparison_records)
    df_modes.to_csv(out_dir / "mode_comparison.csv", index=False)
    logger.info(f"Saved {out_dir / 'mode_comparison.csv'}")

    # =================================================================
    # SCENARIOS BENCHMARK (Mode B Warm Start)
    # =================================================================
    logger.info("\n=======================================================")
    logger.info("EVALUATING DIVERSE OPERATIONAL SCENARIOS (60s)")
    logger.info("=======================================================")
    scenario_defs = [
        ("Stop_and_Go_Traffic", 4600.0, 60.0, "Standstill traffic followed by acceleration"),
        ("Steady_Cruising", 4750.0, 60.0, "Continuous medium-speed cruising (35-45 km/h)"),
        ("Sharp_Cornering_Turn", 4950.0, 50.0, "Sharp urban cornering turn onto cross-street"),
        ("Long_Straight_Highway", 5050.0, 60.0, "Extended straight road segment"),
    ]

    sc_records = []
    for sc_name, sc_start, sc_dur, sc_desc in scenario_defs:
        idx_s = int(np.searchsorted(time_s, sc_start))
        idx_e = int(np.searchsorted(time_s, sc_start + sc_dur))
        if idx_e > n_total:
            continue
        df_sc = df_imu.iloc[idx_s:idx_e].copy()
        ref_p_sc = ref_pos_3d[idx_s:idx_e]
        ref_v_sc = ref_vel_3d[idx_s:idx_e]
        ref_yaw_sc = ref_yaw_enu_rad[idx_s:idx_e]
        cum_sc = enu_cumulative_distance(ref_p_sc[:, 0], ref_p_sc[:, 1])
        sc_dist = float(cum_sc[-1])

        init_p_sc = ref_p_sc[0].copy()
        init_v_sc = ref_v_sc[0].copy()
        acc_cols_sc = ["acc_fwd_veh", "acc_lat_veh", "acc_up_veh"] if "acc_fwd_veh" in df_sc.columns else ["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]
        acc_sc = df_sc[acc_cols_sc].iloc[0].values
        acc_sc_corr = np.array([acc_sc[0], acc_sc[1], -acc_sc[2]], dtype=np.float64)
        init_q_sc = initial_leveling_quaternion(acc_sc_corr, initial_gps_heading_deg=float(gps_heading_deg[idx_s]))

        warm_start_sc = df_imu.iloc[max(0, idx_s - 30):idx_s].copy()

        # Run Phase 6
        ai_engine.reset()
        res_p6_sc = run_phase7_trajectory(
            df_sc, init_p_sc, init_v_sc, init_q_sc, road_db, spatial_index,
            ai_engine=ai_engine, enable_map_matching=False, enable_map_updates=False,
            warm_start_df=warm_start_sc,
        )
        m_p6_sc = compute_phase7_metrics(
            res_p6_sc["p_est"], ref_p_sc, res_p6_sc["v_est"], ref_v_sc, res_p6_sc["yaw_est"], ref_yaw_sc,
            sc_dist, res_p6_sc["matched_ids"], res_p6_sc["map_confidences"], res_p6_sc["candidate_counts"],
            res_p6_sc["cross_track_errors"], res_p6_sc["map_ct_accepted"], res_p6_sc["map_hd_accepted"],
            res_p6_sc["ct_nis_list"], res_p6_sc["hd_nis_list"],
        )

        # Run Phase 7
        ai_engine.reset()
        res_p7_sc = run_phase7_trajectory(
            df_sc, init_p_sc, init_v_sc, init_q_sc, road_db, spatial_index,
            ai_engine=ai_engine, enable_map_matching=True, enable_map_updates=True,
            warm_start_df=warm_start_sc,
        )
        m_p7_sc = compute_phase7_metrics(
            res_p7_sc["p_est"], ref_p_sc, res_p7_sc["v_est"], ref_v_sc, res_p7_sc["yaw_est"], ref_yaw_sc,
            sc_dist, res_p7_sc["matched_ids"], res_p7_sc["map_confidences"], res_p7_sc["candidate_counts"],
            res_p7_sc["cross_track_errors"], res_p7_sc["map_ct_accepted"], res_p7_sc["map_hd_accepted"],
            res_p7_sc["ct_nis_list"], res_p7_sc["hd_nis_list"],
        )

        sc_records.append({
            "scenario": sc_name,
            "description": sc_desc,
            "duration_s": sc_dur,
            "traveled_dist_m": round(sc_dist, 1),
            "p6_drift_pct": m_p6_sc["drift_pct"],
            "p7_drift_pct": m_p7_sc["drift_pct"],
            "p6_endpoint_error_m": m_p6_sc["endpoint_error_m"],
            "p7_endpoint_error_m": m_p7_sc["endpoint_error_m"],
            "p6_cross_track_rmse_m": m_p6_sc["cross_track_rmse_m"],
            "p7_cross_track_rmse_m": m_p7_sc["cross_track_rmse_m"],
            "p6_along_track_rmse_m": m_p6_sc["along_track_rmse_m"],
            "p7_along_track_rmse_m": m_p7_sc["along_track_rmse_m"],
            "p6_yaw_rmse_deg": m_p6_sc["yaw_rmse_deg"],
            "p7_yaw_rmse_deg": m_p7_sc["yaw_rmse_deg"],
            "p7_map_confidence": m_p7_sc["mean_map_confidence"],
            "p7_ct_accepted_rate_pct": m_p7_sc["map_ct_accepted_pct"],
            "p7_hd_accepted_rate_pct": m_p7_sc["map_hd_accepted_pct"],
        })
        logger.info(f"[{sc_name:22s}] Traveled={sc_dist:5.1f}m | Drift: P6={m_p6_sc['drift_pct']:5.1f}% -> P7={m_p7_sc['drift_pct']:5.1f}% | "
                    f"Endpt: P6={m_p6_sc['endpoint_error_m']:5.1f}m -> P7={m_p7_sc['endpoint_error_m']:5.1f}m | "
                    f"CrossTrack: P6={m_p6_sc['cross_track_rmse_m']:5.1f}m -> P7={m_p7_sc['cross_track_rmse_m']:5.1f}m")

    df_scenarios = pd.DataFrame(sc_records)
    df_scenarios.to_csv(out_dir / "scenario_comparison.csv", index=False)
    logger.info(f"Saved {out_dir / 'scenario_comparison.csv'}")

    logger.info("\nRE-BENCHMARKING COMPLETE.")


if __name__ == "__main__":
    main()
