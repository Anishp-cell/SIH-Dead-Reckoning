"""
Phase 7 Master Execution Pipeline:
Runs the comprehensive Phase 7 evaluation suite:
- Experiment 1: Phase 6 vs Phase 7 Multi-Duration Blackout Benchmark (10s, 30s, 60s, 120s)
- Experiment 2: M0 to M5 Component Ablation Study (60s outage)
- Experiment 3: Diverse Operational Scenarios (Cruising, Stop-and-Go, Sharp Turns, Long Straight, Intersections)
- Experiment 4: CPU Runtime Latency Profiling
- Experiment 5: Publication-Grade Research Plots
"""

import logging
import time
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_details.src.dataset_loader import DatasetLoader
from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase4.models import HeteroscedasticSpeedModel
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
import torch

def torch_load_weights(ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

from Data_details.src.phase7.map.osm_loader import OSMLoader
from Data_details.src.phase7.map.coordinate_mapper import CoordinateMapper
from Data_details.src.phase7.map.spatial_index import SpatialIndex
from Data_details.src.phase7.evaluation.map_matching_metrics import compute_phase7_metrics
from Data_details.src.phase7.evaluation.blackout_benchmarks import run_phase7_trajectory
from Data_details.src.phase7.evaluation.ablation import run_m0_to_m5_ablation
from Data_details.src.phase7.map.geometry import wrap_angle_rad

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7Pipeline")


def run_phase7_pipeline():
    logger.info("=================================================================")
    logger.info("STARTING PHASE 7 MASTER EVALUATION PIPELINE")
    logger.info("Offline OSM Map Matching & Road-Constrained Navigation")
    logger.info("=================================================================")

    out_dir = Path("Data_details/outputs/phase7")
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"
    for d in [tables_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load conditioned IMU dataset
    imu_csv = Path("Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv")
    if not imu_csv.exists():
        raise FileNotFoundError(f"Missing required conditioned IMU: {imu_csv}")

    logger.info(f"Loading preconditioned IMU signals from {imu_csv}...")
    df_imu = pd.read_csv(imu_csv)
    n_total = len(df_imu)
    time_s = df_imu["time_s"].values.astype(float)
    logger.info(f"Loaded {n_total} samples ({n_total*0.1/60:.2f} minutes).")

    # 2. Compute reference ground truth in local ENU
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

    # 4. Load offline OSM road database & construct SpatialIndex
    osm_json = Path("Data_details/data/osm/coventry_s1_osm.json")
    if not osm_json.exists():
        raise FileNotFoundError(f"Missing offline OSM road dataset: {osm_json}")

    logger.info("Loading offline OpenStreetMap database...")
    mapper = CoordinateMapper(
        lat0_deg=origin_info["lat0_deg"],
        lon0_deg=origin_info["lon0_deg"],
        alt0_m=origin_info["alt0_m"],
    )
    osm_loader = OSMLoader(coordinate_mapper=mapper)
    road_db = osm_loader.load_from_json(osm_json)
    db_summary = road_db.summary()
    logger.info(f"Loaded road network: {db_summary['total_segments']} segments across {db_summary['total_ways']} ways ({db_summary['total_road_length_km']} km).")

    logger.info("Constructing SpatialIndex over road network...")
    spatial_index = SpatialIndex(road_db, sampling_interval_m=10.0)
    logger.info(f"SpatialIndex ready: {spatial_index.total_indexed_points} points indexed ({spatial_index.memory_bytes/1024:.1f} KB).")

    # =================================================================
    # EXPERIMENT 1: PHASE 6 VS PHASE 7 MULTI-DURATION BLACKOUT BENCHMARK
    # =================================================================
    logger.info("\n=== EXPERIMENT 1: PHASE 6 (M0) VS PHASE 7 (M5) BENCHMARK ===")
    bo_start_time = 4600.0  # Exact Phase 5 & 6 evaluated start point (includes traffic standstill)
    start_idx = int(np.searchsorted(time_s, bo_start_time))
    durations_s = [10, 30, 60, 120]

    p6_vs_p7_results = []
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
        acc_cols = ["acc_fwd_veh", "acc_lat_veh", "acc_up_veh"] if "acc_fwd_veh" in df_slice.columns else ["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]
        acc_entry = df_slice[acc_cols].iloc[0].values
        acc_entry_corr = np.array([acc_entry[0], acc_entry[1], -acc_entry[2]], dtype=np.float64)
        init_q = initial_leveling_quaternion(acc_entry_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]))

        # Causal warm-start buffer pre-filling (t < t_0)
        warm_start_slice = df_imu.iloc[max(0, start_idx - 30):start_idx].copy()

        # Phase 6 Baseline: No map matching (M0)
        ai_engine.reset()
        res_p6 = run_phase7_trajectory(
            df_slice, init_p, init_v, init_q, road_db, spatial_index,
            ai_engine=ai_engine, enable_map_matching=False, enable_map_updates=False,
            warm_start_df=warm_start_slice,
        )
        m_p6 = compute_phase7_metrics(
            res_p6["p_est"], ref_p_slice, res_p6["v_est"], ref_v_slice, res_p6["yaw_est"], ref_yaw_slice,
            traveled_dist, res_p6["matched_ids"], res_p6["map_confidences"], res_p6["candidate_counts"],
            res_p6["cross_track_errors"], res_p6["map_ct_accepted"], res_p6["map_hd_accepted"],
            res_p6["ct_nis_list"], res_p6["hd_nis_list"],
        )
        p6_vs_p7_results.append({"duration_s": dur, "architecture": "Phase6_Baseline_M0", **m_p6})

        # Phase 7 Full: Map Matching + Soft ESKF Updates (M5)
        ai_engine.reset()
        res_p7 = run_phase7_trajectory(
            df_slice, init_p, init_v, init_q, road_db, spatial_index,
            ai_engine=ai_engine, enable_map_matching=True, enable_map_updates=True,
            enable_cross_track_update=True, enable_heading_update=True,
            warm_start_df=warm_start_slice,
        )
        m_p7 = compute_phase7_metrics(
            res_p7["p_est"], ref_p_slice, res_p7["v_est"], ref_v_slice, res_p7["yaw_est"], ref_yaw_slice,
            traveled_dist, res_p7["matched_ids"], res_p7["map_confidences"], res_p7["candidate_counts"],
            res_p7["cross_track_errors"], res_p7["map_ct_accepted"], res_p7["map_hd_accepted"],
            res_p7["ct_nis_list"], res_p7["hd_nis_list"],
        )
        p6_vs_p7_results.append({"duration_s": dur, "architecture": "Phase7_Full_M5", **m_p7})

        traj_records[dur] = {
            "res_p6": res_p6, "res_p7": res_p7,
            "ref_p": ref_p_slice, "ref_v": ref_v_slice, "ref_yaw": ref_yaw_slice,
            "time": df_slice["time_s"].values, "m_p6": m_p6, "m_p7": m_p7,
        }

        logger.info(f"[{dur:3d}s Outage] Traveled={traveled_dist:6.1f}m | "
                    f"Phase 6 Drift: {m_p6['drift_pct']:5.1f}% (Endpt: {m_p6['endpoint_error_m']:5.1f}m, CrossTrack: {m_p6['cross_track_rmse_m']:5.1f}m, YawRMSE: {m_p6['yaw_rmse_deg']:4.1f}deg) | "
                    f"Phase 7 Drift: {m_p7['drift_pct']:5.1f}% (Endpt: {m_p7['endpoint_error_m']:5.1f}m, CrossTrack: {m_p7['cross_track_rmse_m']:5.1f}m, YawRMSE: {m_p7['yaw_rmse_deg']:4.1f}deg)")

    df_p6_vs_p7 = pd.DataFrame(p6_vs_p7_results)
    df_p6_vs_p7.to_csv(tables_dir / "phase6_vs_phase7_comparison.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase6_vs_phase7_comparison.csv'}")

    # =================================================================
    # EXPERIMENT 2: STEP-BY-STEP M0 TO M5 COMPONENT ABLATION STUDY (60s)
    # =================================================================
    logger.info("\n=== EXPERIMENT 2: M0 TO M5 COMPONENT ABLATION STUDY (60s Outage) ===")
    dur_abl = 60
    end_idx_abl = int(np.searchsorted(time_s, bo_start_time + dur_abl))
    df_slice_abl = df_imu.iloc[start_idx:end_idx_abl].copy()
    ref_p_abl = ref_pos_3d[start_idx:end_idx_abl]
    ref_v_abl = ref_vel_3d[start_idx:end_idx_abl]
    ref_yaw_abl = ref_yaw_enu_rad[start_idx:end_idx_abl]
    traveled_dist_abl = float(enu_cumulative_distance(ref_p_abl[:, 0], ref_p_abl[:, 1])[-1])

    init_p_abl = ref_p_abl[0].copy()
    init_v_abl = ref_v_abl[0].copy()
    acc_cols_abl = ["acc_fwd_veh", "acc_lat_veh", "acc_up_veh"] if "acc_fwd_veh" in df_slice_abl.columns else ["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]
    acc_entry_abl = df_slice_abl[acc_cols_abl].iloc[0].values
    acc_entry_abl_corr = np.array([acc_entry_abl[0], acc_entry_abl[1], -acc_entry_abl[2]], dtype=np.float64)
    init_q_abl = initial_leveling_quaternion(acc_entry_abl_corr, initial_gps_heading_deg=float(gps_heading_deg[start_idx]))

    warm_start_abl = df_imu.iloc[max(0, start_idx - 30):start_idx].copy()
    df_ablation = run_m0_to_m5_ablation(
        df_slice=df_slice_abl,
        ref_pos_3d=ref_p_abl,
        ref_vel_3d=ref_v_abl,
        ref_yaw_enu_rad=ref_yaw_abl,
        traveled_dist=traveled_dist_abl,
        init_p=init_p_abl,
        init_v=init_v_abl,
        init_q=init_q_abl,
        road_db=road_db,
        spatial_index=spatial_index,
        ai_engine=ai_engine,
        warm_start_df=warm_start_abl,
    )
    df_ablation.to_csv(tables_dir / "phase7_ablation.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase7_ablation.csv'}")

    for _, row in df_ablation.iterrows():
        logger.info(f"[{row['configuration']}] Drift={row['drift_pct']:5.1f}% | Endpt={row['endpoint_error_m']:6.1f}m | "
                    f"CrossTrack={row['cross_track_rmse_m']:5.2f}m | YawRMSE={row['yaw_rmse_deg']:4.1f}deg | Conf={row['mean_map_confidence']:.2f}")

    # =================================================================
    # EXPERIMENT 3: HIGH-DYNAMIC & DIVERSE OPERATIONAL SCENARIOS
    # =================================================================
    logger.info("\n=== EXPERIMENT 3: HIGH-DYNAMIC OPERATIONAL SCENARIO BENCHMARKS ===")
    # Identify dedicated segments in the held-out test partition (t >= 4403.1s):
    # S1 test partition runs from 4403.1s to 5174.6s
    scenario_defs = [
        ("Stop_and_Go_Traffic", 4600.0, 60.0, "Standstill traffic followed by acceleration"),
        ("Steady_Cruising", 4750.0, 60.0, "Continuous medium-speed cruising (35-45 km/h)"),
        ("Sharp_Cornering_Turn", 4950.0, 50.0, "Sharp urban cornering turn onto cross-street"),
        ("Long_Straight_Highway", 5050.0, 60.0, "Extended straight road segment"),
    ]

    scenario_results = []
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

        scenario_results.append({
            "scenario": sc_name, "description": sc_desc, "traveled_dist_m": round(sc_dist, 1),
            "p6_drift_pct": m_p6_sc["drift_pct"], "p7_drift_pct": m_p7_sc["drift_pct"],
            "p6_cross_track_rmse_m": m_p6_sc["cross_track_rmse_m"], "p7_cross_track_rmse_m": m_p7_sc["cross_track_rmse_m"],
            "p6_yaw_rmse_deg": m_p6_sc["yaw_rmse_deg"], "p7_yaw_rmse_deg": m_p7_sc["yaw_rmse_deg"],
            "p7_map_confidence": m_p7_sc["mean_map_confidence"],
        })
        logger.info(f"[{sc_name:22s}] Traveled={sc_dist:5.1f}m | Drift: P6={m_p6_sc['drift_pct']:5.1f}% -> P7={m_p7_sc['drift_pct']:5.1f}% | "
                    f"CrossTrack: P6={m_p6_sc['cross_track_rmse_m']:5.1f}m -> P7={m_p7_sc['cross_track_rmse_m']:5.1f}m | "
                    f"YawRMSE: P6={m_p6_sc['yaw_rmse_deg']:4.1f}deg -> P7={m_p7_sc['yaw_rmse_deg']:4.1f}deg")

    df_scenarios = pd.DataFrame(scenario_results)
    df_scenarios.to_csv(tables_dir / "phase7_scenarios.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase7_scenarios.csv'}")

    # =================================================================
    # EXPERIMENT 4: CPU RUNTIME LATENCY PROFILING
    # =================================================================
    logger.info("\n=== EXPERIMENT 4: CPU RUNTIME LATENCY PROFILING ===")
    sample_dummy = np.zeros(12)
    sample_dummy[2] = -9.80665
    sample_dummy[0] = 5.0

    from Data_details.src.phase7.matching.candidate_generator import CandidateGenerator
    from Data_details.src.phase7.matching.candidate_scoring import CandidateScorer
    from Data_details.src.phase7.matching.topology import RoadTopologyGraph, TransitionModel
    from Data_details.src.phase7.matching.temporal_matcher import TemporalMapMatcher
    from Data_details.src.phase7.matching.map_measurement import MapMeasurementModel
    from Data_details.src.phase7.matching.map_update import apply_map_cross_track_update, apply_map_heading_update
    from Data_details.src.phase7.streaming import Phase7StreamingEngine
    from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF

    gen_prof = CandidateGenerator(spatial_index, k_sigma=3.0, d_min_m=20.0)
    scorer_prof = CandidateScorer()
    graph_prof = RoadTopologyGraph(road_db)
    trans_prof = TransitionModel(graph_prof)
    matcher_prof = TemporalMapMatcher(gen_prof, scorer_prof, trans_prof)
    map_model_prof = MapMeasurementModel()

    n_iters = 500
    pos_dummy = np.array([-4200.0, 100.0])
    cov_dummy = np.diag([9.0, 9.0])

    # 1. Spatial Candidate Query
    t0 = time.perf_counter()
    for _ in range(n_iters):
        gen_prof.generate_candidates(pos_dummy, yaw_enu_rad=0.0, pos_cov_2x2=cov_dummy)
    cand_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # 2. Complete Map Matching Step
    t0 = time.perf_counter()
    for _ in range(n_iters):
        match_res = matcher_prof.step(1.0, pos_dummy, yaw_enu_rad=0.0, pos_cov_2x2=cov_dummy, v_forward_mps=10.0)
    match_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # 3. Soft Map Kalman Update
    P_dummy = np.eye(15) * 4.0
    state_dummy = Phase6ESKF().state
    t0 = time.perf_counter()
    for _ in range(n_iters):
        apply_map_cross_track_update(state_dummy, P_dummy, match_res, map_model_prof)
        apply_map_heading_update(state_dummy, P_dummy, match_res, map_model_prof, v_forward_mps=10.0)
    update_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # 4. Total Phase 7 Streaming Cycle (IMU Predict + AI Speed + NHC/ZUPT + Map Match + Map Updates)
    engine_prof = Phase7StreamingEngine(
        eskf=Phase6ESKF(),
        ai_engine=ai_engine,
        map_matcher=matcher_prof,
        map_model=map_model_prof,
        enable_map_updates=True,
    )
    engine_prof.initialize(init_p_abl, init_v_abl, init_q_abl)
    # Warmup
    for _ in range(30):
        engine_prof.step(sample_dummy, timestamp=1.0)

    t0 = time.perf_counter()
    for i in range(n_iters):
        engine_prof.step(sample_dummy, timestamp=1.0 + i*0.1)
    total_lat_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    runtime_df = pd.DataFrame([{
        "spatial_candidate_query_ms": round(cand_lat_ms, 3),
        "temporal_map_matching_ms": round(match_lat_ms, 3),
        "map_kalman_updates_ms": round(update_lat_ms, 3),
        "total_phase7_step_ms": round(total_lat_ms, 3),
        "target_10hz_budget_ms": 100.0,
        "phase6_total_step_ms": 1.63,
        "margin_factor": round(100.0 / total_lat_ms, 1),
    }])
    runtime_df.to_csv(tables_dir / "phase7_runtime.csv", index=False)
    logger.info(f"Saved {tables_dir / 'phase7_runtime.csv'}")
    logger.info(f"Latency Profile: MapQuery={cand_lat_ms:.3f}ms | MapMatch={match_lat_ms:.3f}ms | MapUpdate={update_lat_ms:.3f}ms | TotalStep={total_lat_ms:.3f}ms (Budget=100ms)")

    # =================================================================
    # EXPERIMENT 5: PUBLICATION-GRADE RESEARCH PLOTS
    # =================================================================
    logger.info("\n=== EXPERIMENT 5: GENERATING DIAGNOSTIC & TRAJECTORY PLOTS ===")

    # Plot 1: 2D Trajectory Comparison (60s Outage) with Nearby Road Network
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    rec60 = traj_records[60]
    p_ref60 = rec60["ref_p"]
    p_p6_60 = rec60["res_p6"]["p_est"]
    p_p7_60 = rec60["res_p7"]["p_est"]

    # Plot nearby OSM road segments
    bbox_e = (min(p_ref60[:, 0]) - 50, max(p_ref60[:, 0]) + 50)
    bbox_n = (min(p_ref60[:, 1]) - 50, max(p_ref60[:, 1]) + 50)
    for seg in road_db.segments.values():
        e1, n1 = seg.p_start_enu[0], seg.p_start_enu[1]
        e2, n2 = seg.p_end_enu[0], seg.p_end_enu[1]
        if (bbox_e[0] <= e1 <= bbox_e[1] or bbox_e[0] <= e2 <= bbox_e[1]) and (bbox_n[0] <= n1 <= bbox_n[1] or bbox_n[0] <= n2 <= bbox_n[1]):
            ax.plot([e1, e2], [n1, n2], color="#cbd5e1", linewidth=2.0, zorder=1)

    ax.plot(p_ref60[:, 0], p_ref60[:, 1], "k-", linewidth=2.5, label="Ground Truth (GNSS/VBOX Reference)", zorder=3)
    ax.plot(p_p6_60[:, 0], p_p6_60[:, 1], "r--", linewidth=2.0, label=f"Phase 6 Vehicle Physics (Drift={rec60['m_p6']['drift_pct']}%)", zorder=4)
    ax.plot(p_p7_60[:, 0], p_p7_60[:, 1], "b-", linewidth=2.2, label=f"Phase 7 Map Constrained (Drift={rec60['m_p7']['drift_pct']}%)", zorder=5)

    ax.plot(p_ref60[0, 0], p_ref60[0, 1], "go", markersize=8, label="Outage Start (t=4600s)", zorder=6)
    ax.plot(p_ref60[-1, 0], p_ref60[-1, 1], "ks", markersize=7, label="True End", zorder=6)
    ax.plot(p_p7_60[-1, 0], p_p7_60[-1, 1], "b*", markersize=10, label="Phase 7 End", zorder=6)

    ax.set_title("Phase 7: Road-Constrained Trajectory vs Phase 6 Baseline (60s Blackout)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Local East [m]", fontsize=11)
    ax.set_ylabel("Local North [m]", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "plot1_phase6_vs_phase7_trajectory_60s.png")
    plt.close()

    # Plot 2: Cross-Track Error Over Time
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    t_rel = rec60["time"] - rec60["time"][0]
    p_err_p6 = rec60["res_p6"]["p_est"][:, :2] - p_ref60[:, :2]
    p_err_p7 = rec60["res_p7"]["p_est"][:, :2] - p_ref60[:, :2]

    # Normal projection
    yaw_ref = rec60["ref_yaw"]
    norm_vecs = np.column_stack([-np.sin(yaw_ref), np.cos(yaw_ref)])
    ct_p6 = np.abs(np.sum(p_err_p6 * norm_vecs, axis=1))
    ct_p7 = np.abs(np.sum(p_err_p7 * norm_vecs, axis=1))

    ax.plot(t_rel, ct_p6, "r--", linewidth=2.0, label=f"Phase 6 Cross-Track (RMSE={rec60['m_p6']['cross_track_rmse_m']:.2f}m)")
    ax.plot(t_rel, ct_p7, "b-", linewidth=2.0, label=f"Phase 7 Cross-Track (RMSE={rec60['m_p7']['cross_track_rmse_m']:.2f}m)")
    ax.set_title("Cross-Track Position Error Over Time (60s Blackout)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Outage Elapsed Time [s]", fontsize=10)
    ax.set_ylabel("Cross-Track Error [m]", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "plot2_cross_track_error_vs_time.png")
    plt.close()

    # Plot 3: Heading Error Over Time
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    yaw_err_p6 = np.degrees(np.abs(np.array([wrap_angle_rad(ye - yr) for ye, yr in zip(rec60['res_p6']['yaw_est'], yaw_ref)])))
    yaw_err_p7 = np.degrees(np.abs(np.array([wrap_angle_rad(ye - yr) for ye, yr in zip(rec60['res_p7']['yaw_est'], yaw_ref)])))

    ax.plot(t_rel, yaw_err_p6, "r--", linewidth=2.0, label=f"Phase 6 Heading Error (RMSE={rec60['m_p6']['yaw_rmse_deg']:.1f}deg)")
    ax.plot(t_rel, yaw_err_p7, "b-", linewidth=2.0, label=f"Phase 7 Heading Error (RMSE={rec60['m_p7']['yaw_rmse_deg']:.1f}deg)")
    ax.set_title("Absolute Heading Yaw Error Comparison (60s Blackout)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Outage Elapsed Time [s]", fontsize=10)
    ax.set_ylabel("Absolute Yaw Error [deg]", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "plot3_yaw_error_comparison.png")
    plt.close()

    # Plot 4: Map Confidence & Candidate Count
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(t_rel, rec60["res_p7"]["map_confidences"], "b-", linewidth=2.0, label="Map Confidence c_map")
    ax1.axhline(0.35, color="gray", linestyle=":", label="Confidence Gate Threshold (0.35)")
    ax2.plot(t_rel, rec60["res_p7"]["candidate_counts"], "g--", linewidth=1.5, alpha=0.7, label="Candidate Count")
    ax1.set_title("Map Matcher Confidence & Multi-Hypothesis Density (60s Outage)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Outage Elapsed Time [s]", fontsize=10)
    ax1.set_ylabel("Map Confidence c_map in [0, 1]", color="b", fontsize=10)
    ax2.set_ylabel("Number of Road Candidates", color="g", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / "plot4_map_confidence_and_candidates.png")
    plt.close()

    # Plot 5: 120s Extended Blackout Trajectory
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    rec120 = traj_records[120]
    p_ref120 = rec120["ref_p"]
    p_p6_120 = rec120["res_p6"]["p_est"]
    p_p7_120 = rec120["res_p7"]["p_est"]

    bbox_e120 = (min(p_ref120[:, 0]) - 80, max(p_ref120[:, 0]) + 80)
    bbox_n120 = (min(p_ref120[:, 1]) - 80, max(p_ref120[:, 1]) + 80)
    for seg in road_db.segments.values():
        e1, n1 = seg.p_start_enu[0], seg.p_start_enu[1]
        e2, n2 = seg.p_end_enu[0], seg.p_end_enu[1]
        if (bbox_e120[0] <= e1 <= bbox_e120[1] or bbox_e120[0] <= e2 <= bbox_e120[1]) and (bbox_n120[0] <= n1 <= bbox_n120[1] or bbox_n120[0] <= n2 <= bbox_n120[1]):
            ax.plot([e1, e2], [n1, n2], color="#cbd5e1", linewidth=2.0, zorder=1)

    ax.plot(p_ref120[:, 0], p_ref120[:, 1], "k-", linewidth=2.5, label="Ground Truth Reference", zorder=3)
    ax.plot(p_p6_120[:, 0], p_p6_120[:, 1], "r--", linewidth=2.0, label=f"Phase 6 Baseline (Drift={rec120['m_p6']['drift_pct']}%)", zorder=4)
    ax.plot(p_p7_120[:, 0], p_p7_120[:, 1], "b-", linewidth=2.2, label=f"Phase 7 Road-Constrained (Drift={rec120['m_p7']['drift_pct']}%)", zorder=5)
    ax.set_title("Phase 7: Long-Duration 120s Blackout Road-Constrained Navigation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Local East [m]", fontsize=11)
    ax.set_ylabel("Local North [m]", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / "plot5_long_duration_120s_trajectory.png")
    plt.close()

    logger.info(f"Saved all 5 diagnostic plots to {plots_dir}")
    logger.info("=================================================================")
    logger.info("PHASE 7 MASTER EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_phase7_pipeline()
