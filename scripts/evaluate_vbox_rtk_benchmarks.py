"""
Evaluates Phase 8 Dead Reckoning & Map-Constrained Spline Odometry against
the True Racelogic VBOX RTK 100 Hz Ground Truth (Coventry S1).
Generates authoritative benchmark tables verifying <10% drift compliance for ISRO SIH26168.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from Data_details.src.dataset_loader import DatasetLoader
from Data_details.src.coordinate_transform import geodetic_to_enu, enu_cumulative_distance
from Data_details.src.phase5.core.frames import initial_leveling_quaternion, geographic_to_enu_yaw_rad
from Data_details.src.phase4.models import HeteroscedasticSpeedModel
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from Data_details.src.phase7.map.osm_loader import OSMLoader
from Data_details.src.phase7.map.coordinate_mapper import CoordinateMapper
from Data_details.src.phase7.map.spatial_index import SpatialIndex
from Data_details.src.phase7.matching.candidate_generator import CandidateGenerator
from Data_details.src.phase7.matching.candidate_scoring import CandidateScorer
from Data_details.src.phase7.matching.topology import RoadTopologyGraph, TransitionModel
from Data_details.src.phase7.matching.temporal_matcher import TemporalMapMatcher
from Data_details.src.phase7.matching.map_measurement import MapMeasurementModel
from Data_details.src.phase7.matching.spline_odometry import RoadSplineOdometry
from Data_details.src.phase8.streaming import Phase8StreamingEngine
from Data_details.src.phase8.core.phase8_eskf import Phase8ESKF
from Data_details.src.phase8.gnss.coordinate import enu_to_geodetic
from Data_details.src.phase8.evaluation.recovery_metrics import compute_road_aligned_errors


def main():
    print("Executing True VBOX RTK Ground Truth Re-Benchmark...")

    loader = DatasetLoader()
    s_df, v_df = loader.load_sequence("S-Dataset/S-S1.csv", "V-Dataset/V-S1.csv")

    imu_csv = workspace_root / "Data_details" / "outputs" / "phase3" / "processed" / "s1_filtered_causal_imu.csv"
    df_imu = pd.read_csv(imu_csv)
    n_samples = len(df_imu)
    time_s = df_imu["time_s"].values.astype(float)

    # 1. Ground Truth from Racelogic VBOX RTK
    v_lat = v_df["v_gps_lat"].values[:n_samples].astype(float)
    v_lon = v_df["v_gps_lon"].values[:n_samples].astype(float)
    v_alt = v_df["v_gps_height_km"].values[:n_samples].astype(float)  # Already in meters (approx 110m)

    ref_east, ref_north, ref_up, origin_info = geodetic_to_enu(v_lat, v_lon, v_alt)
    ref_pos_vbox = np.column_stack([ref_east, ref_north, ref_up])

    v_speed_gt = v_df["v_speed_mps"].values[:n_samples].astype(np.float64)
    v_heading_deg = v_df["v_gps_heading_deg"].values[:n_samples].astype(float)
    ref_yaw_rad = np.array([geographic_to_enu_yaw_rad(np.radians(h)) for h in v_heading_deg])
    ref_vel_vbox = np.column_stack([
        v_speed_gt * np.cos(ref_yaw_rad),
        v_speed_gt * np.sin(ref_yaw_rad),
        np.zeros(n_samples),
    ])

    # 2. Load Neural Speed Model
    model_ckpt = workspace_root / "Data_details" / "outputs" / "phase4" / "models" / "uncertainty" / "best_model.pt"
    norm_json = workspace_root / "Data_details" / "outputs" / "phase4" / "models" / "normalization.json"
    model = HeteroscedasticSpeedModel(in_channels=12)
    model.load_state_dict(torch.load(model_ckpt, map_location="cpu", weights_only=True))
    model.eval()

    feature_cols = [
        "acc_fwd_veh", "acc_lat_veh", "acc_up_veh",
        "gyro_roll_veh", "gyro_pitch_veh", "gyro_yaw_veh",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    for idx_c, col_name in enumerate(feature_cols):
        if col_name not in df_imu.columns:
            fallback = col_name + "_filtered"
            if fallback in df_imu.columns:
                feature_cols[idx_c] = fallback

    # 3. Load OSM Map
    osm_json = workspace_root / "Data_details" / "data" / "osm" / "coventry_s1_osm.json"
    mapper = CoordinateMapper(lat0_deg=origin_info["lat0_deg"], lon0_deg=origin_info["lon0_deg"], alt0_m=origin_info["alt0_m"])
    osm_loader = OSMLoader(coordinate_mapper=mapper)
    road_db = osm_loader.load_from_json(osm_json)
    spatial_index = SpatialIndex(road_db, sampling_interval_m=10.0)

    # 4. Outage Durations Benchmark Loop
    bo_base_start = 4600.0
    records = []

    for dur in [10, 30, 60, 120]:
        t_pre = 15.0
        t_post = 20.0
        t_win_start = bo_base_start - t_pre
        t_win_end = bo_base_start + dur + t_post

        idx_w_start = int(np.searchsorted(time_s, t_win_start))
        idx_w_end = int(np.searchsorted(time_s, t_win_end))
        df_win = df_imu.iloc[idx_w_start:idx_w_end].copy()

        ref_p_win = ref_pos_vbox[idx_w_start:idx_w_end]
        ref_v_win = ref_vel_vbox[idx_w_start:idx_w_end]
        ref_yaw_win = ref_yaw_rad[idx_w_start:idx_w_end]

        # Blackout interval
        idx_bo_s = int(np.searchsorted(time_s, bo_base_start)) - idx_w_start
        idx_bo_e = int(np.searchsorted(time_s, bo_base_start + dur)) - idx_w_start

        # Distance traveled during blackout according to true VBOX RTK
        ref_p_bo = ref_p_win[idx_bo_s:idx_bo_e]
        cum_d_bo = enu_cumulative_distance(ref_p_bo[:, 0], ref_p_bo[:, 1])
        d_traveled_outage = float(cum_d_bo[-1])

        # Initialize Phase 8 Streaming Engine
        ai_engine = CausalStreamingInferenceEngine(model, str(norm_json), 30, 12, "cpu")
        warm_df = df_imu.iloc[max(0, idx_w_start - 30):idx_w_start].copy()
        if len(warm_df) > 0:
            ai_engine.warm_up(warm_df[feature_cols].values.astype(np.float32))

        generator = CandidateGenerator(spatial_index, k_sigma=3.0, d_min_m=20.0)
        scorer = CandidateScorer()
        topology = RoadTopologyGraph(road_db)
        transition = TransitionModel(topology)
        matcher = TemporalMapMatcher(generator, scorer, transition, min_confidence_threshold=0.30)
        map_model = MapMeasurementModel()
        spline_odo = RoadSplineOdometry()

        engine = Phase8StreamingEngine(
            ref_lat_deg=origin_info["lat0_deg"],
            ref_lon_deg=origin_info["lon0_deg"],
            ref_alt_m=origin_info["alt0_m"],
            ai_engine=ai_engine,
            map_matcher=matcher,
            map_model=map_model,
            spline_odometry=spline_odo,
            enable_map_updates=True,
            enable_along_track_update=True,
        )

        # Initialize with VBOX RTK pose
        init_p = ref_p_win[0].copy()
        init_v = ref_v_win[0].copy()
        acc_entry = df_win[["acc_fwd_veh_filtered", "acc_lat_veh_filtered", "acc_up_veh_filtered"]].iloc[0].values
        acc_entry_corr = np.array([acc_entry[0], acc_entry[1], -acc_entry[2]], dtype=np.float64)
        init_q = initial_leveling_quaternion(acc_entry_corr, initial_gps_heading_deg=float(v_heading_deg[idx_w_start]))

        engine.initialize(init_p, init_v, init_q, initial_timestamp=float(df_win["time_s"].iloc[0]))

        n_win = len(df_win)
        win_times = df_win["time_s"].values
        features_matrix = df_win[feature_cols].values.astype(np.float32)
        est_pos = []

        last_gnss_t = -999.0

        for k in range(n_win):
            t_k = win_times[k]
            in_blackout = (bo_base_start <= t_k < (bo_base_start + dur))

            # 1 Hz GNSS arrival outside blackout
            is_gnss = (t_k - last_gnss_t >= 0.95) and not in_blackout
            if is_gnss:
                last_gnss_t = t_k
                gnss_lat = float(v_lat[idx_w_start + k])
                gnss_lon = float(v_lon[idx_w_start + k])
                gnss_alt = float(v_alt[idx_w_start + k])
                gnss_spd = float(v_speed_gt[idx_w_start + k])
                gnss_hdg = float(v_heading_deg[idx_w_start + k])
                gnss_acc = 1.5
                gnss_sats = 12
                gnss_time = t_k
            else:
                gnss_lat = None
                gnss_lon = None
                gnss_alt = None
                gnss_spd = None
                gnss_hdg = None
                gnss_acc = None
                gnss_sats = None
                gnss_time = None

            feat_k = features_matrix[k]
            est = engine.step(
                imu_sample_12ch=feat_k,
                timestamp=t_k,
                gps_lat=gnss_lat,
                gps_lon=gnss_lon,
                gps_alt=gnss_alt,
                gps_acc_m=gnss_acc,
                gps_sats=gnss_sats,
                gps_speed_mps=gnss_spd,
                gps_heading_deg=gnss_hdg,
                gps_timestamp=gnss_time,
                is_synthetic_outage=in_blackout,
            )
            est_pos.append(est.p_fused[:2])

        est_pos = np.array(est_pos)
        along_err, cross_err, total_err = compute_road_aligned_errors(
            est_pos, ref_p_win[:, :2], ref_yaw_win
        )

        outage_endpt_err = float(total_err[idx_bo_e - 1])
        outage_max_cross = float(np.max(np.abs(cross_err[idx_bo_s:idx_bo_e])))
        drift_pct = (outage_endpt_err / max(1.0, d_traveled_outage)) * 100.0

        records.append({
            "duration_s": dur,
            "distance_traveled_outage_m": round(d_traveled_outage, 2),
            "vbox_endpoint_error_m": round(outage_endpt_err, 2),
            "vbox_max_cross_track_m": round(outage_max_cross, 2),
            "dead_reckoning_drift_pct": round(drift_pct, 2),
            "isro_benchmark_target": "< 10.0 %",
            "isro_compliance_passed": bool(drift_pct < 10.0),
        })

    df_res = pd.DataFrame(records)
    print("\n" + "="*80)
    print("OFFICIAL VBOX RTK DEAD RECKONING BENCHMARK RESULTS (ISRO SIH26168)")
    print("="*80)
    print(df_res.to_string(index=False))
    print("="*80 + "\n")

    # Export to final benchmarks directories
    out_final = workspace_root / "benchmarks" / "phase8_final" / "phase8_durations_vbox_rtk_ground_truth.csv"
    out_tables = workspace_root / "Data_details" / "outputs" / "phase8" / "tables" / "phase8_durations_vbox_rtk_ground_truth.csv"
    df_res.to_csv(out_final, index=False)
    df_res.to_csv(out_tables, index=False)
    print(f"Authoritative VBOX benchmarks saved to {out_final}")


if __name__ == "__main__":
    main()
