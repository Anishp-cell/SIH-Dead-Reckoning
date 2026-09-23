"""
Phase 7 Ablation Study Runner:
Executes M0 to M5 ablation study on 60s blackout outage compliant with Section 36 of the specification:
- M0: Phase 6 without map matching
- M1: Nearest-road projection only (hard geometric snapping)
- M2: Distance + Heading scoring (static scoring)
- M3: Distance + Heading + Topology (Markovian connectivity)
- M4: Full Probabilistic Causal Map Matcher (belief tracking, no filter feedback)
- M5: Full Map Matcher + ESKF Soft Map Updates (cross-track & heading updates)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .blackout_benchmarks import run_phase7_trajectory
from .map_matching_metrics import compute_phase7_metrics
from ..matching.candidate_scoring import ScoringWeights
from ..map.map_database import RoadMapDatabase
from ..map.spatial_index import SpatialIndex
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine


def run_m0_to_m5_ablation(
    df_slice: pd.DataFrame,
    ref_pos_3d: np.ndarray,
    ref_vel_3d: np.ndarray,
    ref_yaw_enu_rad: np.ndarray,
    traveled_dist: float,
    init_p: np.ndarray,
    init_v: np.ndarray,
    init_q: np.ndarray,
    road_db: RoadMapDatabase,
    spatial_index: SpatialIndex,
    ai_engine: CausalStreamingInferenceEngine,
    warm_start_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Executes M0 to M5 ablation study on the given slice.
    """
    configs = [
        # (id, description, en_mm, en_up, en_ct, en_hd, hard_snap, weights)
        ("M0_Phase6_Baseline", "Phase 6 Baseline (No Map Matching)", False, False, False, False, False, None),
        ("M1_Nearest_Road_Snap", "Nearest-Road Hard Snapping", True, False, False, False, True, None),
        ("M2_Dist_Head_Scoring", "Distance + Heading Scoring (Static)", True, False, False, False, False, ScoringWeights(w_distance=1.0, w_heading=1.0, w_motion=0.0)),
        ("M3_Dist_Head_Topo", "Distance + Heading + Topology Graph", True, False, False, False, False, ScoringWeights(w_distance=1.0, w_heading=1.0, w_motion=0.5)),
        ("M4_Prob_Causal_Matcher", "Full Probabilistic Causal Matcher (No ESKF Update)", True, False, False, False, False, ScoringWeights()),
        ("M5_Full_Map_ESKF_Updates", "Full Matcher + Soft Kalman Updates (Cross-Track & Heading)", True, True, True, True, False, ScoringWeights()),
    ]

    results = []
    for cfg_id, desc, en_mm, en_up, en_ct, en_hd, snap, weights in configs:
        ai_engine.reset()
        res = run_phase7_trajectory(
            df_slice=df_slice,
            init_p_enu=init_p,
            init_v_enu=init_v,
            init_q=init_q,
            road_db=road_db,
            spatial_index=spatial_index,
            ai_engine=ai_engine,
            enable_map_matching=en_mm,
            enable_map_updates=en_up,
            enable_cross_track_update=en_ct,
            enable_heading_update=en_hd,
            scoring_weights=weights,
            hard_snapping=snap,
            warm_start_df=warm_start_df,
        )

        metrics = compute_phase7_metrics(
            p_est=res["p_est"],
            p_ref=ref_pos_3d,
            v_est=res["v_est"],
            v_ref=ref_vel_3d,
            yaw_est_rad=res["yaw_est"],
            yaw_ref_rad=ref_yaw_enu_rad,
            traveled_distance_m=traveled_dist,
            matched_segment_ids=res["matched_ids"],
            map_confidences=res["map_confidences"],
            candidate_counts=res["candidate_counts"],
            cross_track_errors=res["cross_track_errors"],
            map_ct_accepted=res["map_ct_accepted"],
            map_hd_accepted=res["map_hd_accepted"],
            ct_nis_list=res["ct_nis_list"],
            hd_nis_list=res["hd_nis_list"],
        )

        results.append({
            "configuration": cfg_id,
            "description": desc,
            **metrics,
        })

    return pd.DataFrame(results)
