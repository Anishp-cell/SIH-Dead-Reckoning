"""
Unit tests for TemporalMapMatcher (temporal_matcher.py).
"""

import numpy as np
import pytest

from Data_details.src.phase7.map.map_database import RoadSegment, RoadMapDatabase
from Data_details.src.phase7.map.spatial_index import SpatialIndex
from Data_details.src.phase7.matching.candidate_generator import CandidateGenerator
from Data_details.src.phase7.matching.candidate_scoring import CandidateScorer
from Data_details.src.phase7.matching.topology import RoadTopologyGraph, TransitionModel
from Data_details.src.phase7.matching.temporal_matcher import TemporalMapMatcher


def setup_matcher():
    db = RoadMapDatabase(name="matcher_test")
    # Seg A: (0, 0) -> (100, 0)
    segA = RoadSegment(
        segment_id="segA", way_id=1, sub_idx=0, start_node=1, end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]), p_end_enu=np.array([100.0, 0.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    )
    # Seg B: (100, 0) -> (200, 0) (connected to A)
    segB = RoadSegment(
        segment_id="segB", way_id=1, sub_idx=1, start_node=2, end_node=3,
        p_start_enu=np.array([100.0, 0.0, 0.0]), p_end_enu=np.array([200.0, 0.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    )
    # Seg C: (0, 30) -> (100, 30) (parallel road 30m north)
    segC = RoadSegment(
        segment_id="segC", way_id=2, sub_idx=0, start_node=10, end_node=11,
        p_start_enu=np.array([0.0, 30.0, 0.0]), p_end_enu=np.array([100.0, 30.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="residential", one_way=True,
    )
    db.add_segment(segA)
    db.add_segment(segB)
    db.add_segment(segC)
    db.build_connectivity()

    sindex = SpatialIndex(db, sampling_interval_m=10.0)
    generator = CandidateGenerator(sindex, k_sigma=3.0, d_min_m=20.0)
    scorer = CandidateScorer()
    graph = RoadTopologyGraph(db)
    trans = TransitionModel(graph)
    matcher = TemporalMapMatcher(generator, scorer, trans, min_confidence_threshold=0.3)
    return matcher


def test_temporal_matcher_continuous_tracking():
    matcher = setup_matcher()

    # Step 1: Start at (10, 1) near segA
    res1 = matcher.step(
        timestamp=0.0,
        p_enu=np.array([10.0, 1.0]),
        yaw_enu_rad=0.0,
        pos_cov_2x2=np.diag([4.0, 4.0]),
        v_forward_mps=10.0,
    )
    assert res1.matched_segment_id == "segA"
    assert res1.accepted
    assert res1.map_confidence > 0.5
    assert np.isclose(res1.cross_track_error, 1.0)

    # Step 2: Drive along segA to (50, 1)
    res2 = matcher.step(
        timestamp=4.0,
        p_enu=np.array([50.0, 1.0]),
        yaw_enu_rad=0.0,
        pos_cov_2x2=np.diag([4.0, 4.0]),
        v_forward_mps=10.0,
    )
    assert res2.matched_segment_id == "segA"
    assert res2.accepted
    assert res2.map_confidence > 0.5

    # Step 3: Transition across intersection onto segB at (110, 0.5)
    res3 = matcher.step(
        timestamp=10.0,
        p_enu=np.array([110.0, 0.5]),
        yaw_enu_rad=0.0,
        pos_cov_2x2=np.diag([4.0, 4.0]),
        v_forward_mps=10.0,
    )
    assert res3.matched_segment_id == "segB"
    assert res3.accepted


def test_temporal_matcher_parallel_road_suppression():
    matcher = setup_matcher()

    # Initialize firmly on segA
    matcher.step(0.0, np.array([20.0, 0.0]), 0.0, np.diag([4.0, 4.0]))
    matcher.step(1.0, np.array([30.0, 0.0]), 0.0, np.diag([4.0, 4.0]))

    # Position briefly drifts towards parallel road segC (e.g. to North=18m, where it is
    # 18m from segA and 12m from segC). Pure Euclidean nearest-neighbor would snap to segC (12m < 18m).
    # But because history and topology favor segA, temporal matcher should retain segA or reject jump!
    res = matcher.step(2.0, np.array([40.0, 16.0]), 0.0, np.diag([9.0, 9.0]))
    # Matcher should prefer staying on segA due to transition penalty to segC
    assert res.matched_segment_id == "segA"


def test_temporal_matcher_off_road_deactivation():
    matcher = setup_matcher()

    # Query far into a field at (500, 500)
    res_far = matcher.step(0.0, np.array([500.0, 500.0]), 0.0)
    assert res_far.matched_segment_id is None
    assert not res_far.accepted
    assert np.isclose(res_far.map_confidence, 0.0)
