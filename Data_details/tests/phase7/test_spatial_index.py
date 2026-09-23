"""
Unit tests for SpatialIndex (spatial_index.py).
"""

import numpy as np
import pytest

from Data_details.src.phase7.map.map_database import RoadSegment, RoadMapDatabase
from Data_details.src.phase7.map.spatial_index import SpatialIndex


def build_dummy_map():
    db = RoadMapDatabase(name="test_db")
    # Segment 1: from (0, 0) to (100, 0) (East)
    seg1 = RoadSegment(
        segment_id="seg1",
        way_id=1,
        sub_idx=0,
        start_node=1,
        end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]),
        p_end_enu=np.array([100.0, 0.0, 0.0]),
        length_m=100.0,
        heading_start_rad=0.0,
        heading_end_rad=0.0,
        heading_rad=0.0,
        heading_gps_deg=90.0,
        road_class="residential",
        one_way=True,
    )
    # Segment 2: from (0, 50) to (100, 50) (Parallel East road, 50m north)
    seg2 = RoadSegment(
        segment_id="seg2",
        way_id=2,
        sub_idx=0,
        start_node=3,
        end_node=4,
        p_start_enu=np.array([0.0, 50.0, 0.0]),
        p_end_enu=np.array([100.0, 50.0, 0.0]),
        length_m=100.0,
        heading_start_rad=0.0,
        heading_end_rad=0.0,
        heading_rad=0.0,
        heading_gps_deg=90.0,
        road_class="secondary",
        one_way=True,
    )
    db.add_segment(seg1)
    db.add_segment(seg2)
    db.build_connectivity()
    return db


def test_spatial_index_query_radius():
    db = build_dummy_map()
    sindex = SpatialIndex(db, sampling_interval_m=10.0)

    # Query near (50, 5) with radius 10m -> should find only seg1
    candidates, lat = sindex.query_radius(np.array([50.0, 5.0]), radius_m=10.0)
    assert len(candidates) == 1
    assert candidates[0].segment_id == "seg1"
    assert lat >= 0.0

    # Query near (50, 25) with radius 35m -> should find both seg1 and seg2
    candidates_both, _ = sindex.query_radius(np.array([50.0, 25.0]), radius_m=35.0)
    seg_ids = {s.segment_id for s in candidates_both}
    assert seg_ids == {"seg1", "seg2"}

    # Query far away (500, 500) with radius 20m -> should find nothing
    empty, _ = sindex.query_radius(np.array([500.0, 500.0]), radius_m=20.0)
    assert len(empty) == 0


def test_spatial_index_query_uncertainty():
    db = build_dummy_map()
    sindex = SpatialIndex(db, sampling_interval_m=10.0)

    # Low uncertainty: sigma_pos = 1m -> r = 3*1 + 10 = 13m
    cov_low = np.diag([1.0, 1.0])
    cands, r_low, _ = sindex.query_uncertainty(np.array([50.0, 2.0]), pos_cov_2x2=cov_low, d_min_m=10.0)
    assert len(cands) == 1
    assert cands[0].segment_id == "seg1"
    assert np.isclose(r_low, 13.0)

    # High uncertainty: sigma_pos = 15m -> r = 3*15 + 10 = 55m
    cov_high = np.diag([225.0, 225.0])
    cands_high, r_high, _ = sindex.query_uncertainty(np.array([50.0, 2.0]), pos_cov_2x2=cov_high, d_min_m=10.0)
    assert len(cands_high) == 2  # Broad search captures both parallel roads
    assert np.isclose(r_high, 55.0)
