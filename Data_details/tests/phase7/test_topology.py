"""
Unit tests for RoadTopologyGraph and TransitionModel (topology.py).
"""

import numpy as np
import pytest

from Data_details.src.phase7.map.map_database import RoadSegment, RoadMapDatabase
from Data_details.src.phase7.matching.candidate_generator import RoadCandidate
from Data_details.src.phase7.matching.topology import RoadTopologyGraph, TransitionModel


def build_connected_test_db():
    db = RoadMapDatabase(name="topo_test")
    # Seg A: node 1 -> node 2
    segA = RoadSegment(
        segment_id="segA", way_id=1, sub_idx=0, start_node=1, end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]), p_end_enu=np.array([100.0, 0.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    )
    # Seg B: node 2 -> node 3 (connected to A)
    segB = RoadSegment(
        segment_id="segB", way_id=1, sub_idx=1, start_node=2, end_node=3,
        p_start_enu=np.array([100.0, 0.0, 0.0]), p_end_enu=np.array([200.0, 0.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    )
    # Seg C: node 10 -> node 11 (disconnected parallel road at North 50m)
    segC = RoadSegment(
        segment_id="segC", way_id=2, sub_idx=0, start_node=10, end_node=11,
        p_start_enu=np.array([0.0, 50.0, 0.0]), p_end_enu=np.array([100.0, 50.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="residential", one_way=True,
    )
    db.add_segment(segA)
    db.add_segment(segB)
    db.add_segment(segC)
    db.build_connectivity()
    return db


def make_cand_for_segment(seg: RoadSegment, along_m: float) -> RoadCandidate:
    frac = along_m / seg.length_m
    pt = seg.p_start_enu[:2] + frac * (seg.p_end_enu[:2] - seg.p_start_enu[:2])
    return RoadCandidate(
        segment=seg,
        segment_id=seg.segment_id,
        way_id=seg.way_id,
        projected_point_enu=pt,
        cross_track_m=0.0,
        abs_distance_m=0.0,
        along_track_m=along_m,
        fraction=frac,
        road_heading_enu_rad=seg.heading_rad,
        road_heading_gps_deg=seg.heading_gps_deg,
        heading_residual_rad=0.0,
        unit_tangent=np.array([1.0, 0.0]),
        unit_normal=np.array([0.0, 1.0]),
        is_clamped=False,
    )


def test_topology_connectivity():
    db = build_connected_test_db()
    graph = RoadTopologyGraph(db)

    assert graph.is_connected("segA", "segB")
    assert not graph.is_connected("segA", "segC")
    assert not graph.is_connected("segB", "segA")  # Directed edge
    assert "segB" in graph.successors("segA")


def test_transition_stay_and_connected():
    db = build_connected_test_db()
    graph = RoadTopologyGraph(db)
    trans = TransitionModel(graph, p_stay=0.85, p_connected=0.14, p_jump_base=1e-4)

    candA_t0 = make_cand_for_segment(db.get_segment("segA"), along_m=50.0)
    candA_t1 = make_cand_for_segment(db.get_segment("segA"), along_m=52.0)
    candB_t1 = make_cand_for_segment(db.get_segment("segB"), along_m=2.0)
    candC_t1 = make_cand_for_segment(db.get_segment("segC"), along_m=52.0)

    # Staying on same segment
    p_stay = trans.transition_probability(candA_t0, candA_t1, delta_dist_m=2.0)
    assert np.isclose(p_stay, 0.85)

    # Transitioning to connected successor segment
    p_conn = trans.transition_probability(candA_t0, candB_t1, delta_dist_m=52.0)
    assert np.isclose(p_conn, 0.14)

    # Disconnected jump to parallel road C
    p_jump = trans.transition_probability(candA_t0, candC_t1, delta_dist_m=2.0)
    assert p_jump < 1e-4
    assert p_jump < p_conn < p_stay
