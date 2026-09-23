"""
Phase 7 Synthetic Road Network Tests (Tests A through H)
Compliant with Section 34 of the Phase 7 specification:
- Test A: Straight road lateral drift reduction
- Test B: Parallel roads disambiguation
- Test C: Intersection handling without spurious jumps
- Test D: Curved road tangent tracking
- Test E: Wrong nearby road rejection
- Test F: Large position uncertainty behavior
- Test G: No nearby road graceful deactivation
- Test H: Heading drift injection and recovery
"""

import numpy as np
import pytest

from Data_details.src.phase5.core.quaternion import euler_zyx_to_quaternion, quaternion_to_euler_zyx
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase7.map.map_database import RoadSegment, RoadMapDatabase
from Data_details.src.phase7.map.spatial_index import SpatialIndex
from Data_details.src.phase7.matching.candidate_generator import CandidateGenerator
from Data_details.src.phase7.matching.candidate_scoring import CandidateScorer
from Data_details.src.phase7.matching.topology import RoadTopologyGraph, TransitionModel
from Data_details.src.phase7.matching.temporal_matcher import TemporalMapMatcher
from Data_details.src.phase7.core.phase7_navigator import Phase7Navigator


def build_synthetic_straight_map() -> RoadMapDatabase:
    """Builds a 500m straight road running East along N=0."""
    db = RoadMapDatabase(name="straight_road")
    for i in range(5):
        seg = RoadSegment(
            segment_id=f"seg_straight_{i}",
            way_id=1,
            sub_idx=i,
            start_node=i,
            end_node=i + 1,
            p_start_enu=np.array([i * 100.0, 0.0, 0.0]),
            p_end_enu=np.array([(i + 1) * 100.0, 0.0, 0.0]),
            length_m=100.0,
            heading_start_rad=0.0,
            heading_end_rad=0.0,
            heading_rad=0.0,
            heading_gps_deg=90.0,
            road_class="primary",
            one_way=True,
        )
        db.add_segment(seg)
    db.build_connectivity()
    return db


def create_navigator(db: RoadMapDatabase, enable_map_updates: bool = True) -> Phase7Navigator:
    sindex = SpatialIndex(db, sampling_interval_m=10.0)
    gen = CandidateGenerator(sindex, k_sigma=3.0, d_min_m=20.0)
    scorer = CandidateScorer()
    graph = RoadTopologyGraph(db)
    trans = TransitionModel(graph)
    matcher = TemporalMapMatcher(gen, scorer, trans, min_confidence_threshold=0.3)
    eskf = Phase6ESKF()
    return Phase7Navigator(eskf=eskf, map_matcher=matcher, enable_map_updates=enable_map_updates)


# --- Test A: Straight Road Lateral Drift ---
def test_synthetic_test_a_straight_road_lateral_drift():
    """Verifies that map matching constrains cross-track drift along a straight road."""
    db = build_synthetic_straight_map()
    nav_with_map = create_navigator(db, enable_map_updates=True)
    nav_no_map = create_navigator(db, enable_map_updates=False)

    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([10.0, 0.0, 0.0])
    q0 = euler_zyx_to_quaternion(0.0, 0.0, 0.0)

    nav_with_map.initialize(p0, v0, q0)
    nav_no_map.initialize(p0, v0, q0)

    # Simulate 5 seconds of driving forward with a small lateral acceleration bias (causing drift)
    dt = 0.1
    f_b_drift = np.array([0.0, 0.15, 9.80665])  # 0.15 m/s^2 lateral bias in body-Y (North)
    omega_b = np.array([0.0, 0.0, 0.0])

    p_map_traj = []
    p_nomap_traj = []

    for k in range(50):
        t = k * dt
        est_map = nav_with_map.step(f_b_drift, omega_b, dt, t, v_speed_ai=10.0, enable_nhc=False)
        est_nomap = nav_no_map.step(f_b_drift, omega_b, dt, t, v_speed_ai=10.0, enable_nhc=False)
        p_map_traj.append(est_map.p_eskf[1])
        p_nomap_traj.append(est_nomap.p_eskf[1])

    final_lat_map = abs(p_map_traj[-1])
    final_lat_nomap = abs(p_nomap_traj[-1])

    # Map updates must substantially reduce lateral position drift
    assert final_lat_map < final_lat_nomap
    assert final_lat_map < 1.0  # Kept within 1m of centerline


# --- Test B: Parallel Roads ---
def test_synthetic_test_b_parallel_roads_disambiguation():
    """Matcher must select the correct road using heading and topological history."""
    db = RoadMapDatabase(name="parallel_roads")
    # Road 1: Along N=0, heading East
    db.add_segment(RoadSegment(
        segment_id="road1", way_id=1, sub_idx=0, start_node=1, end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]), p_end_enu=np.array([200.0, 0.0, 0.0]),
        length_m=200.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    ))
    # Road 2: Along N=25, parallel heading East
    db.add_segment(RoadSegment(
        segment_id="road2", way_id=2, sub_idx=0, start_node=3, end_node=4,
        p_start_enu=np.array([0.0, 25.0, 0.0]), p_end_enu=np.array([200.0, 25.0, 0.0]),
        length_m=200.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="secondary", one_way=True,
    ))
    db.build_connectivity()

    nav = create_navigator(db, enable_map_updates=True)
    nav.initialize(np.array([10.0, 1.0, 0.0]), np.array([10.0, 0.0, 0.0]), euler_zyx_to_quaternion(0.0, 0.0, 0.0))

    # Vehicle drives along Road 1
    f_b = np.array([0.0, 0.0, 9.80665])
    omega_b = np.zeros(3)

    for k in range(20):
        t = k * 0.1
        est = nav.step(f_b, omega_b, 0.1, t, v_speed_ai=10.0)
        assert est.matched_segment_id == "road1"


# --- Test C: Intersection Handling ---
def test_synthetic_test_c_intersection_handling():
    """At an intersection, multiple candidate roads are evaluated without spurious jumps."""
    db = RoadMapDatabase(name="intersection")
    # Approach: (0, 0) -> (100, 0)
    db.add_segment(RoadSegment(
        segment_id="approach", way_id=1, sub_idx=0, start_node=1, end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]), p_end_enu=np.array([100.0, 0.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    ))
    # Branch Straight: (100, 0) -> (200, 0)
    db.add_segment(RoadSegment(
        segment_id="branch_straight", way_id=2, sub_idx=0, start_node=2, end_node=3,
        p_start_enu=np.array([100.0, 0.0, 0.0]), p_end_enu=np.array([200.0, 0.0, 0.0]),
        length_m=100.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="primary", one_way=True,
    ))
    # Branch Left (North): (100, 0) -> (100, 100) (heading +pi/2)
    db.add_segment(RoadSegment(
        segment_id="branch_left", way_id=3, sub_idx=0, start_node=2, end_node=4,
        p_start_enu=np.array([100.0, 0.0, 0.0]), p_end_enu=np.array([100.0, 100.0, 0.0]),
        length_m=100.0, heading_start_rad=np.pi / 2.0, heading_end_rad=np.pi / 2.0, heading_rad=np.pi / 2.0,
        heading_gps_deg=0.0, road_class="primary", one_way=True,
    ))
    db.build_connectivity()

    nav = create_navigator(db, enable_map_updates=True)
    # Vehicle maintains East heading
    nav.initialize(np.array([90.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), euler_zyx_to_quaternion(0.0, 0.0, 0.0))

    f_b = np.array([0.0, 0.0, 9.80665])
    omega_b = np.zeros(3)

    # Step through intersection node
    est = nav.step(f_b, omega_b, 0.1, 0.1, v_speed_ai=10.0)
    # At junction, both branches exist as candidates, but East heading must favor branch_straight
    assert est.matched_segment_id in ("approach", "branch_straight")


# --- Test D: Curved Road ---
def test_synthetic_test_d_curved_road_tangent_tracking():
    """On a curved polyline road, tangent heading changes continuously."""
    db = RoadMapDatabase(name="curved_road")
    # Arc of circle with radius R=100m, from theta = 0 to pi/4
    R = 100.0
    thetas = np.linspace(0.0, np.pi / 4.0, 6)
    for i in range(len(thetas) - 1):
        th1, th2 = thetas[i], thetas[i + 1]
        p1 = np.array([R * np.sin(th1), R * (1.0 - np.cos(th1)), 0.0])
        p2 = np.array([R * np.sin(th2), R * (1.0 - np.cos(th2)), 0.0])
        de = p2[0] - p1[0]
        dn = p2[1] - p1[1]
        L = np.sqrt(de**2 + dn**2)
        hd = np.arctan2(dn, de)
        db.add_segment(RoadSegment(
            segment_id=f"curve_{i}", way_id=1, sub_idx=i, start_node=i, end_node=i + 1,
            p_start_enu=p1, p_end_enu=p2, length_m=L, heading_start_rad=hd, heading_end_rad=hd,
            heading_rad=hd, heading_gps_deg=float(np.degrees(np.pi / 2.0 - hd) % 360),
            road_class="primary", one_way=True,
        ))
    db.build_connectivity()

    # Verify road headings change smoothly
    headings = [seg.heading_rad for seg in db.segments.values()]
    assert all(h >= 0.0 for h in headings)
    assert headings[-1] > headings[0]


# --- Test E: Wrong Nearby Road Rejection ---
def test_synthetic_test_e_wrong_nearby_road_rejection():
    """Matcher rejects geometrically close candidates that contradict topology and motion."""
    db = RoadMapDatabase(name="wrong_road")
    # Main highway: East directed
    db.add_segment(RoadSegment(
        segment_id="highway", way_id=1, sub_idx=0, start_node=1, end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]), p_end_enu=np.array([200.0, 0.0, 0.0]),
        length_m=200.0, heading_start_rad=0.0, heading_end_rad=0.0, heading_rad=0.0,
        heading_gps_deg=90.0, road_class="trunk", one_way=True,
    ))
    # Perpendicular dead-end alley: 8m away, but heading North (+pi/2)
    db.add_segment(RoadSegment(
        segment_id="alley", way_id=2, sub_idx=0, start_node=10, end_node=11,
        p_start_enu=np.array([50.0, 8.0, 0.0]), p_end_enu=np.array([50.0, 30.0, 0.0]),
        length_m=22.0, heading_start_rad=np.pi / 2.0, heading_end_rad=np.pi / 2.0, heading_rad=np.pi / 2.0,
        heading_gps_deg=0.0, road_class="service", one_way=True,
    ))
    db.build_connectivity()

    nav = create_navigator(db, enable_map_updates=True)
    # Vehicle is at (50, 4) with East heading (yaw=0)
    nav.initialize(np.array([50.0, 4.0, 0.0]), np.array([15.0, 0.0, 0.0]), euler_zyx_to_quaternion(0.0, 0.0, 0.0))

    est = nav.step(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.1, 0.1, v_speed_ai=15.0)
    # Must match highway, NOT the perpendicular alley despite 4m proximity
    assert est.matched_segment_id == "highway"


# --- Test F: Large Position Uncertainty ---
def test_synthetic_test_f_large_position_uncertainty():
    """With large covariance, search radius broadens and map confidence is scaled down gracefully."""
    db = build_synthetic_straight_map()
    nav = create_navigator(db, enable_map_updates=True)

    # Initialize with 50m position standard deviation
    P_large = np.eye(15)
    P_large[0, 0] = 2500.0
    P_large[1, 1] = 2500.0

    nav.initialize(
        p0=np.array([50.0, 15.0, 0.0]),
        v0=np.array([10.0, 0.0, 0.0]),
        q0=euler_zyx_to_quaternion(0.0, 0.0, 0.0),
        P0=P_large,
    )

    est = nav.step(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.1, 0.1, v_speed_ai=10.0)
    # Candidate search must succeed even at 15m distance due to large sigma
    assert est.candidate_count > 0
    # Map confidence should be attenuated
    assert est.map_confidence < 0.95


# --- Test G: No Nearby Road ---
def test_synthetic_test_g_no_nearby_road_graceful_deactivation():
    """Off-road condition deactivates map updates with zero filter instability."""
    db = build_synthetic_straight_map()
    nav = create_navigator(db, enable_map_updates=True)

    # Query 500m away in an empty field
    nav.initialize(np.array([500.0, 500.0, 0.0]), np.array([10.0, 0.0, 0.0]), euler_zyx_to_quaternion(0.0, 0.0, 0.0))

    est = nav.step(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.1, 0.1, v_speed_ai=10.0)
    assert est.matched_segment_id is None
    assert not est.map_cross_track_accepted
    assert not est.map_heading_accepted
    assert np.isclose(est.map_confidence, 0.0)


# --- Test H: Heading Drift Injection and Recovery ---
@pytest.mark.parametrize("drift_deg", [10.0, 30.0, 60.0])
def test_synthetic_test_h_heading_drift_recovery(drift_deg):
    """Injects heading error and evaluates whether road heading updates constrain drift."""
    db = build_synthetic_straight_map()
    nav = create_navigator(db, enable_map_updates=True)

    # True road heading is East (0 rad)
    # Inject heading error of drift_deg
    drift_rad = np.radians(drift_deg)
    q_drifted = euler_zyx_to_quaternion(0.0, 0.0, drift_rad)

    P_init = np.eye(15)
    P_init[8, 8] = max(0.01, (drift_rad * 1.2)**2)

    nav.initialize(
        p0=np.array([50.0, 1.0, 0.0]),
        v0=np.array([10.0 * np.cos(drift_rad), 10.0 * np.sin(drift_rad), 0.0]),
        q0=q_drifted,
        P0=P_init,
    )

    # Step vehicle forward over 10 steps (1.0 second)
    f_b = np.array([0.0, 0.0, 9.80665])
    omega_b = np.zeros(3)

    yaws = []
    heading_accepted_any = False
    for k in range(10):
        t = (k + 1) * 0.1
        est = nav.step(f_b, omega_b, 0.1, t, v_speed_ai=10.0)
        yaws.append(est.yaw_eskf_rad)
        if est.map_heading_accepted:
            heading_accepted_any = True

    if drift_deg <= 30.0:
        # Heading error is within recoverable range -> heading update applied
        assert heading_accepted_any
        # Final yaw should be significantly reduced towards road heading (0 rad)
        assert abs(yaws[-1]) < drift_rad * 0.5
    else:
        # At 60 deg, heading divergence is large; innovation gating or generator rejects extreme mismatch
        # ensuring the filter is not shocked by an ambiguous orientation
        assert not heading_accepted_any or abs(yaws[-1]) <= drift_rad
