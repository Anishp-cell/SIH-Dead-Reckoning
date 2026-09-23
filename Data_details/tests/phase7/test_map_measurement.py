"""
Unit tests for MapMeasurementModel and map updates (map_measurement.py, map_update.py).
"""

import numpy as np
import pytest

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import euler_zyx_to_quaternion
from Data_details.src.phase7.matching.temporal_matcher import MapMatchResult
from Data_details.src.phase7.matching.map_measurement import MapMeasurementModel, MapMeasurementConfig
from Data_details.src.phase7.matching.map_update import apply_map_cross_track_update, apply_map_heading_update


def make_dummy_match(cross_track: float, heading_diff: float, road_heading: float, c_map: float = 0.9) -> MapMatchResult:
    return MapMatchResult(
        timestamp=1.0,
        matched_segment_id="seg1",
        matched_way_id=10,
        projected_position_enu=np.array([50.0, 0.0]),
        road_heading_enu=road_heading,
        road_heading_gps_deg=90.0,
        cross_track_error=cross_track,
        along_track_position=50.0,
        candidate_count=1,
        candidate_probability=0.95,
        map_confidence=c_map,
        heading_residual=heading_diff,
        distance_residual=abs(cross_track),
        transition_score=0.85,
        accepted=True,
    )


def test_cross_track_jacobian_analytical_vs_numerical():
    # Road heading = +pi/4 (45 deg)
    road_heading = np.pi / 4.0
    # Left normal: n = [-sin(45), cos(45)] = [-sqrt(2)/2, sqrt(2)/2]
    match = make_dummy_match(cross_track=3.0, heading_diff=0.0, road_heading=road_heading)

    model = MapMeasurementModel()
    P = np.eye(15) * 4.0
    nu, H, R, nis, accepted = model.compute_cross_track_update(match, P)

    assert accepted
    assert np.isclose(nu, -3.0)

    # Numerical verification of H w.r.t p = [E, N, U]
    eps = 1e-6
    p0 = np.array([50.0, 3.0])
    # Distance to line passing through (50, 0) with tangent [cos(psi), sin(psi)]
    # normal n = [-sin(psi), cos(psi)]
    n = np.array([-np.sin(road_heading), np.cos(road_heading)])
    h0 = float(n @ (p0 - np.array([50.0, 0.0])))

    # Perturb E
    p_pert_E = p0 + np.array([eps, 0.0])
    h_pert_E = float(n @ (p_pert_E - np.array([50.0, 0.0])))
    dH_dE_num = (h_pert_E - h0) / eps

    # Perturb N
    p_pert_N = p0 + np.array([0.0, eps])
    h_pert_N = float(n @ (p_pert_N - np.array([50.0, 0.0])))
    dH_dN_num = (h_pert_N - h0) / eps

    assert np.isclose(H[0, 0], dH_dE_num, atol=1e-5)
    assert np.isclose(H[0, 1], dH_dN_num, atol=1e-5)
    assert np.allclose(H[0, 2:], 0.0)


def test_heading_jacobian():
    match = make_dummy_match(cross_track=0.0, heading_diff=np.radians(5.0), road_heading=0.0)
    model = MapMeasurementModel()
    P = np.eye(15) * 0.01

    nu, H, R, nis, accepted = model.compute_heading_update(match, P, v_forward_mps=10.0)
    assert accepted
    assert np.isclose(nu, -np.radians(5.0))
    assert H[0, 8] == 1.0
    assert np.isclose(np.sum(np.abs(H)) - 1.0, 0.0)


def test_adaptive_covariance_scaling():
    model = MapMeasurementModel()
    P = np.eye(15) * 1.0

    # High confidence: c_map = 0.9 -> R = sigma^2 / 0.9^2
    match_high = make_dummy_match(cross_track=1.0, heading_diff=0.0, road_heading=0.0, c_map=0.9)
    _, _, R_high, _, _ = model.compute_cross_track_update(match_high, P)

    # Low confidence: c_map = 0.2 -> R = sigma^2 / 0.2^2
    match_low = make_dummy_match(cross_track=1.0, heading_diff=0.0, road_heading=0.0, c_map=0.2)
    _, _, R_low, _, _ = model.compute_cross_track_update(match_low, P)

    assert R_low > R_high
    assert np.isclose(R_low / R_high, (0.9 / 0.2)**2)


def test_map_cross_track_kalman_update():
    # Vehicle is 4m north of East road (n = [0, 1])
    q = euler_zyx_to_quaternion(0.0, 0.0, 0.0)  # Heading East
    state = NominalState(
        p=np.array([50.0, 4.0, 0.0]),
        v=np.array([10.0, 0.0, 0.0]),
        q=q,
        ba=np.zeros(3),
        bg=np.zeros(3),
    )
    P = np.diag([25.0, 25.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])

    match = make_dummy_match(cross_track=4.0, heading_diff=0.0, road_heading=0.0, c_map=0.9)
    model = MapMeasurementModel()

    state_plus, P_plus, nu, S, nis, accepted = apply_map_cross_track_update(state, P, match, model)
    assert accepted
    # North position must be corrected downwards towards 0
    assert state_plus.p[1] < 4.0
    assert state_plus.p[1] > 0.0  # Soft update doesn't snap to 0
    # North position variance must be reduced
    assert P_plus[1, 1] < P[1, 1]
    # East position (along-track) should remain essentially unshifted
    assert np.isclose(state_plus.p[0], 50.0, atol=1e-3)
