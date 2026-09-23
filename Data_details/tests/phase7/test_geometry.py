"""
Unit tests for Phase 7 Road Geometry Engine (geometry.py).
"""

import numpy as np
import pytest

from Data_details.src.phase7.map.geometry import (
    wrap_angle_rad,
    wrap_angle_deg,
    compute_segment_azimuth_enu,
    enu_yaw_to_geographic_azimuth_deg,
    geographic_azimuth_to_enu_yaw_rad,
    project_point_to_segment,
    PointToSegmentResult,
)


def test_wrap_angle_rad():
    assert np.isclose(wrap_angle_rad(0.0), 0.0)
    assert np.isclose(wrap_angle_rad(np.pi), np.pi) or np.isclose(wrap_angle_rad(np.pi), -np.pi)
    assert np.isclose(wrap_angle_rad(3.0 * np.pi), np.pi) or np.isclose(wrap_angle_rad(3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_angle_rad(-3.0 * np.pi), np.pi) or np.isclose(wrap_angle_rad(-3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_angle_rad(np.pi / 2.0), np.pi / 2.0)
    assert np.isclose(wrap_angle_rad(-np.pi / 2.0), -np.pi / 2.0)


def test_wrap_angle_deg():
    assert np.isclose(wrap_angle_deg(0.0), 0.0)
    assert np.isclose(abs(wrap_angle_deg(180.0)), 180.0)
    assert np.isclose(wrap_angle_deg(370.0), 10.0)
    assert np.isclose(wrap_angle_deg(-370.0), -10.0)
    assert np.isclose(wrap_angle_deg(90.0), 90.0)


def test_compute_segment_azimuth_enu():
    # East directed segment: de > 0, dn = 0 -> yaw = 0
    p0 = np.array([0.0, 0.0])
    p_east = np.array([10.0, 0.0])
    assert np.isclose(compute_segment_azimuth_enu(p0, p_east), 0.0)

    # North directed segment: de = 0, dn > 0 -> yaw = +pi/2
    p_north = np.array([0.0, 10.0])
    assert np.isclose(compute_segment_azimuth_enu(p0, p_north), np.pi / 2.0)

    # West directed segment: de < 0, dn = 0 -> yaw = pi
    p_west = np.array([-10.0, 0.0])
    assert np.isclose(abs(compute_segment_azimuth_enu(p0, p_west)), np.pi)

    # South directed segment: de = 0, dn < 0 -> yaw = -pi/2
    p_south = np.array([0.0, -10.0])
    assert np.isclose(compute_segment_azimuth_enu(p0, p_south), -np.pi / 2.0)


def test_enu_geographic_conversions():
    # East in ENU (yaw=0 rad) is 90 deg azimuth (East) in Geographic
    assert np.isclose(enu_yaw_to_geographic_azimuth_deg(0.0), 90.0)
    assert np.isclose(geographic_azimuth_to_enu_yaw_rad(90.0), 0.0)

    # North in ENU (yaw=pi/2 rad) is 0 deg azimuth (North) in Geographic
    assert np.isclose(enu_yaw_to_geographic_azimuth_deg(np.pi / 2.0), 0.0)
    assert np.isclose(geographic_azimuth_to_enu_yaw_rad(0.0), np.pi / 2.0)

    # South in ENU (yaw=-pi/2 rad) is 180 deg azimuth in Geographic
    assert np.isclose(enu_yaw_to_geographic_azimuth_deg(-np.pi / 2.0), 180.0)
    assert np.isclose(geographic_azimuth_to_enu_yaw_rad(180.0), -np.pi / 2.0)


def test_project_point_to_horizontal_segment():
    # Segment from (0, 0) to (100, 0) (East directed)
    a = np.array([0.0, 0.0])
    b = np.array([100.0, 0.0])

    # Point at (50, 5) -> 5m to the North (Left of East-directed road)
    p = np.array([50.0, 5.0])
    res = project_point_to_segment(p, a, b)
    assert np.allclose(res.projected_point_enu, [50.0, 0.0])
    assert np.isclose(res.cross_track_m, 5.0)  # Positive = to the left
    assert np.isclose(res.abs_distance_m, 5.0)
    assert np.isclose(res.along_track_m, 50.0)
    assert np.isclose(res.fraction, 0.5)
    assert not res.is_clamped
    assert np.allclose(res.unit_tangent, [1.0, 0.0])
    assert np.allclose(res.unit_normal, [0.0, 1.0])

    # Point at (50, -5) -> 5m to the South (Right of East-directed road)
    p_right = np.array([50.0, -5.0])
    res_right = project_point_to_segment(p_right, a, b)
    assert np.isclose(res_right.cross_track_m, -5.0)  # Negative = to the right
    assert np.isclose(res_right.abs_distance_m, 5.0)


def test_project_point_clamped_endpoints():
    # Segment from (0, 0) to (100, 0)
    a = np.array([0.0, 0.0])
    b = np.array([100.0, 0.0])

    # Query before start: (-10, 5)
    p_before = np.array([-10.0, 5.0])
    res_before = project_point_to_segment(p_before, a, b)
    assert np.allclose(res_before.projected_point_enu, a)
    assert res_before.is_clamped
    assert np.isclose(res_before.fraction, 0.0)
    assert np.isclose(res_before.along_track_m, 0.0)

    # Query after end: (120, -5)
    p_after = np.array([120.0, -5.0])
    res_after = project_point_to_segment(p_after, a, b)
    assert np.allclose(res_after.projected_point_enu, b)
    assert res_after.is_clamped
    assert np.isclose(res_after.fraction, 1.0)
    assert np.isclose(res_after.along_track_m, 100.0)


def test_project_point_degenerate_segment():
    # Identical start and end
    a = np.array([10.0, 20.0])
    p = np.array([13.0, 24.0])  # dist = 5
    res = project_point_to_segment(p, a, a)
    assert np.allclose(res.projected_point_enu, a)
    assert np.isclose(res.abs_distance_m, 5.0)
    assert res.is_clamped
