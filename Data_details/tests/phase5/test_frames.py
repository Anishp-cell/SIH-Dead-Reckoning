"""
Phase 5 Unit Tests: Coordinate Frames & Gravity Conventions.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.frames import (
    GRAVITY_MAGNITUDE,
    GRAVITY_VECTOR_ENU,
    body_to_nav,
    nav_to_body,
    geographic_to_enu_yaw_deg,
    enu_yaw_to_geographic_deg,
    estimate_tilt_from_static_accel,
    initial_leveling_quaternion,
)
from Data_details.src.phase5.core.quaternion import (
    quaternion_to_rotation_matrix,
    euler_zyx_to_quaternion,
)


class TestPhase5Frames:

    def test_gravity_vector_definition(self):
        assert np.isclose(GRAVITY_MAGNITUDE, 9.80665, atol=1e-5)
        assert np.allclose(GRAVITY_VECTOR_ENU, [0.0, 0.0, -9.80665], atol=1e-5)

    def test_level_stationary_gravity_cancellation(self):
        # A stationary vehicle on a flat horizontal road
        # Accelerometer measures upward normal force: [0, 0, +g]^T
        f_b_level = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        q_level = np.array([1.0, 0.0, 0.0, 0.0])

        f_n = body_to_nav(f_b_level, q_level)
        assert np.allclose(f_n, [0.0, 0.0, GRAVITY_MAGNITUDE], atol=1e-6)

        # Total acceleration in navigation frame: a_n = f_n + g_n
        a_n = f_n + GRAVITY_VECTOR_ENU
        assert np.allclose(a_n, [0.0, 0.0, 0.0], atol=1e-6)

    def test_body_nav_roundtrip(self):
        q = euler_zyx_to_quaternion(0.2, -0.4, 1.1)
        v_body_orig = np.array([10.5, -2.3, 0.8])

        v_nav = body_to_nav(v_body_orig, q)
        v_body_rec = nav_to_body(v_nav, q)

        assert np.allclose(v_body_orig, v_body_rec, atol=1e-12)

    def test_geographic_heading_to_enu_yaw_mapping(self):
        # North: 0 deg GPS -> +90 deg ENU
        assert np.isclose(geographic_to_enu_yaw_deg(0.0), 90.0, atol=1e-7)

        # East: 90 deg GPS -> 0 deg ENU
        assert np.isclose(geographic_to_enu_yaw_deg(90.0), 0.0, atol=1e-7)

        # South: 180 deg GPS -> -90 deg ENU
        assert np.isclose(geographic_to_enu_yaw_deg(180.0), -90.0, atol=1e-7)

        # West: 270 deg GPS -> 180 deg ENU
        assert np.isclose(np.abs(geographic_to_enu_yaw_deg(270.0)), 180.0, atol=1e-7)

    def test_enu_yaw_to_geographic_heading_roundtrip(self):
        for deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            enu_yaw = geographic_to_enu_yaw_deg(deg)
            gps_heading_rec = enu_yaw_to_geographic_deg(enu_yaw)
            assert np.isclose(deg, gps_heading_rec, atol=1e-7)

    def test_initial_leveling_quaternion(self):
        # Accelerometer measures [0, 0, g] when flat, vehicle facing East (90 deg GPS)
        acc_b = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        q_init = initial_leveling_quaternion(acc_b, initial_gps_heading_deg=90.0)

        # Facing East in ENU means yaw_enu = 0 deg!
        # Rotation matrix should have forward body X [1, 0, 0] along East [1, 0, 0]
        R = quaternion_to_rotation_matrix(q_init)
        v_fwd_nav = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(v_fwd_nav, [1.0, 0.0, 0.0], atol=1e-6)
