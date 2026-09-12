"""
Phase 3 Unit Tests: Mathematical Audit Verification & Coordinate Rotations.
Verifies frame conversions, cardinal direction projections (North, East, South, West),
quaternion vector rotations, gravity leveling, and theoretical error propagation formulas.
"""

import numpy as np
import pytest

from Data_details.src.attitude_estimation import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    euler_to_quaternion,
    quaternion_to_euler_deg,
    estimate_tilt_from_accel,
)


class TestMathematicalAuditRotations:
    """Verifies coordinate frame transformations and mathematical audit findings."""

    def test_cardinal_azimuth_to_enu_projections(self):
        """
        Verifies that vehicle forward motion at cardinal azimuth headings projects
        into the correct ENU (East, North, Up) axes.
        
        Navigation Azimuth:
        - 0 deg (North) -> v_east = 0, v_north = +v
        - 90 deg (East)  -> v_east = +v, v_north = 0
        - 180 deg (South) -> v_east = 0, v_north = -v
        - 270 deg (West)  -> v_east = -v, v_north = 0
        """
        def project_fwd_to_enu(v_fwd: float, azimuth_deg: float):
            psi_rad = np.radians(azimuth_deg)
            v_east = v_fwd * np.sin(psi_rad)
            v_north = v_fwd * np.cos(psi_rad)
            return v_east, v_north

        v = 15.0  # 15 m/s forward speed

        # 1. North
        ve, vn = project_fwd_to_enu(v, 0.0)
        assert np.isclose(ve, 0.0, atol=1e-6)
        assert np.isclose(vn, v, atol=1e-6)

        # 2. East
        ve, vn = project_fwd_to_enu(v, 90.0)
        assert np.isclose(ve, v, atol=1e-6)
        assert np.isclose(vn, 0.0, atol=1e-6)

        # 3. South
        ve, vn = project_fwd_to_enu(v, 180.0)
        assert np.isclose(ve, 0.0, atol=1e-6)
        assert np.isclose(vn, -v, atol=1e-6)

        # 4. West
        ve, vn = project_fwd_to_enu(v, 270.0)
        assert np.isclose(ve, -v, atol=1e-6)
        assert np.isclose(vn, 0.0, atol=1e-6)

    def test_azimuth_to_cartesian_yaw_conversion(self):
        """
        Verifies the conversion: psi_cartesian = 90 deg - psi_azimuth.
        In Cartesian coordinates (X=East, Y=North):
        - North (azimuth 0) -> Cartesian angle 90 deg (along Y)
        - East (azimuth 90) -> Cartesian angle 0 deg (along X)
        - South (azimuth 180) -> Cartesian angle -90 / 270 deg (along -Y)
        - West (azimuth 270) -> Cartesian angle 180 deg (along -X)
        """
        test_cases = [
            (0.0, 90.0),    # North
            (90.0, 0.0),    # East
            (180.0, -90.0), # South
            (270.0, 180.0), # West
        ]
        for az_deg, expected_cart_deg in test_cases:
            cart_rad = np.pi / 2 - np.radians(az_deg)
            cart_deg = np.degrees(cart_rad)
            # Normalize to [-180, 180]
            cart_norm = (cart_deg + 180) % 360 - 180
            exp_norm = (expected_cart_deg + 180) % 360 - 180
            assert np.isclose(cart_norm, exp_norm, atol=1e-6)

    def test_quaternion_rotation_matrix_orthonormality(self):
        """Verifies that R(q) is strictly in SO(3): det(R) = +1 and R * R^T = I."""
        angles = [(15.0, -25.0, 40.0), (0.0, 85.0, -120.0), (-45.0, 10.0, 180.0)]
        for r_deg, p_deg, y_deg in angles:
            q = euler_to_quaternion(np.radians(r_deg), np.radians(p_deg), np.radians(y_deg))
            R = quaternion_to_rotation_matrix(q)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)

    def test_gravity_removal_stationary_leveled(self):
        """
        Stationary accelerometer measuring upward specific force [0, 0, g]
        must produce zero dynamic acceleration after gravity removal.
        """
        g = 9.80665
        f_meas = np.array([0.0, 0.0, g])
        q_level = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        R = quaternion_to_rotation_matrix(q_level)

        a_dyn = R @ f_meas - np.array([0.0, 0.0, g])
        assert np.allclose(a_dyn, np.zeros(3), atol=1e-6)

    def test_gravity_leakage_analytical_formula(self):
        """
        Tests the analytical derivation e_p(t) = 1/2 * g * sin(theta) * t^2.
        For theta = 1.0 deg and t = 60s, error must equal ~308.09 m.
        """
        g = 9.80665
        theta_rad = np.radians(1.0)
        t = 60.0

        a_leak = g * np.sin(theta_rad)
        e_p = 0.5 * a_leak * (t ** 2)

        assert 305.0 < e_p < 310.0
        assert np.isclose(e_p, 308.09, atol=0.1)

    def test_constant_acceleration_bias_drift_formula(self):
        """
        Tests error propagation: e_p(t) = 1/2 * b_a * t^2.
        For b_a = 0.05 m/s^2 (typical low-cost smartphone accelerometer offset)
        and t = 60s, error must equal 90.0 m.
        """
        b_a = 0.05  # m/s^2
        t = 60.0
        e_p = 0.5 * b_a * (t ** 2)
        assert np.isclose(e_p, 90.0, atol=1e-6)
