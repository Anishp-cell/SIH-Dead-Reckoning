"""
Phase 2 Unit Tests: Quaternion Algebra, Attitude Propagation, and Complementary Filter.
Uses purely synthetic mathematical data to guarantee fast, deterministic test execution.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from Data_details.src.attitude_estimation import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_euler_deg,
    euler_to_quaternion,
    estimate_tilt_from_accel,
    propagate_quaternion_gyro,
    ComplementaryAttitudeEstimator,
)


class TestQuaternionAlgebra:
    """Tests core quaternion operations and conversions."""

    def test_quaternion_normalization(self):
        """Quaternion normalization must produce unit norm."""
        q_unnorm = np.array([2.0, 3.0, -1.0, 5.0])
        q_norm = quaternion_normalize(q_unnorm)
        assert np.isclose(np.linalg.norm(q_norm), 1.0, atol=1e-12)

    def test_quaternion_multiplication_identity(self):
        """Multiplying by identity quaternion [1, 0, 0, 0] leaves quaternion unchanged."""
        q = quaternion_normalize(np.array([0.7071, 0.0, 0.7071, 0.0]))
        q_ident = np.array([1.0, 0.0, 0.0, 0.0])
        q_prod = quaternion_multiply(q, q_ident)
        np.testing.assert_allclose(q_prod, q, atol=1e-10)

    def test_rotation_matrix_orthonormal(self):
        """Rotation matrix R(q) must satisfy R * R^T = I and det(R) = +1."""
        q = euler_to_quaternion(np.radians(25.0), np.radians(-15.0), np.radians(40.0))
        R = quaternion_to_rotation_matrix(q)
        
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_matrix_quaternion_roundtrip(self):
        """Converting quaternion -> matrix -> quaternion must preserve rotation."""
        q_orig = euler_to_quaternion(np.radians(10.0), np.radians(20.0), np.radians(30.0))
        R = quaternion_to_rotation_matrix(q_orig)
        q_rec = rotation_matrix_to_quaternion(R)
        
        # Quaternions q and -q represent identical rotations
        dot = np.abs(np.dot(q_orig, q_rec))
        assert np.isclose(dot, 1.0, atol=1e-7)

    def test_euler_conversion(self):
        """Known Euler angles (Roll=15°, Pitch=-30°, Yaw=45°) must be accurately recovered."""
        r_in, p_in, y_in = 15.0, -30.0, 45.0
        q = euler_to_quaternion(np.radians(r_in), np.radians(p_in), np.radians(y_in))
        r_out, p_out, y_out = quaternion_to_euler_deg(q)
        
        assert np.isclose(r_out, r_in, atol=1e-5)
        assert np.isclose(p_out, p_in, atol=1e-5)
        assert np.isclose(y_out, y_in, atol=1e-5)


class TestAttitudePropagation:
    """Tests gyro-based integration and tilt estimation."""

    def test_zero_gyro_preserves_orientation(self):
        """Zero angular velocity must leave attitude quaternion invariant."""
        q0 = euler_to_quaternion(np.radians(10.0), np.radians(0.0), np.radians(90.0))
        omega_zero = np.array([0.0, 0.0, 0.0])
        q1 = propagate_quaternion_gyro(q0, omega_zero, dt=0.1)
        
        np.testing.assert_allclose(q1, q0, atol=1e-10)

    def test_known_constant_rotation(self):
        """Rotating around Z-axis at 90 deg/s for 1 second must produce a 90° yaw rotation."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # zero orientation
        omega_z = np.array([0.0, 0.0, np.radians(90.0)])  # 90 deg/s around Z
        
        # Propagate 10 steps of dt=0.1s (total 1.0s)
        q = q0.copy()
        for _ in range(10):
            q = propagate_quaternion_gyro(q, omega_z, dt=0.1)
            
        r, p, y = quaternion_to_euler_deg(q)
        assert np.isclose(r, 0.0, atol=1e-4)
        assert np.isclose(p, 0.0, atol=1e-4)
        assert np.isclose(y, 90.0, atol=1e-3)

    def test_tilt_from_gravity_level(self):
        """When phone is level (a = [0, 0, 9.81]), tilt angles must be zero."""
        roll, pitch = estimate_tilt_from_accel(0.0, 0.0, 9.80665)
        assert np.isclose(roll, 0.0, atol=1e-7)
        assert np.isclose(pitch, 0.0, atol=1e-7)

    def test_tilt_from_gravity_pitched(self):
        """Pitching up by 30° creates ax = -g*sin(30°), az = g*cos(30°)."""
        g = 9.80665
        angle_rad = np.radians(30.0)
        ax = -g * np.sin(angle_rad)
        ay = 0.0
        az = g * np.cos(angle_rad)
        
        roll, pitch = estimate_tilt_from_accel(ax, ay, az)
        assert np.isclose(roll, 0.0, atol=1e-5)
        assert np.isclose(np.degrees(pitch), 30.0, atol=1e-5)


class TestComplementaryFilter:
    """Tests the complementary attitude estimator filter."""

    def test_filter_converges_to_static_gravity(self):
        """Under static conditions with small initial error, filter converges to gravity tilt."""
        estimator = ComplementaryAttitudeEstimator(alpha=0.9)
        estimator.initialize(0.0, 0.0, 9.80665, initial_yaw_deg=0.0)
        
        # Introduce a steady 10 deg pitch tilt in accel readings
        g = 9.80665
        ax = -g * np.sin(np.radians(10.0))
        ay = 0.0
        az = g * np.cos(np.radians(10.0))
        acc = np.array([ax, ay, az])
        w = np.array([0.0, 0.0, 0.0])
        
        for _ in range(50):
            q = estimator.update(acc, w, dt=0.1)
            
        r, p, y = quaternion_to_euler_deg(q)
        assert np.isclose(p, 10.0, atol=0.5)
