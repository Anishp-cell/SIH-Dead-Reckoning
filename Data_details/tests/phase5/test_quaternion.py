"""
Phase 5 Unit Tests: Quaternion Mathematics & SO(3) Operations.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.quaternion import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_rotate_vector,
    quaternion_integrate,
    small_angle_to_quaternion,
    inject_small_angle_error,
    quaternion_to_euler_zyx,
    euler_zyx_to_quaternion,
)


class TestPhase5Quaternion:

    def test_identity_quaternion(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        R_id = quaternion_to_rotation_matrix(q_id)
        assert np.allclose(R_id, np.eye(3), atol=1e-12)

        v = np.array([1.2, -3.4, 5.6])
        v_rot = quaternion_rotate_vector(q_id, v)
        assert np.allclose(v_rot, v, atol=1e-12)

    def test_90_deg_rotations(self):
        # 90 deg rotation around z-axis: yaw = pi/2
        # Body X [1, 0, 0] should rotate to Nav Y [0, 1, 0]
        q_z90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        R_z90 = quaternion_to_rotation_matrix(q_z90)

        v_body_x = np.array([1.0, 0.0, 0.0])
        v_nav = R_z90 @ v_body_x
        assert np.allclose(v_nav, [0.0, 1.0, 0.0], atol=1e-7)

        # 90 deg rotation around x-axis: roll = pi/2
        # Body Y [0, 1, 0] should rotate to Nav Z [0, 0, 1]
        q_x90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        v_body_y = np.array([0.0, 1.0, 0.0])
        v_nav_x90 = quaternion_rotate_vector(q_x90, v_body_y)
        assert np.allclose(v_nav_x90, [0.0, 0.0, 1.0], atol=1e-7)

    def test_rotation_matrix_orthonormality(self):
        # Arbitrary attitude
        q = quaternion_normalize(np.array([0.5, -0.3, 0.8, -0.1]))
        R = quaternion_to_rotation_matrix(q)

        # R * R^T = I
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        # det(R) = +1
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_roundtrip_rotation_matrix_and_quaternion(self):
        q_orig = quaternion_normalize(np.array([0.7071, 0.2, -0.4, 0.5]))
        R = quaternion_to_rotation_matrix(q_orig)
        q_rec = rotation_matrix_to_quaternion(R)

        # Quaternions are identical up to sign: q == -q
        dot = np.abs(np.dot(q_orig, q_rec))
        assert np.isclose(dot, 1.0, atol=1e-7)

    def test_quaternion_multiplication_and_conjugate(self):
        q1 = quaternion_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q1_conj = quaternion_conjugate(q1)
        q_id = quaternion_multiply(q1, q1_conj)
        assert np.allclose(q_id, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_small_angle_injection(self):
        q_nom = np.array([1.0, 0.0, 0.0, 0.0])
        dtheta = np.array([0.01, -0.02, 0.03])
        q_pert = inject_small_angle_error(q_nom, dtheta)

        # Check unit length
        assert np.isclose(np.linalg.norm(q_pert), 1.0, atol=1e-12)

        # Vector vector part should be approximately 0.5 * dtheta
        assert np.allclose(q_pert[1:], 0.5 * dtheta, atol=1e-4)

    def test_quaternion_integration(self):
        # Pure yaw rate 1 rad/s around z-axis for 1 second -> delta angle = 1 rad
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 1.0])
        dt = 1.0

        q1 = quaternion_integrate(q0, omega, dt)
        expected_q = np.array([np.cos(0.5), 0.0, 0.0, np.sin(0.5)])
        assert np.allclose(q1, expected_q, atol=1e-7)

    def test_euler_zyx_roundtrip(self):
        roll, pitch, yaw = 0.15, -0.25, 1.35
        q = euler_zyx_to_quaternion(roll, pitch, yaw)
        r_rec, p_rec, y_rec = quaternion_to_euler_zyx(q)

        assert np.isclose(roll, r_rec, atol=1e-7)
        assert np.isclose(pitch, p_rec, atol=1e-7)
        assert np.isclose(yaw, y_rec, atol=1e-7)
