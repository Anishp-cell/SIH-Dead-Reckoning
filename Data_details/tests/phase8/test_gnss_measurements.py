"""
Unit tests for Phase 8 GNSS measurement models, observation equations,
and finite-difference verification of 3x15 Jacobians.
"""

import numpy as np
import pytest

from Data_details.src.phase5.core.quaternion import (
    quaternion_to_rotation_matrix,
    inject_small_angle_error,
)
from Data_details.src.phase8.fusion.gnss_measurement import (
    GNSSMeasurementModel,
    GNSSMeasurementConfig,
    skew_symmetric,
)


def test_position_jacobian_finite_difference():
    """
    Rigorously validates analytical 3x15 Jacobian H_p against numerical central differences.
    Tolerance: max absolute error < 1e-6.
    """
    lever_arm = np.array([0.8, -0.3, 1.2], dtype=np.float64)
    cfg = GNSSMeasurementConfig(lever_arm_b=lever_arm)
    model = GNSSMeasurementModel(cfg)

    # Nominal state
    p0 = np.array([120.5, -45.2, 10.3], dtype=np.float64)
    # Pitch/roll/yaw attitude quaternion
    q0 = np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float64)  # ~45 deg yaw

    H_analytical = model.position_jacobian(q0, lever_arm)
    assert H_analytical.shape == (3, 15)

    eps = 1e-6

    # 1. Test position columns 0:3
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = eps
        p_plus = p0 + dp
        p_minus = p0 - dp

        h_plus = model.predict_position(p_plus, q0, lever_arm)
        h_minus = model.predict_position(p_minus, q0, lever_arm)
        dh_num = (h_plus - h_minus) / (2.0 * eps)

        err = np.max(np.abs(dh_num - H_analytical[:, i]))
        assert err < 1e-6, f"Position block column {i} finite diff error: {err}"

    # 2. Test velocity columns 3:6 (must be zero)
    assert np.allclose(H_analytical[:, 3:6], 0.0)

    # 3. Test attitude columns 6:9
    for k in range(3):
        dtheta = np.zeros(3)
        dtheta[k] = eps

        q_plus = inject_small_angle_error(q0, dtheta)
        q_minus = inject_small_angle_error(q0, -dtheta)

        h_plus = model.predict_position(p0, q_plus, lever_arm)
        h_minus = model.predict_position(p0, q_minus, lever_arm)
        dh_num = (h_plus - h_minus) / (2.0 * eps)

        err = np.max(np.abs(dh_num - H_analytical[:, 6 + k]))
        assert err < 1e-6, f"Attitude block column {6+k} finite diff error: {err}"

    # 4. Test bias columns 9:15 (must be zero)
    assert np.allclose(H_analytical[:, 9:15], 0.0)


def test_velocity_jacobian_finite_difference():
    """
    Rigorously validates analytical 3x15 Jacobian H_v against numerical central differences.
    Tolerance: max absolute error < 1e-6.
    """
    lever_arm = np.array([0.5, 0.1, 0.9], dtype=np.float64)
    cfg = GNSSMeasurementConfig(lever_arm_b=lever_arm)
    model = GNSSMeasurementModel(cfg)

    v0 = np.array([14.2, 3.1, -0.4], dtype=np.float64)
    q0 = np.array([0.9659, 0.0, 0.2588, 0.0], dtype=np.float64)  # ~30 deg pitch
    omega0 = np.array([0.05, -0.02, 0.15], dtype=np.float64)

    H_analytical = model.velocity_jacobian(q0, omega0, lever_arm)
    assert H_analytical.shape == (3, 15)

    eps = 1e-6

    # 1. Test velocity columns 3:6
    for i in range(3):
        dv = np.zeros(3)
        dv[i] = eps
        v_plus = v0 + dv
        v_minus = v0 - dv

        h_plus = model.predict_velocity(v_plus, q0, omega0, lever_arm)
        h_minus = model.predict_velocity(v_minus, q0, omega0, lever_arm)
        dh_num = (h_plus - h_minus) / (2.0 * eps)

        err = np.max(np.abs(dh_num - H_analytical[:, 3 + i]))
        assert err < 1e-6, f"Velocity block column {3+i} finite diff error: {err}"

    # 2. Test attitude columns 6:9
    for k in range(3):
        dtheta = np.zeros(3)
        dtheta[k] = eps

        q_plus = inject_small_angle_error(q0, dtheta)
        q_minus = inject_small_angle_error(q0, -dtheta)

        h_plus = model.predict_velocity(v0, q_plus, omega0, lever_arm)
        h_minus = model.predict_velocity(v0, q_minus, omega0, lever_arm)
        dh_num = (h_plus - h_minus) / (2.0 * eps)

        err = np.max(np.abs(dh_num - H_analytical[:, 6 + k]))
        assert err < 1e-6, f"Attitude block column {6+k} finite diff error: {err}"

    # 3. Test gyro bias columns 12:15
    # Gyro measurement model: omega_true = omega_meas - bg -> delta_omega = -delta_bg
    for j in range(3):
        dbg = np.zeros(3)
        dbg[j] = eps

        # Perturbation in gyro bias reduces true angular rate by eps
        omega_plus = omega0 - dbg
        omega_minus = omega0 + dbg

        h_plus = model.predict_velocity(v0, q0, omega_plus, lever_arm)
        h_minus = model.predict_velocity(v0, q0, omega_minus, lever_arm)
        dh_num = (h_plus - h_minus) / (2.0 * eps)

        err = np.max(np.abs(dh_num - H_analytical[:, 12 + j]))
        assert err < 1e-6, f"Gyro bias block column {12+j} finite diff error: {err}"


def test_zero_lever_arm_simplification():
    """Verifies that with zero lever arm, H_p and H_v collapse to canonical standard form."""
    cfg = GNSSMeasurementConfig(lever_arm_b=np.zeros(3))
    model = GNSSMeasurementModel(cfg)

    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.1, 0.2, 0.3])

    H_p = model.position_jacobian(q)
    assert np.allclose(H_p[:, 0:3], np.eye(3))
    assert np.allclose(H_p[:, 3:], 0.0)

    H_v = model.velocity_jacobian(q, omega)
    assert np.allclose(H_v[:, 0:3], 0.0)
    assert np.allclose(H_v[:, 3:6], np.eye(3))
    assert np.allclose(H_v[:, 6:], 0.0)
