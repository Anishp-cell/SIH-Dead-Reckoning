"""
Phase 6 Unit Tests: Zero-Velocity Update (ZUPT) and Zero-Angular-Rate Update (ZARU).
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import inject_small_angle_error
from Data_details.src.phase6.constraints.zupt import (
    predict_zupt,
    compute_zupt_jacobian,
    apply_zupt_update,
)
from Data_details.src.phase6.constraints.zaru import (
    predict_zaru,
    compute_zaru_jacobian,
    apply_zaru_update,
)


def perturb_nominal_state(state: NominalState, delta_x: np.ndarray) -> NominalState:
    return NominalState(
        p=state.p + delta_x[0:3],
        v=state.v + delta_x[3:6],
        q=inject_small_angle_error(state.q, delta_x[6:9]),
        ba=state.ba + delta_x[9:12],
        bg=state.bg + delta_x[12:15],
    )


class TestPhase6ZUPTZARU:

    def test_zupt_analytical_jacobian_vs_finite_difference(self):
        state = NominalState(
            p=np.array([1.0, 2.0, 3.0]),
            v=np.array([0.1, -0.2, 0.05]),
            q=np.array([0.9238, 0.0, 0.0, 0.3826]),
            ba=np.array([0.01, -0.01, 0.02]),
            bg=np.array([1e-3, 5e-4, -1e-3]),
        )

        H_analytical = compute_zupt_jacobian(state)
        assert H_analytical.shape == (3, 15)

        H_numerical = np.zeros((3, 15), dtype=np.float64)
        epsilon = 1e-6

        for i in range(15):
            dx_plus = np.zeros(15)
            dx_minus = np.zeros(15)
            dx_plus[i] = epsilon
            dx_minus[i] = -epsilon

            s_plus = perturb_nominal_state(state, dx_plus)
            s_minus = perturb_nominal_state(state, dx_minus)

            h_plus = predict_zupt(s_plus)
            h_minus = predict_zupt(s_minus)

            H_numerical[:, i] = (h_plus - h_minus) / (2.0 * epsilon)

        assert np.allclose(H_analytical, H_numerical, atol=1e-8)

    def test_zaru_analytical_jacobian_vs_finite_difference(self):
        state = NominalState(
            p=np.array([1.0, 2.0, 3.0]),
            v=np.array([0.1, -0.2, 0.05]),
            q=np.array([0.9238, 0.0, 0.0, 0.3826]),
            ba=np.array([0.01, -0.01, 0.02]),
            bg=np.array([1e-3, 5e-4, -1e-3]),
        )

        H_analytical = compute_zaru_jacobian(state)
        assert H_analytical.shape == (3, 15)

        H_numerical = np.zeros((3, 15), dtype=np.float64)
        epsilon = 1e-6

        for i in range(15):
            dx_plus = np.zeros(15)
            dx_minus = np.zeros(15)
            dx_plus[i] = epsilon
            dx_minus[i] = -epsilon

            s_plus = perturb_nominal_state(state, dx_plus)
            s_minus = perturb_nominal_state(state, dx_minus)

            h_plus = predict_zaru(s_plus)
            h_minus = predict_zaru(s_minus)

            H_numerical[:, i] = (h_plus - h_minus) / (2.0 * epsilon)

        assert np.allclose(H_analytical, H_numerical, atol=1e-8)

    def test_zupt_drives_velocity_to_zero(self):
        # Small residual velocity v = [0.4, -0.3, 0.1]
        state = NominalState(
            p=np.array([10.0, 20.0, 0.0]),
            v=np.array([0.4, -0.3, 0.1]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        P0 = np.eye(15, dtype=np.float64) * 0.25

        res = apply_zupt_update(state, P0, sigma_zupt=0.02)
        assert res.accepted is True
        # Posterior velocity norm should be clamped close to 0
        assert np.linalg.norm(res.state_plus.v) < 0.05
        # Velocity covariance diagonal should collapse
        assert res.P_plus[3, 3] < 0.01

    def test_zaru_estimates_gyro_bias_without_attitude_reset(self):
        # Sensor measures small constant bias [0.005, -0.003, 0.008] rad/s while stationary
        omega_meas = np.array([0.005, -0.003, 0.008])
        state = NominalState(
            p=np.zeros(3),
            v=np.zeros(3),
            q=np.array([0.7071, 0.0, 0.0, 0.7071]),  # Heading = 90 deg
            ba=np.zeros(3),
            bg=np.zeros(3),  # Initial estimate is zero
        )
        P0 = np.eye(15, dtype=np.float64) * 1e-4

        res = apply_zaru_update(state, P0, omega_meas, sigma_zaru=0.002)
        assert res.accepted is True
        # Gyro bias estimate should shift towards measured bias
        assert np.all(np.abs(res.state_plus.bg) > 1e-4)
        assert np.isclose(res.state_plus.bg[0], 0.005, atol=1e-3)
        assert np.isclose(res.state_plus.bg[2], 0.008, atol=1e-3)
        # Orientation quaternion q should NOT be altered or reset by ZARU
        assert np.allclose(res.state_plus.q, state.q, atol=1e-6)
