"""
Phase 6 Unit Tests: Non-Holonomic Constraints (NHC) and Analytical Jacobian Verification.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import inject_small_angle_error
from Data_details.src.phase6.constraints.nhc import (
    predict_nhc_velocity,
    compute_nhc_jacobian,
    apply_nhc_update,
)


def perturb_nominal_state(state: NominalState, delta_x: np.ndarray) -> NominalState:
    """Applies 15-state indirect error vector to nominal state."""
    return NominalState(
        p=state.p + delta_x[0:3],
        v=state.v + delta_x[3:6],
        q=inject_small_angle_error(state.q, delta_x[6:9]),
        ba=state.ba + delta_x[9:12],
        bg=state.bg + delta_x[12:15],
    )


class TestPhase6NHC:

    def test_nhc_velocity_prediction(self):
        # Car traveling East at 10 m/s with heading East (q = identity)
        # Body frame: x=Fwd=East, y=Right=South, z=Up=Up
        state = NominalState(
            p=np.zeros(3),
            v=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        h_nhc, v_b = predict_nhc_velocity(state)
        # v_b should be [10.0, 0.0, 0.0], so lateral=0, vertical=0
        assert np.isclose(v_b[0], 10.0)
        assert np.isclose(h_nhc[0], 0.0)
        assert np.isclose(h_nhc[1], 0.0)

        # Now car has non-zero upward velocity 2.0 m/s and lateral right 1.5 m/s
        state.v = np.array([10.0, -1.5, 2.0])
        h_nhc, v_b = predict_nhc_velocity(state)
        assert np.isclose(v_b[0], 10.0)
        assert np.isclose(v_b[1], -1.5)
        assert np.isclose(v_b[2], 2.0)
        assert np.allclose(h_nhc, [-1.5, 2.0])

    def test_nhc_analytical_jacobian_vs_finite_difference(self):
        """
        Validates analytical 2x15 Jacobian H_NHC against central finite differences.
        Must match within numerical precision (< 1e-5).
        """
        # Realistic nominal state with 3D orientation, velocity, and biases
        state = NominalState(
            p=np.array([15.2, -30.4, 4.2]),
            v=np.array([12.5, 6.3, -0.4]),
            q=np.array([0.866, 0.1, -0.2, 0.463]),  # Arbitrary 3D orientation
            ba=np.array([0.02, -0.01, 0.04]),
            bg=np.array([1e-3, -2e-3, 5e-4]),
        )
        # Normalize q
        state.q = state.q / np.linalg.norm(state.q)

        H_analytical = compute_nhc_jacobian(state)
        assert H_analytical.shape == (2, 15)

        H_numerical = np.zeros((2, 15), dtype=np.float64)
        epsilon = 1e-6

        for i in range(15):
            dx_plus = np.zeros(15)
            dx_minus = np.zeros(15)
            dx_plus[i] = epsilon
            dx_minus[i] = -epsilon

            state_plus = perturb_nominal_state(state, dx_plus)
            state_minus = perturb_nominal_state(state, dx_minus)

            h_plus, _ = predict_nhc_velocity(state_plus)
            h_minus, _ = predict_nhc_velocity(state_minus)

            H_numerical[:, i] = (h_plus - h_minus) / (2.0 * epsilon)

        # Difference must be strictly within tolerance
        diff = np.abs(H_analytical - H_numerical)
        max_diff = float(np.max(diff))
        assert max_diff < 1e-5, f"NHC Jacobian mismatch: max diff = {max_diff}"

        # Unobservable blocks: position, accel bias, gyro bias must be exactly zero
        assert np.allclose(H_analytical[:, 0:3], 0.0, atol=1e-12)
        assert np.allclose(H_analytical[:, 9:15], 0.0, atol=1e-12)

    def test_nhc_update_reduces_lateral_velocity(self):
        # State with significant lateral velocity error (v_lat = 2.0 m/s)
        state = NominalState(
            p=np.zeros(3),
            v=np.array([10.0, 2.0, 0.0]),  # Non-zero lateral in ENU
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        P0 = np.eye(15, dtype=np.float64) * 0.5
        R_nhc = np.diag([0.04, 0.04])  # sigma = 0.2 m/s

        res = apply_nhc_update(state, P0, R_nhc)
        assert res.accepted is True
        # Posterior lateral velocity should be substantially closer to zero
        v_b_plus = res.state_plus.v
        assert abs(v_b_plus[1]) < abs(state.v[1])
        # Covariance diagonal for lateral velocity should shrink
        assert res.P_plus[4, 4] < P0[4, 4]
