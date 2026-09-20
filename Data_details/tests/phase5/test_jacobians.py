"""
Phase 5 Numerical Jacobian Validation: Finite-Difference Verification of F and H.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.state import NominalState, ErrorState
from Data_details.src.phase5.core.propagation import (
    propagate_nominal_ins,
    compute_discrete_F_matrix,
)
from Data_details.src.phase5.core.measurement import (
    predict_forward_speed,
    compute_speed_jacobian,
)
from Data_details.src.phase5.core.quaternion import (
    inject_small_angle_error,
    quaternion_to_rotation_matrix,
    quaternion_multiply,
    quaternion_conjugate,
)


def perturb_nominal_state(state: NominalState, delta_x: np.ndarray) -> NominalState:
    """Applies error vector delta_x (15,) to nominal state."""
    dp = delta_x[0:3]
    dv = delta_x[3:6]
    dtheta = delta_x[6:9]
    dba = delta_x[9:12]
    dbg = delta_x[12:15]

    return NominalState(
        p=state.p + dp,
        v=state.v + dv,
        q=inject_small_angle_error(state.q, dtheta),
        ba=state.ba + dba,
        bg=state.bg + dbg,
    )


def state_minus_nominal(state_pert: NominalState, state_nom: NominalState) -> np.ndarray:
    """Computes error vector (15,) from perturbed state minus nominal state."""
    dp = state_pert.p - state_nom.p
    dv = state_pert.v - state_nom.v

    # Quaternion difference in body frame: delta_q = q_nom^* ⊗ q_pert
    dq = quaternion_multiply(quaternion_conjugate(state_nom.q), state_pert.q)
    # delta_theta ≈ 2 * vector_part
    # Account for sign flip if scalar part is negative
    if dq[0] < 0:
        dq = -dq
    dtheta = 2.0 * dq[1:4]

    dba = state_pert.ba - state_nom.ba
    dbg = state_pert.bg - state_nom.bg

    return np.concatenate([dp, dv, dtheta, dba, dbg])


class TestPhase5Jacobians:

    def test_measurement_jacobian_H_numerical_check(self):
        """
        Validates analytical measurement Jacobian H against central finite differences.
        """
        # Set up realistic nominal state
        state = NominalState(
            p=np.array([12.5, -45.2, 3.1]),
            v=np.array([8.2, 5.4, -0.6]),
            q=np.array([0.7071, 0.1, -0.3, 0.6324]),  # Arbitrary 3D orientation
            ba=np.array([0.02, -0.01, 0.05]),
            bg=np.array([1e-3, -2e-3, 5e-4]),
        )

        H_analytical = compute_speed_jacobian(state)
        H_numerical = np.zeros((1, 15), dtype=np.float64)

        epsilon = 1e-6
        for i in range(15):
            dx_plus = np.zeros(15)
            dx_minus = np.zeros(15)
            dx_plus[i] = epsilon
            dx_minus[i] = -epsilon

            state_plus = perturb_nominal_state(state, dx_plus)
            state_minus = perturb_nominal_state(state, dx_minus)

            v_fwd_plus, _ = predict_forward_speed(state_plus)
            v_fwd_minus, _ = predict_forward_speed(state_minus)

            H_numerical[0, i] = (v_fwd_plus - v_fwd_minus) / (2.0 * epsilon)

        # Compare analytical vs finite difference
        # Max difference should be < 1e-5
        diff = np.abs(H_analytical - H_numerical)
        max_diff = np.max(diff)
        assert max_diff < 1e-5, f"Jacobian H mismatch: max diff = {max_diff}"

        # Verify zero blocks: position, accel bias, gyro bias are zero
        assert np.allclose(H_analytical[0, 0:3], 0.0, atol=1e-12)
        assert np.allclose(H_analytical[0, 9:15], 0.0, atol=1e-12)

    def test_state_transition_jacobian_F_numerical_check(self):
        """
        Validates analytical discrete state transition matrix F against finite differences.
        """
        state = NominalState(
            p=np.array([0.0, 0.0, 0.0]),
            v=np.array([5.0, -2.0, 0.5]),
            q=np.array([0.9238, 0.0, 0.0, 0.3826]),  # Yaw = 45 deg
            ba=np.array([0.01, -0.02, 0.03]),
            bg=np.array([1e-3, 0.0, -2e-3]),
        )
        f_meas = np.array([1.5, 0.2, 9.80665])
        omega_meas = np.array([0.02, -0.01, 0.1])
        dt = 0.1

        # Base nominal propagation
        nom_next, f_b_corr, omega_b_corr = propagate_nominal_ins(state, f_meas, omega_meas, dt)
        F_analytical = compute_discrete_F_matrix(state, f_b_corr, omega_b_corr, dt)

        F_numerical = np.zeros((15, 15), dtype=np.float64)
        epsilon = 1e-6

        for i in range(15):
            dx_plus = np.zeros(15)
            dx_minus = np.zeros(15)
            dx_plus[i] = epsilon
            dx_minus[i] = -epsilon

            state_plus = perturb_nominal_state(state, dx_plus)
            state_minus = perturb_nominal_state(state, dx_minus)

            pert_next_plus, _, _ = propagate_nominal_ins(state_plus, f_meas, omega_meas, dt)
            pert_next_minus, _, _ = propagate_nominal_ins(state_minus, f_meas, omega_meas, dt)

            dx_next_plus = state_minus_nominal(pert_next_plus, nom_next)
            dx_next_minus = state_minus_nominal(pert_next_minus, nom_next)

            F_numerical[:, i] = (dx_next_plus - dx_next_minus) / (2.0 * epsilon)

        # Compare F analytical vs numerical
        diff = np.abs(F_analytical - F_numerical)
        max_diff = np.max(diff)
        assert max_diff < 5e-5, f"Jacobian F mismatch: max diff = {max_diff}"
