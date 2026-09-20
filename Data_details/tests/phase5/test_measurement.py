"""
Phase 5 Unit Tests: Speed Measurement Model, NIS Gating, & Joseph Update.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.measurement import (
    predict_forward_speed,
    compute_speed_jacobian,
    kalman_update_speed,
)


class TestPhase5Measurement:

    def test_predict_forward_speed_level_vehicle(self):
        # Level vehicle moving East (yaw = 0 ENU) at 15 m/s
        state = NominalState(
            p=np.zeros(3),
            v=np.array([15.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        v_fwd, v_b = predict_forward_speed(state)
        assert np.isclose(v_fwd, 15.0, atol=1e-7)
        assert np.allclose(v_b, [15.0, 0.0, 0.0], atol=1e-7)

    def test_kalman_update_reduces_uncertainty(self):
        state = NominalState(
            p=np.zeros(3),
            v=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        P_prior = np.eye(15, dtype=np.float64)

        z_speed = 10.5
        sigma_v = 0.5

        state_plus, P_plus, nu, S, nis, accepted = kalman_update_speed(
            state, P_prior, z_speed, sigma_v, nis_threshold=9.0
        )

        assert accepted is True
        # Joseph form symmetry check
        assert np.allclose(P_plus, P_plus.T, atol=1e-12)

        # Forward velocity uncertainty must decrease
        # Velocity block is indices 3, 4, 5. Forward is X (index 3).
        assert P_plus[3, 3] < P_prior[3, 3]

    def test_nis_gating_rejects_severe_outliers(self):
        state = NominalState(
            p=np.zeros(3),
            v=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        P_prior = np.eye(15, dtype=np.float64) * 0.1  # Highly confident prior

        # Wild erroneous spike: 50 m/s when true is 10 m/s (40 m/s error)
        z_speed_outlier = 50.0
        sigma_v = 0.5  # Model claims 0.5 m/s uncertainty

        state_plus, P_plus, nu, S, nis, accepted = kalman_update_speed(
            state, P_prior, z_speed_outlier, sigma_v, nis_threshold=9.0
        )

        # Should be rejected!
        assert accepted is False
        assert nis > 9.0
        # State and covariance must remain identical
        assert np.allclose(state_plus.v, state.v, atol=1e-12)
        assert np.allclose(P_plus, P_prior, atol=1e-12)
