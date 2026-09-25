"""
Unit Tests for Indian Road Dynamics: Two-Wheeler Lean-Compensated NHC and Pothole Impulse Gating.
"""

import pytest
import numpy as np

from Data_details.src.phase6.constraints.nhc import (
    predict_nhc_velocity,
    compute_nhc_jacobian,
    apply_nhc_update,
    compute_two_wheeler_steady_lean_angle,
)
from Data_details.src.phase6.detection.disturbance_detector import (
    DisturbanceDetector,
    DisturbanceReport,
)
from Data_details.src.phase5.core.eskf import NominalState


class TestIndianRoadDynamics:
    def test_two_wheeler_steady_lean_angle(self):
        # 15 m/s (~54 km/h), 0.2 rad/s turn
        # tan(phi) = (15 * 0.2) / 9.80665 = 3.0 / 9.80665 = 0.3059
        # phi = 17.0 deg (0.2968 rad)
        phi_rad = compute_two_wheeler_steady_lean_angle(
            v_forward_mps=15.0,
            yaw_rate_rads=0.20,
        )
        assert pytest.approx(np.degrees(phi_rad), abs=0.5) == 17.0

        # Stationary or very slow: lean angle must be 0
        assert compute_two_wheeler_steady_lean_angle(0.1, 0.2) == 0.0

    def test_lean_compensated_nhc_projection(self):
        state = NominalState(
            p=np.zeros(3),
            v=np.array([12.0, 0.0, 0.0]),  # 12 m/s forward East
            q=np.array([1.0, 0.0, 0.0, 0.0]), # identity attitude
            ba=np.zeros(3),
            bg=np.zeros(3),
        )

        phi_lean = np.radians(25.0)  # 25 degree banked motorcycle turn

        # In vehicle body frame, bike has rolled by phi
        # Simulate lateral roll where contact patch is aligned:
        # v_lat = v_contact * cos(phi), v_up = -v_contact * sin(phi)
        h_nhc_banked, _ = predict_nhc_velocity(state, lean_angle_rad=phi_lean)
        assert len(h_nhc_banked) == 2
        assert abs(h_nhc_banked[0]) < 1e-4

        # Compare unbanked vs banked Jacobians
        H_unbanked = compute_nhc_jacobian(state, lean_angle_rad=0.0)
        H_banked = compute_nhc_jacobian(state, lean_angle_rad=phi_lean)

        assert H_unbanked.shape == (2, 15)
        assert H_banked.shape == (2, 15)
        # Banked Jacobian must differ from unbanked when phi > 0
        assert not np.allclose(H_unbanked, H_banked)

    def test_lean_nhc_kalman_update(self):
        state = NominalState(
            p=np.zeros(3),
            v=np.array([10.0, 0.5, -0.2]),  # slight transverse slip in body
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            ba=np.zeros(3),
            bg=np.zeros(3),
        )
        P = np.eye(15, dtype=np.float64) * 0.5
        R_nhc = np.diag([0.1**2, 0.15**2])

        phi_lean = np.radians(15.0)
        result = apply_nhc_update(state, P, R_nhc, lean_angle_rad=phi_lean)
        assert result.accepted is True
        # Transverse uncertainty should be reduced
        assert result.P_plus[4, 4] < P[4, 4]

    def test_pothole_shock_gating_and_covariance_inflation(self):
        detector = DisturbanceDetector(
            gravity_ref=9.80665,
            pothole_shock_threshold=10.0,  # 10 m/s^2 deviation
        )

        # 1. Normal smooth driving (no pothole)
        f_smooth = np.array([0.5, 0.0, 9.81])
        omega_smooth = np.zeros(3)
        rep_smooth = detector.evaluate(f_smooth, omega_smooth)
        assert rep_smooth.is_shock is False
        assert rep_smooth.is_pothole_shock is False
        assert rep_smooth.r_up_scale_factor == 1.0

        # 2. Violent Indian pothole impact (sharp vertical acceleration spike)
        # a_up jumps to 26 m/s^2 (~2.65g)
        f_pothole = np.array([0.2, 0.1, 26.0])
        rep_pothole = detector.evaluate(f_pothole, omega_smooth)

        assert rep_pothole.is_shock is True
        assert rep_pothole.is_pothole_shock is True
        assert rep_pothole.shock_magnitude_mps2 > 15.0
        # R_up must be strongly inflated (> 50x) to protect filter from shock corrupting attitude
        assert rep_pothole.r_up_scale_factor > 50.0
