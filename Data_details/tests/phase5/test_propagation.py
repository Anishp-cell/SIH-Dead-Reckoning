"""
Phase 5 Unit Tests: Nominal INS Kinematic Propagation.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.propagation import propagate_nominal_ins
from Data_details.src.phase5.core.frames import GRAVITY_MAGNITUDE


class TestPhase5Propagation:

    def test_stationary_sensor_remains_at_rest(self):
        # Stationary on flat ground: f_meas = [0, 0, +g], omega = 0
        state = NominalState(
            p=np.zeros(3),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        f_meas = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        omega_meas = np.zeros(3)
        dt = 0.1

        # Propagate for 10 seconds (100 steps)
        for _ in range(100):
            state, _, _ = propagate_nominal_ins(state, f_meas, omega_meas, dt)

        # Position and velocity should remain exactly zero
        assert np.allclose(state.p, np.zeros(3), atol=1e-10)
        assert np.allclose(state.v, np.zeros(3), atol=1e-10)
        assert np.allclose(state.q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_constant_forward_velocity(self):
        # Moving East at constant 10 m/s: a_dyn = 0 -> f_meas = [0, 0, +g]
        state = NominalState(
            p=np.zeros(3),
            v=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        f_meas = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        omega_meas = np.zeros(3)
        dt = 0.1

        # Propagate 5 seconds (50 steps)
        for _ in range(50):
            state, _, _ = propagate_nominal_ins(state, f_meas, omega_meas, dt)

        # Expected position: p = v * t = [50.0, 0.0, 0.0]
        assert np.allclose(state.p, [50.0, 0.0, 0.0], atol=1e-7)
        assert np.allclose(state.v, [10.0, 0.0, 0.0], atol=1e-7)

    def test_constant_acceleration_analytical_match(self):
        # Accelerating East at 2.0 m/s^2 from rest
        # Level attitude (q = identity) means body X is East
        # Accelerometer measures dynamic fwd accel + upward gravity: [2.0, 0.0, +g]
        state = NominalState(
            p=np.zeros(3),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        f_meas = np.array([2.0, 0.0, GRAVITY_MAGNITUDE])
        omega_meas = np.zeros(3)
        dt = 0.05
        total_time = 4.0
        n_steps = int(total_time / dt)

        for _ in range(n_steps):
            state, _, _ = propagate_nominal_ins(state, f_meas, omega_meas, dt)

        # Analytical: v = a * t = 2.0 * 4.0 = 8.0 m/s
        # Analytical: p = 0.5 * a * t^2 = 0.5 * 2.0 * 16.0 = 16.0 m
        assert np.isclose(state.v[0], 8.0, atol=1e-7)
        assert np.isclose(state.p[0], 16.0, atol=1e-7)
        assert np.isclose(state.v[1], 0.0, atol=1e-7)
        assert np.isclose(state.p[1], 0.0, atol=1e-7)

    def test_uncompensated_accelerometer_bias_drift(self):
        # Uncompensated forward accelerometer bias ba = 0.1 m/s^2
        # Measured is gravity only, but true bias exists
        state_ideal = NominalState(p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
        state_biased = NominalState(p=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))

        # We simulate that the sensor has a +0.1 m/s^2 bias
        f_meas_ideal = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        f_meas_biased = np.array([0.1, 0.0, GRAVITY_MAGNITUDE])
        dt = 0.1

        for _ in range(100):  # 10 seconds
            state_ideal, _, _ = propagate_nominal_ins(state_ideal, f_meas_ideal, np.zeros(3), dt)
            state_biased, _, _ = propagate_nominal_ins(state_biased, f_meas_biased, np.zeros(3), dt)

        # Expected error: delta_p = 0.5 * ba * t^2 = 0.5 * 0.1 * 100 = 5.0 m
        delta_p = state_biased.p[0] - state_ideal.p[0]
        assert np.isclose(delta_p, 5.0, atol=1e-5)
