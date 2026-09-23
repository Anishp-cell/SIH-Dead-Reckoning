"""
Phase 6 Synthetic Scenario Validation: Deterministic Benchmark Tests.
"""

import pytest
import numpy as np

from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase6.streaming import Phase6StreamingEngine


class TestPhase6SyntheticScenarios:

    def test_scenario_a_stationary_vehicle_zero_creep(self):
        """
        Test A: Stationary vehicle sitting at rest for 10 seconds (100 steps).
        Position must remain zero and velocity must remain clamped to zero.
        """
        eskf = Phase6ESKF(enable_nhc=True, enable_zupt=True, enable_zaru=True)
        engine = Phase6StreamingEngine(eskf=eskf)

        p0 = np.zeros(3)
        v0 = np.zeros(3)
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        engine.initialize(p0, v0, q0, initial_timestamp=0.0)

        # 12-channel stationary sample
        sample = np.zeros(12)
        sample[2] = -9.80665  # acc_up_veh in Phase 3 convention (-g)

        for i in range(100):
            t = (i + 1) * 0.1
            est = engine.step(sample, timestamp=t, enable_ai_speed=False)

        # Position should remain strictly zero (< 1e-4 m)
        assert np.linalg.norm(est.p) < 1e-4
        assert np.linalg.norm(est.v) < 1e-4
        assert est.is_stationary is True
        assert est.zupt_accepted is True

    def test_scenario_b_constant_forward_motion_nhc_preservation(self):
        """
        Test B: Vehicle moving East at constant 10 m/s for 5 seconds.
        NHC should maintain lateral and vertical velocity near zero while preserving forward velocity.
        """
        eskf = Phase6ESKF(enable_nhc=True, enable_zupt=False, enable_zaru=False)
        engine = Phase6StreamingEngine(eskf=eskf)

        p0 = np.zeros(3)
        v0 = np.array([10.0, 0.0, 0.0])  # Moving East
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # Heading East
        engine.initialize(p0, v0, q0, initial_timestamp=0.0)

        sample = np.zeros(12)
        sample[2] = -9.80665

        for i in range(50):
            t = (i + 1) * 0.1
            est = engine.step(sample, timestamp=t, enable_ai_speed=False)

        # Forward position should grow linearly: p_x = 10 * 5 = 50 m
        assert np.isclose(est.p[0], 50.0, atol=1e-3)
        # Lateral (North) and vertical (Up) position should stay 0
        assert abs(est.p[1]) < 1e-4
        assert abs(est.p[2]) < 1e-4
        # Body velocities
        assert np.isclose(est.v_b[0], 10.0, atol=1e-3)
        assert abs(est.v_b[1]) < 1e-4
        assert abs(est.v_b[2]) < 1e-4

    def test_scenario_d_injected_lateral_disturbance_correction(self):
        """
        Test D: Inject synthetic lateral velocity disturbance into state.
        NHC should immediately detect non-zero lateral velocity and correct it.
        """
        eskf = Phase6ESKF(enable_nhc=True, enable_zupt=False, enable_zaru=False)
        engine = Phase6StreamingEngine(eskf=eskf)

        p0 = np.zeros(3)
        # Corrupted initial velocity with large lateral slip (3.0 m/s)
        v0 = np.array([10.0, 3.0, 0.0])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        engine.initialize(p0, v0, q0, initial_timestamp=0.0)

        sample = np.zeros(12)
        sample[2] = -9.80665

        # After a few steps of NHC updates, lateral velocity should be corrected toward zero
        for i in range(5):
            t = (i + 1) * 0.1
            est = engine.step(sample, timestamp=t, enable_ai_speed=False)

        # Lateral velocity should be substantially reduced
        assert abs(est.v_b[1]) < 1.0  # Reduced from 3.0 m/s
        assert est.nhc_accepted is True

    def test_scenario_e_injected_gyro_bias_suppression_via_zaru(self):
        """
        Test E: Stationary vehicle with uncompensated +0.01 rad/s gyro bias.
        ZARU should estimate the gyro bias and suppress heading rotation.
        """
        eskf = Phase6ESKF(enable_nhc=True, enable_zupt=True, enable_zaru=True)
        engine = Phase6StreamingEngine(eskf=eskf)

        p0 = np.zeros(3)
        v0 = np.zeros(3)
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        engine.initialize(p0, v0, q0, initial_timestamp=0.0)

        # IMU sample has a constant yaw gyro bias: +0.01 rad/s
        sample = np.zeros(12)
        sample[2] = -9.80665
        sample[5] = 0.010  # gyro_yaw_veh

        for i in range(100):  # 10 seconds
            t = (i + 1) * 0.1
            est = engine.step(sample, timestamp=t, enable_ai_speed=False)

        # Gyro bias bg should be estimated close to 0.010 rad/s (converging towards 0.010)
        assert np.isclose(est.bg[2], 0.010, atol=2.5e-3)
        assert est.zaru_accepted is True
