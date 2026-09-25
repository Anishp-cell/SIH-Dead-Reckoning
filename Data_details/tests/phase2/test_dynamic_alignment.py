"""
Unit Tests for Dynamic In-Vehicle Alignment & Mount Slip Recalibration.
"""

import pytest
import numpy as np

from Data_details.src.phase2.dynamic_alignment import (
    MountSlipState,
    DynamicMountSlipDetector,
    DynamicAlignmentConfig,
)


class TestDynamicAlignment:
    def test_stable_driving_no_false_alarm(self):
        detector = DynamicMountSlipDetector()
        detector.reset(np.eye(3))

        # Gravity pointing down in phone Z: [0, 0, -9.81]
        # With normal vehicle road vibrations (+-0.2 m/s^2) and gentle turns (0.1 rad/s)
        rng = np.random.RandomState(42)
        for i in range(100):
            t = i * 0.1
            accel = np.array([0.0, 0.0, -9.81]) + rng.normal(0, 0.1, 3)
            gyro = np.array([0.0, 0.0, 0.05]) + rng.normal(0, 0.02, 3)

            state, R_cur, inflation = detector.step(t, accel, gyro, dt=0.1)
            assert state == MountSlipState.STABLE
            assert inflation == 1.0

    def test_mount_slip_detection_and_recalibration(self):
        cfg = DynamicAlignmentConfig(
            slip_angle_threshold_deg=5.0,
            gyro_spike_threshold_rads=1.0,
            stabilization_duration_s=0.5,
            covariance_inflation_factor=12.0,
        )
        detector = DynamicMountSlipDetector(config=cfg)
        detector.reset(np.eye(3))

        # Phase 1: 30 steps of stable baseline
        for i in range(30):
            t = i * 0.1
            detector.step(t, np.array([0.0, 0.0, -9.81]), np.zeros(3), dt=0.1)

        # Phase 2: Violent mount shift at t = 3.0s (phone tilts by 20 deg pitch with gyro spike)
        theta_slip = np.radians(20.0)
        # Rotated gravity vector: [ -9.81 * sin(20), 0, -9.81 * cos(20) ]
        g_slipped = np.array([-9.81 * np.sin(theta_slip), 0.0, -9.81 * np.cos(theta_slip)])
        w_spike = np.array([0.0, 2.5, 0.0])  # 2.5 rad/s pitch spike

        state_slip, _, inf_slip = detector.step(3.0, g_slipped, w_spike, dt=0.1)
        assert state_slip == MountSlipState.SLIP_DETECTED
        assert inf_slip == 12.0

        # Phase 3: Settling in new mount orientation over 8 steps (0.8s)
        settled_states = []
        for j in range(8):
            t = 3.1 + j * 0.1
            st, R_new, inf = detector.step(t, g_slipped, np.zeros(3), dt=0.1)
            settled_states.append(st)

        # Should have transitioned through RECALIBRATING and returned to STABLE
        assert detector.state == MountSlipState.STABLE
        assert len(detector.slip_events) == 1

        event = detector.slip_events[0]
        assert pytest.approx(event.slip_angle_deg, abs=2.0) == 20.0
        assert event.gyro_peak_rads >= 2.5

        # Verify SO(3) orthonormality of new R_phone_to_veh
        R_after = detector.R_phone_to_veh
        assert np.allclose(R_after @ R_after.T, np.eye(3), atol=1e-5)
        assert pytest.approx(np.linalg.det(R_after), abs=1e-4) == 1.0

        # Verify that in vehicle frame, new gravity points strictly DOWN: [0, 0, -9.81]
        g_veh_after = R_after @ g_slipped
        assert abs(g_veh_after[0]) < 0.2  # fwd component ~ 0
        assert abs(g_veh_after[1]) < 0.2  # lat component ~ 0
        assert pytest.approx(g_veh_after[2], abs=0.2) == 9.81 or pytest.approx(g_veh_after[2], abs=0.2) == -9.81
