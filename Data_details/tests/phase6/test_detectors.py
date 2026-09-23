"""
Phase 6 Unit Tests: Robust Stationary Detector & Disturbance Detector.
"""

import pytest
import numpy as np

from Data_details.src.phase6.detection.stationary_detector import RobustStationaryDetector
from Data_details.src.phase6.detection.disturbance_detector import DisturbanceDetector


class TestPhase6Detectors:

    def test_stationary_detector_hysteresis_and_dwell(self):
        detector = RobustStationaryDetector(min_enter_samples=5, min_exit_samples=2)

        # Feed 4 stationary samples: should NOT declare stationary yet (dwell = 5)
        f_stat = np.array([0.01, -0.02, 9.80665])
        w_stat = np.array([0.001, -0.002, 0.001])

        for _ in range(4):
            is_stat, dur, _ = detector.update(f_stat, w_stat, ai_speed_mps=0.0)
            assert is_stat is False

        # 5th sample: satisfies dwell requirement -> declares stationary!
        is_stat, dur, _ = detector.update(f_stat, w_stat, ai_speed_mps=0.0)
        assert is_stat is True
        assert dur > 0.0

        # Feed 1 moving sample (e.g. speed = 5.0 m/s): should NOT exit immediately (exit dwell = 2)
        is_stat, dur, _ = detector.update(f_stat, w_stat, ai_speed_mps=5.0)
        assert is_stat is True

        # 2nd moving sample: exits stationary state!
        is_stat, dur, _ = detector.update(f_stat, w_stat, ai_speed_mps=5.0)
        assert is_stat is False
        assert dur == 0.0

    def test_disturbance_detector_cornering_and_shock(self):
        detector = DisturbanceDetector(
            lat_accel_onset=1.5,
            yaw_rate_onset=0.15,
            vert_accel_onset=2.0,
        )

        # Case 1: Quiet straight cruising
        f_quiet = np.array([0.2, 0.05, 9.80665])
        w_quiet = np.array([0.01, 0.01, 0.01])
        rep1 = detector.evaluate(f_quiet, w_quiet)
        assert rep1.is_cornering is False
        assert rep1.is_shock is False
        assert rep1.c_nhc > 0.9

        # Case 2: Sharp cornering (lat accel = 3.0 m/s^2, yaw rate = 0.3 rad/s)
        f_turn = np.array([0.5, 3.0, 9.80665])
        w_turn = np.array([0.02, 0.01, 0.30])
        rep2 = detector.evaluate(f_turn, w_turn)
        assert rep2.is_cornering is True
        assert rep2.d_lat > 0.0
        assert rep2.c_nhc < rep1.c_nhc  # Confidence down-weighted

        # Case 3: Road shock / pothole (vertical force = 14.5 m/s^2)
        f_bump = np.array([0.2, 0.05, 14.5])
        w_bump = np.array([0.01, 0.01, 0.01])
        rep3 = detector.evaluate(f_bump, w_bump, vibration_energy=2.5)
        assert rep3.is_shock is True
        assert rep3.d_up > 0.0
        assert rep3.c_nhc < rep1.c_nhc
