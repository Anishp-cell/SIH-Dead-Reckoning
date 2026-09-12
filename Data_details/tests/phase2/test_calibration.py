"""
Phase 2 Unit Tests: Multi-Sensor Stationary Detection and Bias Calibration.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from Data_details.src.sensor_calibration import (
    detect_stationary_periods_multi,
    estimate_gyroscope_bias,
    estimate_accelerometer_bias,
    audit_sensor_units_and_axes,
)


class TestStationaryDetectionAndCalibration:
    """Tests multi-sensor stationary detection and bias recovery on synthetic data."""

    def _make_synthetic_df_with_stop(self, n=300):
        """
        Creates synthetic DataFrame:
        - 0 to 100 samples (0 to 10s): Driving (speed=15 m/s, high gyro)
        - 100 to 200 samples (10 to 20s): Stationary (speed=0, gyro~0.005 bias, acc~g)
        - 200 to 300 samples (20 to 30s): Driving (speed=15 m/s)
        """
        t = np.arange(n) * 0.1
        dt = np.full(n, 0.1)
        
        # Accelerometer: during stop [100:200], pure gravity z = 9.80665
        ax = np.concatenate([np.random.randn(100) * 1.5, np.zeros(100), np.random.randn(100) * 1.5])
        ay = np.concatenate([np.random.randn(100) * 1.5, np.zeros(100), np.random.randn(100) * 1.5])
        az = np.concatenate([np.random.randn(100) * 1.5 + 9.8, np.full(100, 9.80665), np.random.randn(100) * 1.5 + 9.8])
        
        # Gyro: true bias [0.004, -0.003, 0.005] rad/s
        gx = np.concatenate([np.random.randn(100) * 0.2, np.full(100, 0.004), np.random.randn(100) * 0.2])
        gy = np.concatenate([np.random.randn(100) * 0.2, np.full(100, -0.003), np.random.randn(100) * 0.2])
        gz = np.concatenate([np.random.randn(100) * 0.2, np.full(100, 0.005), np.random.randn(100) * 0.2])
        
        # Magnetometer
        mx = np.full(n, 15.0)
        my = np.full(n, -20.0)
        mz = np.full(n, 40.0)
        
        # GPS Speed
        speed_mps = np.concatenate([np.full(100, 15.0), np.zeros(100), np.full(100, 15.0)])
        
        df = pd.DataFrame({
            "time_s": t, "dt": dt,
            "acc_x": ax, "acc_y": ay, "acc_z": az,
            "gyro_x": gx, "gyro_y": gy, "gyro_z": gz,
            "mag_x": mx, "mag_y": my, "mag_z": mz,
            "gps_speed_mps": speed_mps,
        })
        return df

    def test_multi_sensor_stationary_detection(self):
        """Correctly isolates the 10-second stationary window from 10s to 20s."""
        df = self._make_synthetic_df_with_stop()
        periods, mask = detect_stationary_periods_multi(df, speed_threshold_mps=0.5, min_duration_s=5.0)
        
        assert len(periods) == 1
        assert periods[0]["start_idx"] == 100
        assert periods[0]["end_idx"] == 199
        assert np.isclose(periods[0]["duration_s"], 9.9, atol=0.2)
        assert mask[100:200].all()
        assert not mask[:100].any()
        assert not mask[200:].any()

    def test_gyroscope_bias_recovery(self):
        """Recovers the known injected gyro zero-rate bias during the stop."""
        df = self._make_synthetic_df_with_stop()
        _, mask = detect_stationary_periods_multi(df)
        bias_info = estimate_gyroscope_bias(df, mask)
        
        assert np.isclose(bias_info["bx"], 0.004, atol=1e-4)
        assert np.isclose(bias_info["by"], -0.003, atol=1e-4)
        assert np.isclose(bias_info["bz"], 0.005, atol=1e-4)

    def test_sensor_unit_audit_checks(self):
        """Sensor unit audit flags m/s^2, rad/s, and uT appropriately."""
        df = self._make_synthetic_df_with_stop()
        _, mask = detect_stationary_periods_multi(df)
        audit = audit_sensor_units_and_axes(df, mask)
        
        assert audit["accel_is_mps2"] is True
        assert audit["gyro_is_rads"] is True
        assert audit["mag_is_uT"] is True
