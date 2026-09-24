"""
Unit and integration tests for Phase 8 Streaming Engine.
"""

import numpy as np
import pytest

from Data_details.src.phase8.streaming import Phase8StreamingEngine
from Data_details.src.phase8.core.phase8_eskf import Phase8ESKF


def test_phase8_streaming_lifecycle():
    """
    Verifies that Phase8StreamingEngine steps through normal navigation,
    transitions to outage on signal loss, and smoothly recovers when signal returns.
    """
    ref_lat, ref_lon, ref_alt = 12.9716, 77.5946, 920.0

    engine = Phase8StreamingEngine(
        ref_lat_deg=ref_lat,
        ref_lon_deg=ref_lon,
        ref_alt_m=ref_alt,
        enable_map_updates=False,  # Test core fusion and outage logic
    )

    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([10.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    engine.initialize(p0, v0, q0, initial_timestamp=0.0)

    # Synthetic IMU sample: 10 m/s^2 along z (gravity), 0 gyro, 10 Hz
    # Format: [ax, ay, az, gx, gy, gz, roll, pitch, yaw, mag, vibe, temp]
    imu_sample = [0.0, 0.0, -9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 25.0]

    # 1. 5 normal steps with valid GNSS
    for step_i in range(1, 6):
        t = step_i * 0.1
        # In ENU, moving East at 10 m/s -> dx = 10 * t
        d_east = 10.0 * t
        lat = ref_lat
        lon = ref_lon + (d_east / (111132.954 * np.cos(np.radians(ref_lat))))

        est = engine.step(
            imu_sample_12ch=imu_sample,
            timestamp=t,
            gps_lat=lat,
            gps_lon=lon,
            gps_alt=ref_alt,
            gps_acc_m=2.0,
            gps_sats=8,
            gps_speed_mps=10.0,
            gps_heading_deg=90.0,  # East
        )
        assert est.gnss_state in ["GNSS_HEALTHY", "GNSS_SUSPECT"]
        assert est.filter_healthy
        assert np.all(np.isfinite(est.p_fused))

    # 2. 5 steps of synthetic outage
    for step_i in range(6, 11):
        t = step_i * 0.1
        est = engine.step(
            imu_sample_12ch=imu_sample,
            timestamp=t,
            is_synthetic_outage=True,
        )
        assert est.filter_healthy
        assert not est.gnss_pos_accepted

    # Outage should be declared
    assert engine.state_machine.current_state.value == "GNSS_OUTAGE"

    # 3. GNSS returns: 5 recovery steps
    for step_i in range(11, 16):
        t = step_i * 0.1
        d_east = 10.0 * t
        lat = ref_lat
        lon = ref_lon + (d_east / (111132.954 * np.cos(np.radians(ref_lat))))

        est = engine.step(
            imu_sample_12ch=imu_sample,
            timestamp=t,
            gps_lat=lat,
            gps_lon=lon,
            gps_alt=ref_alt,
            gps_acc_m=2.0,
            gps_sats=8,
            gps_speed_mps=10.0,
            gps_heading_deg=90.0,
        )
        assert est.filter_healthy
        # Step clamping prevents jumps
        assert est.gnss_p_step_m <= 3.5
