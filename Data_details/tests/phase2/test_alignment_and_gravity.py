
"""
Phase 2 Unit Tests: Phone-to-Vehicle Alignment and Gravity Compensation.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from Data_details.src.attitude_estimation import euler_to_quaternion
from Data_details.src.phone_vehicle_alignment import (
    estimate_phone_to_vehicle_rotation,
    transform_vector_to_vehicle_frame,
    detect_orientation_changes,
)
from Data_details.src.gravity_compensation import (
    compensate_gravity_world_frame,
    evaluate_gravity_compensation,
)


class TestPhoneVehicleAlignment:
    """Tests 3D rotation matrix estimation and frame transformations."""

    def test_alignment_matrix_orthonormality(self):
        """Estimated rotation matrix must be in SO(3) with det(R) = +1."""
        n = 100
        # Phone mounted with top pointing forward (Y), screen facing up (Z)
        df = pd.DataFrame({
            "time_s": np.arange(n) * 0.1,
            "acc_x": np.zeros(n),
            "acc_y": np.concatenate([np.zeros(50), np.full(50, 1.5)]), # acceleration forward
            "acc_z": np.full(n, 9.80665),
            "gyro_x": np.zeros(n),
            "gyro_y": np.zeros(n),
            "gyro_z": np.zeros(n),
            "gps_speed_mps": np.concatenate([np.zeros(50), np.full(50, 10.0)]),
        })
        stat_mask = np.concatenate([np.ones(50, dtype=bool), np.zeros(50, dtype=bool)])
        
        R_p2v, meta = estimate_phone_to_vehicle_rotation(df, stat_mask, min_speed_mps=2.0)
        
        assert meta["is_orthonormal"] is True
        assert np.isclose(np.linalg.det(R_p2v), 1.0, atol=1e-6)

    def test_vector_magnitude_preserved_under_rotation(self):
        """Rotating vectors by R_phone_to_vehicle must strictly preserve Euclidean norm."""
        vx = np.array([1.0, 2.0, -3.0])
        vy = np.array([4.0, -1.0, 2.0])
        vz = np.array([0.5, 3.5, -1.5])
        orig_norm = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Random valid rotation matrix in SO(3)
        q = euler_to_quaternion(np.radians(35.0), np.radians(-20.0), np.radians(70.0))
        from Data_details.src.attitude_estimation import quaternion_to_rotation_matrix
        R = quaternion_to_rotation_matrix(q)
        
        fx, fy, fz = transform_vector_to_vehicle_frame(vx, vy, vz, R)
        rot_norm = np.sqrt(fx**2 + fy**2 + fz**2)
        
        np.testing.assert_allclose(rot_norm, orig_norm, atol=1e-10)

    def test_orientation_change_detector(self):
        """Detects sudden tilt shift in synthetic gravity sequence."""
        n = 100
        # First 50 samples: standard upright tilt
        # Next 50 samples: phone flipped by 45 degrees
        gx = np.concatenate([np.zeros(50), np.full(50, 6.0)])
        gy = np.zeros(n)
        gz = np.concatenate([np.full(50, 9.80665), np.full(50, 7.0)])
        
        df = pd.DataFrame({"time_s": np.arange(n) * 0.1, "grav_x": gx, "grav_y": gy, "grav_z": gz})
        initial_g = np.array([0.0, 0.0, 9.80665])
        
        mask, events = detect_orientation_changes(df, initial_g, cos_tolerance=0.95, window_samples=5)
        assert events[0]["detected"] is True


class TestGravityCompensation:
    """Tests orientation-aware gravity compensation."""

    def test_gravity_compensation_on_tilted_stationary_phone(self):
        """
        When a phone is tilted by 30° roll, raw accelerometer reads [0, g*sin(30), g*cos(30)].
        After attitude-aware world-frame gravity compensation, dynamic acceleration must be [0, 0, 0].
        """
        g = 9.80665
        roll_deg = 30.0
        roll_rad = np.radians(roll_deg)
        
        # Phone body acceleration due to gravity
        ax = 0.0
        ay = g * np.sin(roll_rad)
        az = g * np.cos(roll_rad)
        acc_body = np.array([[ax, ay, az]])
        
        # Phone attitude quaternion (roll=30 deg, pitch=0, yaw=0)
        q = euler_to_quaternion(roll_rad, 0.0, 0.0)
        quats = np.array([q])
        
        a_dyn_nav, a_total_nav = compensate_gravity_world_frame(acc_body, quats, gravity_norm=g)
        
        # Total in world frame should be [0, 0, g]
        np.testing.assert_allclose(a_total_nav[0], [0.0, 0.0, g], atol=1e-5)
        # Dynamic acceleration should be [0, 0, 0] (no movement)
        np.testing.assert_allclose(a_dyn_nav[0], [0.0, 0.0, 0.0], atol=1e-5)
