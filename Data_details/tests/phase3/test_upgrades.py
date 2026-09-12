"""
Unit tests for Architectural Upgrades across Phases 1, 2, and 3.

Tests:
1. Streaming ZUPT detector on stationary and dynamic signals
2. Vehicle-body frame acceleration and angular velocity transformations
3. Multi-rate dataframe resampling (10 Hz -> 50 Hz/100 Hz)
4. Ground truth wheel odometry extraction
5. 12-channel enriched ML sliding window tensor generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import pytest

from Data_details.src.zupt_detector import ZUPTDetector, compute_streaming_zupt
from Data_details.src.phone_vehicle_alignment import compute_vehicle_frame_signals
from Data_details.src.dataset_loader import resample_dataframe, extract_ground_truth_odometry
from Data_details.src.phase3_pipeline import prepare_ml_ready_windows


class TestZUPTDetector:
    """Tests for real-time streaming Zero-Velocity Update (ZUPT) detector."""

    def test_stationary_detection(self):
        """Stationary sensor values must trigger ZUPT after persistence threshold."""
        detector = ZUPTDetector(window_size=10, min_duration_samples=3)
        acc_still = np.array([0.02, 0.03, 9.80])
        gyro_still = np.array([0.001, 0.001, -0.001])

        # First 2 samples should not trigger (need min_duration_samples = 3)
        stat1, dur1 = detector.update(acc_still, gyro_still, dt=0.1)
        stat2, dur2 = detector.update(acc_still, gyro_still, dt=0.1)
        assert not stat1
        assert not stat2

        # 3rd sample triggers ZUPT
        stat3, dur3 = detector.update(acc_still, gyro_still, dt=0.1)
        assert stat3 is True
        assert dur3 > 0.0

        # Subsequent stationary samples increase duration
        stat4, dur4 = detector.update(acc_still, gyro_still, dt=0.1)
        assert stat4 is True
        assert dur4 > dur3

    def test_dynamic_motion_rejection(self):
        """Dynamic maneuvers (high gyro or large accel changes) must reject ZUPT."""
        detector = ZUPTDetector(window_size=10, min_duration_samples=3)
        acc_move = np.array([1.50, 0.80, 9.20])
        gyro_move = np.array([0.25, 0.10, 0.35])

        for _ in range(10):
            stat, dur = detector.update(acc_move, gyro_move, dt=0.1)
            assert stat is False
            assert dur == 0.0

    def test_dataframe_vectorized_zupt(self):
        """Vectorized compute_streaming_zupt correctly segments stopped periods."""
        n = 50
        df = pd.DataFrame({
            "acc_x": np.concatenate([np.zeros(25), np.random.normal(0, 1.5, 25)]),
            "acc_y": np.concatenate([np.zeros(25), np.random.normal(0, 1.5, 25)]),
            "acc_z": np.concatenate([np.full(25, 9.80665), np.random.normal(9.8, 1.5, 25)]),
            "gyro_x": np.concatenate([np.zeros(25), np.random.normal(0, 0.2, 25)]),
            "gyro_y": np.concatenate([np.zeros(25), np.random.normal(0, 0.2, 25)]),
            "gyro_z": np.concatenate([np.zeros(25), np.random.normal(0, 0.2, 25)]),
            "dt": np.full(n, 0.1),
        })

        is_stat, dur = compute_streaming_zupt(df, min_duration_samples=3)
        assert len(is_stat) == n
        # First half should have stationary samples
        assert np.any(is_stat[:25])
        # Second half should be mostly non-stationary
        assert not is_stat[35]


class TestVehicleFrameTransform:
    """Tests for phone-to-vehicle body frame signal conversion."""

    def test_vehicle_frame_projection(self):
        """Known rotation matrix must project forward acceleration into X_v."""
        # Assume phone is mounted upright: +Y_phone = +Z_veh (Up), -Z_phone = +X_veh (Forward), +X_phone = +Y_veh (Right)
        R = np.array([
            [0.0,  0.0, -1.0],  # X_veh = -Z_phone
            [1.0,  0.0,  0.0],  # Y_veh = +X_phone
            [0.0,  1.0,  0.0],  # Z_veh = +Y_phone
        ])

        df = pd.DataFrame({
            "acc_x": [0.10, 0.20],
            "acc_y": [9.80, 9.81],
            "acc_z": [-1.50, -2.00],  # Forward push in -Z
            "gyro_x": [0.01, 0.02],
            "gyro_y": [0.03, 0.04],
            "gyro_z": [0.05, 0.06],
        })

        df_veh = compute_vehicle_frame_signals(df, R)

        assert "acc_fwd_veh" in df_veh.columns
        assert "acc_lat_veh" in df_veh.columns
        assert "acc_up_veh" in df_veh.columns

        # -Z_phone should map to +X_veh
        assert np.allclose(df_veh["acc_fwd_veh"].values, [1.50, 2.00])
        # +X_phone should map to +Y_veh
        assert np.allclose(df_veh["acc_lat_veh"].values, [0.10, 0.20])
        # +Y_phone should map to +Z_veh
        assert np.allclose(df_veh["acc_up_veh"].values, [9.80, 9.81])


class TestMultiRateResampling:
    """Tests for multi-rate interpolation engine."""

    def test_10hz_to_100hz_resampling(self):
        """Resampling from 10 Hz to 100 Hz must preserve bounds and interpolate smoothly."""
        t = np.arange(0, 1.0, 0.1)  # 10 samples
        df = pd.DataFrame({
            "time_s": t,
            "acc_x": 2.0 * t,
            "acc_y": np.sin(2 * np.pi * t),
        })

        df_100hz = resample_dataframe(df, target_freq_hz=100.0, method="linear")

        # Duration must be preserved
        assert np.isclose(df_100hz["time_s"].iloc[0], 0.0)
        assert np.isclose(df_100hz["time_s"].iloc[-1], 0.9)
        # 10 Hz (10 samples) -> 100 Hz (~91 samples)
        assert len(df_100hz) >= 90
        # Linear slope must be preserved: acc_x = 2 * time_s
        assert np.allclose(df_100hz["acc_x"].values, 2.0 * df_100hz["time_s"].values, atol=1e-3)


class TestGroundTruthOdometry:
    """Tests for wheel speed ground truth odometry extraction."""

    def test_odometry_extraction(self):
        """Wheel speeds must be scaled to m/s and integrated into cumulative distance."""
        n = 20
        df_v = pd.DataFrame({
            "time_s": np.arange(n) * 0.1,
            "dt": np.full(n, 0.1),
            "v_wheel_fl_rads": np.full(n, 33.33),  # 33.33 rad/s * 0.301 m ~ 10.03 m/s
            "v_wheel_fr_rads": np.full(n, 33.33),
            "v_wheel_rl_rads": np.full(n, 33.33),
            "v_wheel_rr_rads": np.full(n, 33.33),
            "v_speed_mps": np.full(n, 10.0),
        })

        odo = extract_ground_truth_odometry(df_v, wheel_radius_m=0.301)

        assert "v_true_mps" in odo.columns
        assert "dist_true_m" in odo.columns
        assert np.allclose(odo["v_true_mps"].values, 10.0, atol=0.1)
        # In 2.0 seconds at 10 m/s, distance is ~20 m
        assert np.isclose(odo["dist_true_m"].iloc[-1], 20.0, atol=0.5)


class Test12ChannelMLWindowing:
    """Tests for 12-channel enriched ML sliding window tensor generation."""

    def test_12_channel_window_tensor_generation(self, tmp_path):
        """prepare_ml_ready_windows must generate (N, 30, 12) tensor and NPZ archive."""
        n_samples = 100
        df = pd.DataFrame({
            "time_s": np.arange(n_samples) * 0.1,
            "dt": np.full(n_samples, 0.1),
            "acc_fwd_veh": np.random.normal(0.5, 0.2, n_samples),
            "acc_lat_veh": np.random.normal(0.0, 0.1, n_samples),
            "acc_up_veh": np.random.normal(9.8, 0.1, n_samples),
            "gyro_roll_veh": np.random.normal(0.0, 0.01, n_samples),
            "gyro_pitch_veh": np.random.normal(0.0, 0.01, n_samples),
            "gyro_yaw_veh": np.random.normal(0.02, 0.05, n_samples),
            "jerk_fwd": np.random.normal(0.0, 0.1, n_samples),
            "acc_horiz_norm": np.full(n_samples, 0.5),
            "gyro_norm": np.full(n_samples, 0.05),
            "ori_pitch_deg": np.full(n_samples, 2.5),
            "vibration_energy": np.full(n_samples, 0.01),
            "is_stationary": np.zeros(n_samples),
            "gps_speed_mps": np.full(n_samples, 12.5),
            "pos_east": np.cumsum(np.full(n_samples, 1.25)),
            "pos_north": np.cumsum(np.full(n_samples, 0.0)),
        })

        meta_file = tmp_path / "test_ml_meta.json"
        npz_file = tmp_path / "test_ml_tensor.npz"

        summary = prepare_ml_ready_windows(
            df,
            window_length=30,
            stride=5,
            output_meta_path=str(meta_file),
            output_npz_path=str(npz_file),
        )

        assert meta_file.exists()
        assert npz_file.exists()

        assert summary["feature_count"] == 12
        assert summary["window_length_samples"] == 30
        assert summary["total_extracted_windows"] == (100 - 30) // 5 + 1  # 15 windows

        # Check NPZ archive contents
        data = np.load(npz_file)
        assert "X" in data
        assert "y_speed" in data
        assert "y_disp" in data
        assert "feature_names" in data

        X = data["X"]
        assert X.shape == (15, 30, 12)
        assert len(data["y_speed"]) == 15
        assert data["y_disp"].shape == (15, 2)
