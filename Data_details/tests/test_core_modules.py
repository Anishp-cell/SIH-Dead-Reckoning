"""
Tests for coordinate transform, blackout simulator, and inertial baseline.
Uses synthetic data independent of IO-VNBD.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestCoordinateTransform:
    """Tests for WGS84 -> ENU coordinate transformation."""

    def test_origin_is_zero(self):
        """The origin point should map to (0, 0, 0) in ENU."""
        from Data_details.src.coordinate_transform import geodetic_to_enu

        lat = np.array([52.4])
        lon = np.array([-1.5])
        alt = np.array([100.0])

        e, n, u, info = geodetic_to_enu(lat, lon, alt)
        assert abs(e[0]) < 0.01, f"East should be ~0, got {e[0]}"
        assert abs(n[0]) < 0.01, f"North should be ~0, got {n[0]}"
        assert abs(u[0]) < 0.01, f"Up should be ~0, got {u[0]}"

    def test_north_movement(self):
        """Moving 0.001 degrees north (~111m) should show up in the north component."""
        from Data_details.src.coordinate_transform import geodetic_to_enu

        lat = np.array([52.4, 52.401])
        lon = np.array([-1.5, -1.5])
        alt = np.array([100.0, 100.0])

        e, n, u, info = geodetic_to_enu(lat, lon, alt)
        assert n[1] > 100, f"North displacement should be >100m, got {n[1]:.1f}"
        assert n[1] < 120, f"North displacement should be <120m, got {n[1]:.1f}"
        assert abs(e[1]) < 1, f"East should be ~0, got {e[1]:.1f}"

    def test_east_movement(self):
        """Moving east should show up primarily in the east component."""
        from Data_details.src.coordinate_transform import geodetic_to_enu

        lat = np.array([52.4, 52.4])
        lon = np.array([-1.5, -1.499])
        alt = np.array([100.0, 100.0])

        e, n, u, info = geodetic_to_enu(lat, lon, alt)
        assert e[1] > 50, f"East displacement should be significant, got {e[1]:.1f}"
        assert abs(n[1]) < 1, f"North should be ~0, got {n[1]:.1f}"

    def test_haversine_distance(self):
        """Known geographic distance check."""
        from Data_details.src.coordinate_transform import haversine_distance

        # ~111 km between 0° and 1° latitude at equator
        dist = haversine_distance(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        assert 110000 < dist[0] < 112000, f"Expected ~111km, got {dist[0]/1000:.1f}km"

    def test_cumulative_distance(self):
        """Cumulative distance of a straight line should equal direct distance."""
        from Data_details.src.coordinate_transform import cumulative_distance_m

        lat = np.array([0.0, 0.001, 0.002])
        lon = np.array([0.0, 0.0, 0.0])

        cum_dist = cumulative_distance_m(lat, lon)
        assert cum_dist[0] == 0.0
        assert cum_dist[-1] > 200  # ~222m for 0.002 degrees


class TestBlackoutSimulator:
    """Tests for the GNSS blackout simulator."""

    def _make_df(self, n=200):
        """Creates a minimal standardized DataFrame for testing."""
        return pd.DataFrame({
            "time_s": np.arange(n) * 0.1,
            "dt": [0.1] * n,
            "gps_lat": 52.4 + np.arange(n) * 0.00001,
            "gps_lon": -1.5 + np.arange(n) * 0.00001,
            "gps_alt": [100.0] * n,
            "gps_speed_kmh": [20.0] * n,
            "gps_speed_mps": [20.0 / 3.6] * n,
            "gps_acc_m": [3.0] * n,
            "gps_heading_deg": [180.0] * n,
        })

    def test_blackout_mask_correct_count(self):
        """Mask should cover exactly the right number of samples."""
        from Data_details.src.blackout_simulator import create_blackout_mask

        time_s = np.arange(200) * 0.1  # 0 to 19.9s
        mask = create_blackout_mask(time_s, 5.0, 5.0)  # 5s to 10s
        # Expected: samples from t=5.0 to t<10.0 -> 50 samples
        assert mask.sum() == 50

    def test_blackout_masks_gps_columns(self):
        """After blackout, GPS columns should be NaN within the window."""
        from Data_details.src.blackout_simulator import simulate_blackout

        df = self._make_df()
        bo_df = simulate_blackout(df, 5.0, 5.0, "smartphone")

        # During blackout: GPS should be NaN
        blackout_rows = bo_df[(bo_df["time_s"] >= 5.0) & (bo_df["time_s"] < 10.0)]
        assert blackout_rows["gps_lat"].isna().all()
        assert blackout_rows["gps_lon"].isna().all()

        # Before blackout: GPS should NOT be NaN
        before_rows = bo_df[bo_df["time_s"] < 5.0]
        assert not before_rows["gps_lat"].isna().any()

        # After blackout: GPS should NOT be NaN
        after_rows = bo_df[bo_df["time_s"] >= 10.0]
        assert not after_rows["gps_lat"].isna().any()

    def test_blackout_preserves_non_gps(self):
        """Non-GPS columns should be unchanged by blackout."""
        from Data_details.src.blackout_simulator import simulate_blackout

        df = self._make_df()
        df["acc_x"] = np.random.randn(len(df))
        original_acc = df["acc_x"].copy()

        bo_df = simulate_blackout(df, 5.0, 5.0, "smartphone")
        np.testing.assert_array_equal(bo_df["acc_x"].values, original_acc.values)

    def test_gnss_available_flag(self):
        """gnss_available flag should be added and correctly set."""
        from Data_details.src.blackout_simulator import simulate_blackout

        df = self._make_df()
        bo_df = simulate_blackout(df, 5.0, 5.0, "smartphone")

        assert "gnss_available" in bo_df.columns
        blackout_flags = bo_df[(bo_df["time_s"] >= 5.0) & (bo_df["time_s"] < 10.0)]["gnss_available"]
        assert not blackout_flags.any()

        available_flags = bo_df[bo_df["time_s"] < 5.0]["gnss_available"]
        assert available_flags.all()


class TestInertialBaseline:
    """Tests for classical dead reckoning integration."""

    def test_zero_acceleration_no_drift(self):
        """With zero acceleration, position should remain at origin."""
        from Data_details.src.inertial_baseline import integrate_dead_reckoning

        n = 100
        acc_e = np.zeros(n)
        acc_n = np.zeros(n)
        dt = np.full(n, 0.1)

        vel_e, vel_n, pos_e, pos_n = integrate_dead_reckoning(acc_e, acc_n, dt)

        np.testing.assert_allclose(pos_e, 0.0, atol=1e-10)
        np.testing.assert_allclose(pos_n, 0.0, atol=1e-10)

    def test_constant_velocity_linear_motion(self):
        """With zero acceleration and initial velocity, should move linearly."""
        from Data_details.src.inertial_baseline import integrate_dead_reckoning

        n = 100
        acc_e = np.zeros(n)
        acc_n = np.zeros(n)
        dt = np.full(n, 0.1)

        vel_e, vel_n, pos_e, pos_n = integrate_dead_reckoning(
            acc_e, acc_n, dt,
            initial_vel_east=10.0, initial_vel_north=0.0,
        )

        # After 10s (100 samples at 0.1s), should be at ~100m east
        expected_pos = 10.0 * 10.0  # v * t
        assert abs(pos_e[-1] - expected_pos) < 2.0, f"Expected ~{expected_pos}m, got {pos_e[-1]:.1f}m"
        assert abs(pos_n[-1]) < 1e-10

    def test_constant_acceleration(self):
        """With constant acceleration, position should grow quadratically."""
        from Data_details.src.inertial_baseline import integrate_dead_reckoning

        n = 100
        acc_e = np.full(n, 1.0)  # 1 m/s²
        acc_n = np.zeros(n)
        dt = np.full(n, 0.1)

        vel_e, vel_n, pos_e, pos_n = integrate_dead_reckoning(acc_e, acc_n, dt)

        # After 10s at 1 m/s²: pos = 0.5 * a * t² = 0.5 * 1 * 100 = 50m
        total_time = 10.0
        expected_pos = 0.5 * 1.0 * total_time**2
        assert abs(pos_e[-1] - expected_pos) < 1.0, f"Expected ~{expected_pos}m, got {pos_e[-1]:.1f}m"

    def test_gravity_removal(self):
        """Gravity removal should produce near-zero output for stationary sensor."""
        from Data_details.src.inertial_baseline import gravity_removal_simple

        n = 100
        # Simulating a stationary phone: acc = gravity
        grav = np.array([0.0, 0.0, 9.81])
        acc_x = np.full(n, grav[0]) + np.random.randn(n) * 0.01
        acc_y = np.full(n, grav[1]) + np.random.randn(n) * 0.01
        acc_z = np.full(n, grav[2]) + np.random.randn(n) * 0.01
        grav_x = np.full(n, grav[0])
        grav_y = np.full(n, grav[1])
        grav_z = np.full(n, grav[2])

        lin_x, lin_y, lin_z = gravity_removal_simple(acc_x, acc_y, acc_z, grav_x, grav_y, grav_z)

        assert np.abs(np.mean(lin_x)) < 0.1
        assert np.abs(np.mean(lin_y)) < 0.1
        assert np.abs(np.mean(lin_z)) < 0.1


class TestMetrics:
    """Tests for trajectory error metrics."""

    def test_zero_error(self):
        """Identical trajectories should produce zero error."""
        from Data_details.src.metrics import compute_trajectory_metrics

        pos = np.arange(100, dtype=float)
        time_s = np.arange(100) * 0.1

        metrics = compute_trajectory_metrics(pos, pos, pos, pos, time_s, 100.0)
        assert metrics["endpoint_error_m"] == 0.0
        assert metrics["rmse_m"] == 0.0
        assert metrics["drift_pct"] == 0.0

    def test_known_error(self):
        """Known constant offset should produce correct endpoint error."""
        from Data_details.src.metrics import compute_trajectory_metrics

        ref_e = np.zeros(100)
        ref_n = np.zeros(100)
        est_e = np.full(100, 3.0)
        est_n = np.full(100, 4.0)
        time_s = np.arange(100) * 0.1

        metrics = compute_trajectory_metrics(ref_e, ref_n, est_e, est_n, time_s, 100.0)
        assert abs(metrics["endpoint_error_m"] - 5.0) < 0.01  # 3-4-5 triangle
        assert metrics["drift_pct"] == 5.0
