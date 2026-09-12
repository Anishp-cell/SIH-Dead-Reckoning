"""
Tests for dataset inventory and loader modules.
Uses synthetic data so tests don't depend on the actual IO-VNBD dataset.
"""

import sys
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def create_synthetic_smartphone_csv(path: Path, n_rows: int = 100) -> None:
    """Creates a synthetic smartphone CSV file for testing."""
    np.random.seed(42)
    t_ms = np.arange(0, n_rows * 100, 100)
    data = {
        "GPS LATITUDE (degrees)": 52.4 + np.cumsum(np.random.randn(n_rows) * 0.00001),
        "GPS LONGITUDE (degrees)": -1.5 + np.cumsum(np.random.randn(n_rows) * 0.00001),
        "GPS ALTITUDE (m)": 100 + np.random.randn(n_rows) * 2,
        "GPS SPEED (Kmh)": np.abs(20 + np.random.randn(n_rows) * 5),
        "GPS ACCURACY (m)": 3 + np.random.rand(n_rows) * 2,
        "GPS ORIENTATION (°)": 180 + np.random.randn(n_rows) * 10,
        "GPS SATELLITES IN RANGE": ["10 / 12"] * n_rows,
        "TIME SINCE START (ms)": t_ms,
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)": [f"2024-01-01 10:00:{i//10:02d}:{(i%10)*100:03d}" for i in range(n_rows)],
        "ACCELEROMETER X (m/s²)": np.random.randn(n_rows) * 0.5,
        "ACCELEROMETER Y (m/s²)": np.random.randn(n_rows) * 0.5,
        "ACCELEROMETER Z (m/s²)": 9.8 + np.random.randn(n_rows) * 0.3,
        "GRAVITY X (m/s²)": np.random.randn(n_rows) * 0.01,
        "GRAVITY Y (m/s²)": np.random.randn(n_rows) * 0.01,
        "GRAVITY Z (m/s²)": 9.806 + np.random.randn(n_rows) * 0.001,
        "GYROSCOPE X (rad/s)": np.random.randn(n_rows) * 0.05,
        "GYROSCOPE Y (rad/s)": np.random.randn(n_rows) * 0.05,
        "GYROSCOPE Z (rad/s)": np.random.randn(n_rows) * 0.05,
        "MAGNETIC FIELD X (μT)": -5 + np.random.randn(n_rows) * 1,
        "MAGNETIC FIELD Y (μT)": -25 + np.random.randn(n_rows) * 1,
        "MAGNETIC FIELD Z (μT)": 30 + np.random.randn(n_rows) * 1,
        "ORIENTATION (Azimuth) (°)": 180 + np.random.randn(n_rows) * 10,
        "ORIENTATION (Pitch) (°)": -80 + np.random.randn(n_rows) * 2,
        "ORIENTATION (Roll ) (°)": -155 + np.random.randn(n_rows) * 2,
    }
    pd.DataFrame(data).to_csv(path, index=False)


def create_synthetic_vehicle_csv(path: Path, n_rows: int = 100) -> None:
    """Creates a synthetic vehicle CSV file for testing."""
    np.random.seed(42)
    data = {
        "No of GPS Satellites Available": [11.0] * n_rows,
        "Time Since Start of Day (seconds)": 32869 + np.arange(n_rows) * 0.1,
        "Latitude (degrees)": 52.4 + np.cumsum(np.random.randn(n_rows) * 0.00001),
        "Longitude (degrees)": -1.5 + np.cumsum(np.random.randn(n_rows) * 0.00001),
        "Velocity (km/hr)": np.abs(20 + np.random.randn(n_rows) * 5),
        "Heading (degrees)": 240 + np.random.randn(n_rows) * 5,
        "Height (km)": 0.11 + np.random.randn(n_rows) * 0.001,
        "Vertical velocity (km/hr)": np.random.randn(n_rows) * 0.1,
        "Sample period (seconds)": [0.1] * n_rows,
        "Steering Angle (degrees)": np.random.randn(n_rows) * 10,
        "Wheel Speed Front Left (rad/sec)": 20 + np.random.randn(n_rows) * 1,
        "Wheel Speed Front Right (rad/sec)": 20 + np.random.randn(n_rows) * 1,
        "Wheel Speed Rear Left (rad/sec)": 20 + np.random.randn(n_rows) * 1,
        "Wheel Speed Rear Right (rad/sec)": 20 + np.random.randn(n_rows) * 1,
        "Yaw Rate (deg/sec)": np.random.randn(n_rows) * 2,
        "Indicated Vehicle Speed (km/hr)": np.abs(20 + np.random.randn(n_rows) * 5),
        "Indicated Longitudinal Acceleration (g)": np.random.randn(n_rows) * 0.1,
        "Indicated Lateral Acceleration (g)": np.random.randn(n_rows) * 0.1,
        "Handbrake (0 or 1)": [0.0] * n_rows,
        "Gear Requested (Number fof gear employed 1-5)": [3.0] * n_rows,
        "Gear (Number fof gear employed 1-5)": [3.0] * n_rows,
        "Engine Speed (rev/min)": 1300 + np.random.randn(n_rows) * 100,
        "Coolant Temperature (degrees)": [40.0] * n_rows,
        "Clutch Position (0 or 1)": [0.0] * n_rows,
        "Brake Pressure (psi)": np.maximum(0, np.random.randn(n_rows) * 0.5),
        "Brake Position (0 or 1)": [0.0] * n_rows,
        "Battery Voltage (volts)": 14.2 + np.random.randn(n_rows) * 0.1,
        "Air Temperature (degrees)": [15.0] * n_rows,
        "Accelerator Pedal Position (0 or 1)": np.random.rand(n_rows) * 15,
    }
    pd.DataFrame(data).to_csv(path, index=False)


class TestDatasetLoader:
    """Tests for the DatasetLoader class."""

    def test_load_raw_dataframe(self, tmp_path):
        csv_path = tmp_path / "S-test.csv"
        create_synthetic_smartphone_csv(csv_path)

        from Data_details.src.dataset_loader import DatasetLoader
        loader = DatasetLoader(repo_root=str(tmp_path), cache_dir=str(tmp_path / "cache"))
        # Place file where loader expects it
        df = pd.read_csv(csv_path, encoding="latin1")
        assert len(df) == 100
        assert len(df.columns) == 24

    def test_standardize_smartphone_df(self, tmp_path):
        csv_path = tmp_path / "S-test.csv"
        create_synthetic_smartphone_csv(csv_path)

        from Data_details.src.dataset_loader import DatasetLoader
        loader = DatasetLoader(repo_root=str(tmp_path), cache_dir=str(tmp_path / "cache"))
        raw_df = pd.read_csv(csv_path, encoding="latin1")
        raw_df.columns = raw_df.columns.str.strip()
        std_df = loader.standardize_smartphone_df(raw_df)

        assert "time_s" in std_df.columns
        assert "dt" in std_df.columns
        assert "acc_x" in std_df.columns
        assert "gps_lat" in std_df.columns
        assert "gyro_x" in std_df.columns
        assert std_df["time_s"].iloc[0] == 0.0
        assert all(std_df["dt"] > 0)

    def test_standardize_vehicle_df(self, tmp_path):
        csv_path = tmp_path / "V-test.csv"
        create_synthetic_vehicle_csv(csv_path)

        from Data_details.src.dataset_loader import DatasetLoader
        loader = DatasetLoader(repo_root=str(tmp_path), cache_dir=str(tmp_path / "cache"))
        raw_df = pd.read_csv(csv_path, encoding="latin1")
        raw_df.columns = raw_df.columns.str.strip()
        std_df = loader.standardize_vehicle_df(raw_df)

        assert "time_s" in std_df.columns
        assert "v_gps_lat" in std_df.columns
        assert "v_speed_kmh" in std_df.columns
        assert "v_yaw_rate_degs" in std_df.columns


class TestSchemaInspector:
    """Tests for the schema inspector module."""

    def test_inspect_raw_schema(self, tmp_path):
        csv_path = tmp_path / "S-test.csv"
        create_synthetic_smartphone_csv(csv_path)

        from Data_details.src.schema_inspector import inspect_raw_schema
        schema = inspect_raw_schema(csv_path)

        assert schema["n_rows"] == 100
        assert schema["n_cols"] == 24
        assert len(schema["columns"]) == 24
        assert schema["delimiter"] == ","

    def test_build_data_dictionary(self, tmp_path):
        csv_path = tmp_path / "S-test.csv"
        create_synthetic_smartphone_csv(csv_path)

        from Data_details.src.dataset_loader import DatasetLoader
        from Data_details.src.schema_inspector import inspect_raw_schema, build_data_dictionary

        loader = DatasetLoader(repo_root=str(tmp_path), cache_dir=str(tmp_path / "cache"))
        raw_df = pd.read_csv(csv_path, encoding="latin1")
        raw_df.columns = raw_df.columns.str.strip()
        std_df = loader.standardize_smartphone_df(raw_df)
        schema = inspect_raw_schema(csv_path)

        dd = build_data_dictionary(schema, std_df, "smartphone", "S-test.csv")
        assert len(dd) > 0
        assert "column_name" in dd.columns
        assert "semantic_meaning" in dd.columns
        assert "unit" in dd.columns
