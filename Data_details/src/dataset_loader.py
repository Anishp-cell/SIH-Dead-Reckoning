"""
Dataset Loader & LFS Resolver for IO-VNBD Dataset.
Handles LFS pointer resolution, CSV caching, encoding normalization, and timestamp parsing.
"""

import os
import re
import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

LFS_BATCH_URL = "https://github.com/onyekpeu/IO-VNBD.git/info/lfs/objects/batch"


class DatasetLoader:
    """Loads and standardizes IO-VNBD smartphone and vehicle datasets."""

    def __init__(self, repo_root: str = "IO-VNBD-master", cache_dir: str = "Data_details/data/raw"):
        self.repo_root = Path(repo_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_lfs_pointer(self, file_path: Path) -> Tuple[bool, Optional[str], Optional[int]]:
        """Checks if a file is a Git LFS pointer file."""
        if not file_path.exists():
            return False, None, None
        if file_path.stat().st_size > 1024:
            return False, None, None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            oid_match = re.search(r"oid sha256:([a-f0-9]+)", content)
            size_match = re.search(r"size (\d+)", content)
            if oid_match and size_match:
                return True, oid_match.group(1), int(size_match.group(1))
        except Exception as e:
            logger.debug(f"Error checking LFS pointer for {file_path}: {e}")
        return False, None, None

    def fetch_lfs_object(self, oid: str, size: int, output_path: Path) -> bool:
        """Downloads an LFS object from GitHub LFS batch endpoint."""
        logger.info(f"Downloading LFS file ({size / 1e6:.2f} MB) -> {output_path.name}...")
        payload = {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}],
        }
        req = urllib.request.Request(
            LFS_BATCH_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            download_url = data["objects"][0]["actions"]["download"]["href"]
            urllib.request.urlretrieve(download_url, output_path)
            logger.info(f"Successfully downloaded {output_path.name} ({output_path.stat().st_size} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to download LFS object {oid}: {e}")
            raise

    def get_real_file_path(self, relative_path: str) -> Path:
        """Resolves full data file path, downloading LFS object to cache if needed."""
        source_path = self.repo_root / relative_path
        cache_path = self.cache_dir / Path(relative_path).name

        # Check if already cached
        if cache_path.exists() and cache_path.stat().st_size > 2000:
            return cache_path

        # Check if source_path is full data
        if source_path.exists():
            is_lfs, oid, size = self.is_lfs_pointer(source_path)
            if not is_lfs and source_path.stat().st_size > 2000:
                return source_path
            if is_lfs and oid and size:
                self.fetch_lfs_object(oid, size, cache_path)
                return cache_path

        # If file only exists in cache or needs direct resolution
        if cache_path.exists():
            return cache_path

        raise FileNotFoundError(f"Could not locate or resolve file: {relative_path}")

    def load_raw_dataframe(self, relative_path: str, encoding: str = "latin1") -> pd.DataFrame:
        """Loads raw CSV into DataFrame with standardized column headers."""
        real_path = self.get_real_file_path(relative_path)
        logger.info(f"Loading CSV: {real_path} ({real_path.stat().st_size / 1e6:.2f} MB)")
        df = pd.read_csv(real_path, encoding=encoding)
        # Strip trailing/leading spaces in column headers
        df.columns = df.columns.str.strip()
        return df

    def standardize_smartphone_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and adds parsed timestamps for smartphone data."""
        df = df.copy()

        # Map column names to clean snake_case identifiers
        name_map = {}
        for col in df.columns:
            c_lower = col.lower()
            if "latitude" in c_lower:
                name_map[col] = "gps_lat"
            elif "longitude" in c_lower:
                name_map[col] = "gps_lon"
            elif "altitude" in c_lower:
                name_map[col] = "gps_alt"
            elif "gps speed" in c_lower:
                name_map[col] = "gps_speed_kmh"
            elif "gps accuracy" in c_lower:
                name_map[col] = "gps_acc_m"
            elif "gps orientation" in c_lower:
                name_map[col] = "gps_heading_deg"
            elif "satellites" in c_lower:
                name_map[col] = "gps_sats"
            elif "time since start" in c_lower:
                name_map[col] = "time_ms"
            elif "date" in c_lower:
                name_map[col] = "datetime_raw"
            elif "accelerometer x" in c_lower:
                name_map[col] = "acc_x"
            elif "accelerometer y" in c_lower:
                name_map[col] = "acc_y"
            elif "accelerometer z" in c_lower:
                name_map[col] = "acc_z"
            elif "gravity x" in c_lower:
                name_map[col] = "grav_x"
            elif "gravity y" in c_lower:
                name_map[col] = "grav_y"
            elif "gravity z" in c_lower:
                name_map[col] = "grav_z"
            elif "gyroscope x" in c_lower or ("gyroscope" in c_lower and "yaw" in c_lower):
                name_map[col] = "gyro_x"
            elif "gyroscope y" in c_lower or ("gyroscope" in c_lower and "pitch" in c_lower):
                name_map[col] = "gyro_y"
            elif "gyroscope z" in c_lower or ("gyroscope" in c_lower and "roll" in c_lower):
                name_map[col] = "gyro_z"
            elif "magnetic field x" in c_lower:
                name_map[col] = "mag_x"
            elif "magnetic field y" in c_lower:
                name_map[col] = "mag_y"
            elif "magnetic field z" in c_lower:
                name_map[col] = "mag_z"
            elif "azimuth" in c_lower or ("orientation" in c_lower and "yaw" in c_lower):
                name_map[col] = "ori_yaw_deg"
            elif "pitch" in c_lower and "orientation" in c_lower:
                name_map[col] = "ori_pitch_deg"
            elif "roll" in c_lower and "orientation" in c_lower:
                name_map[col] = "ori_roll_deg"

        df.rename(columns=name_map, inplace=True)

        # Standardize timestamp
        if "time_ms" in df.columns:
            # First time offset normalized to start at 0
            df["time_s"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0
        else:
            df["time_s"] = np.arange(len(df)) * 0.1

        # Calculate actual dt
        df["dt"] = df["time_s"].diff().fillna(0.1)
        # Avoid non-positive dt for first element or jitter glitches
        df.loc[df["dt"] <= 0, "dt"] = 0.1

        # GPS Speed in m/s
        if "gps_speed_kmh" in df.columns:
            df["gps_speed_mps"] = df["gps_speed_kmh"] / 3.6

        return df

    def standardize_vehicle_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and adds parsed timestamps for vehicle ECU data."""
        df = df.copy()

        name_map = {}
        for col in df.columns:
            c_lower = col.lower()
            if "satellites" in c_lower:
                name_map[col] = "v_gps_sats"
            elif "time since start of day" in c_lower:
                name_map[col] = "v_time_day_s"
            elif "latitude" in c_lower:
                name_map[col] = "v_gps_lat"
            elif "longitude" in c_lower:
                name_map[col] = "v_gps_lon"
            elif "vertical velocity" in c_lower:
                name_map[col] = "v_vert_vel_kmh"
            elif "velocity" in c_lower:
                name_map[col] = "v_gps_speed_kmh"
            elif "heading" in c_lower:
                name_map[col] = "v_gps_heading_deg"
            elif "height" in c_lower:
                name_map[col] = "v_gps_height_km"
            elif "sample period" in c_lower:
                name_map[col] = "v_sample_period_s"
            elif "steering angle" in c_lower:
                name_map[col] = "v_steering_deg"
            elif "wheel speed front left" in c_lower:
                name_map[col] = "v_wheel_fl_rads"
            elif "wheel speed front right" in c_lower:
                name_map[col] = "v_wheel_fr_rads"
            elif "wheel speed rear left" in c_lower:
                name_map[col] = "v_wheel_rl_rads"
            elif "wheel speed rear right" in c_lower:
                name_map[col] = "v_wheel_rr_rads"
            elif "yaw rate" in c_lower:
                name_map[col] = "v_yaw_rate_degs"
            elif "indicated vehicle speed" in c_lower:
                name_map[col] = "v_speed_kmh"
            elif "indicated longitudinal acceleration" in c_lower:
                name_map[col] = "v_long_acc_g"
            elif "indicated lateral acceleration" in c_lower:
                name_map[col] = "v_lat_acc_g"
            elif "handbrake" in c_lower:
                name_map[col] = "v_handbrake"
            elif "gear requested" in c_lower:
                name_map[col] = "v_gear_req"
            elif "gear" in c_lower:
                name_map[col] = "v_gear"
            elif "engine speed" in c_lower:
                name_map[col] = "v_engine_rpm"
            elif "coolant temperature" in c_lower:
                name_map[col] = "v_coolant_temp_c"
            elif "clutch position" in c_lower:
                name_map[col] = "v_clutch"
            elif "brake pressure" in c_lower:
                name_map[col] = "v_brake_psi"
            elif "brake position" in c_lower:
                name_map[col] = "v_brake_pos"
            elif "battery voltage" in c_lower:
                name_map[col] = "v_battery_v"
            elif "air temperature" in c_lower:
                name_map[col] = "v_air_temp_c"
            elif "accelerator pedal position" in c_lower:
                name_map[col] = "v_accel_pedal"

        df.rename(columns=name_map, inplace=True)

        if "v_time_day_s" in df.columns:
            df["time_s"] = df["v_time_day_s"] - df["v_time_day_s"].iloc[0]
        else:
            df["time_s"] = np.arange(len(df)) * 0.1

        df["dt"] = df["time_s"].diff().fillna(0.1)
        df.loc[df["dt"] <= 0, "dt"] = 0.1

        if "v_speed_kmh" in df.columns:
            df["v_speed_mps"] = df["v_speed_kmh"] / 3.6

        return df

    def load_sequence(self, smartphone_rel_path: str, vehicle_rel_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Loads and standardizes a synchronized sequence."""
        s_df = self.load_raw_dataframe(smartphone_rel_path)
        s_df_std = self.standardize_smartphone_df(s_df)

        v_df_std = None
        if vehicle_rel_path:
            try:
                v_df = self.load_raw_dataframe(vehicle_rel_path)
                v_df_std = self.standardize_vehicle_df(v_df)
            except Exception as e:
                logger.warning(f"Could not load vehicle counterpart {vehicle_rel_path}: {e}")

        return s_df_std, v_df_std



def resample_dataframe(
    df: pd.DataFrame,
    target_freq_hz: float = 100.0,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Resamples a sensor DataFrame onto a uniform high-rate time grid.
    
    Supports 10 Hz -> 50/100/200 Hz upsampling (or downsampling)
    using piecewise linear or cubic spline interpolation on numeric columns.
    Preserves first/last timestamps exactly.
    """
    from scipy.interpolate import interp1d
    
    if "time_s" not in df.columns:
        raise ValueError("DataFrame must contain 'time_s' column for resampling")
        
    t_orig = df["time_s"].values
    t_start = t_orig[0]
    t_end = t_orig[-1]
    
    dt_new = 1.0 / target_freq_hz
    t_new = np.arange(t_start, t_end, dt_new)
    if len(t_new) == 0 or t_new[-1] < t_end:
        t_new = np.append(t_new, t_end)
        
    resampled_data = {"time_s": t_new}
    dt_vals = np.diff(t_new, prepend=t_new[0] - dt_new)
    dt_vals[0] = dt_new
    resampled_data["dt"] = dt_vals
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ("time_s", "dt"):
            continue
        vals = df[col].values.astype(float)
        if np.isnan(vals).any():
            vals = pd.Series(vals).ffill().bfill().values
            
        interp_func = interp1d(t_orig, vals, kind=method, bounds_error=False, fill_value="extrapolate")
        resampled_data[col] = interp_func(t_new)
        
    return pd.DataFrame(resampled_data)


def extract_ground_truth_odometry(
    v_df: pd.DataFrame,
    wheel_radius_m: float = 0.301,
) -> pd.DataFrame:
    """
    Extracts high-fidelity ground truth forward velocity and displacement
    from CAN-bus four-wheel speeds and VBOX DGPS velocity.
    """
    df_out = pd.DataFrame()
    df_out["time_s"] = v_df["time_s"].values
    
    wheel_cols = ["v_wheel_fl_rads", "v_wheel_fr_rads", "v_wheel_rl_rads", "v_wheel_rr_rads"]
    available_wheels = [c for c in wheel_cols if c in v_df.columns]
    
    if available_wheels:
        mean_wheel_rads = v_df[available_wheels].mean(axis=1).values
        df_out["v_wheel_speed_mps"] = mean_wheel_rads * wheel_radius_m
    else:
        df_out["v_wheel_speed_mps"] = 0.0
        
    if "v_speed_mps" in v_df.columns:
        df_out["v_can_speed_mps"] = v_df["v_speed_mps"].values
    elif "v_speed_kmh" in v_df.columns:
        df_out["v_can_speed_mps"] = v_df["v_speed_kmh"].values / 3.6
    else:
        df_out["v_can_speed_mps"] = df_out["v_wheel_speed_mps"]
        
    df_out["v_true_mps"] = np.where(df_out["v_can_speed_mps"] > 0, df_out["v_can_speed_mps"], df_out["v_wheel_speed_mps"])
    
    dt = v_df["dt"].values if "dt" in v_df.columns else np.full(len(v_df), 0.1)
    df_out["dist_true_m"] = np.cumsum(df_out["v_true_mps"].values * dt)
    
    return df_out

