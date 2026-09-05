"""
Real-Time Streaming Zero-Velocity Update (ZUPT) Detector.

Provides causal, sample-by-sample and vectorized rolling-window stationary detection
fusing accelerometer variance, gyroscope energy, and gravity norm consistency.
Enables instant velocity clamping (v = 0) and error-state reset during vehicle stops.
"""

from collections import deque
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd


class ZUPTDetector:
    """
    Causal streaming Zero-Velocity Update (ZUPT) detector.
    Operates sample-by-sample with O(1) time complexity and minimal memory.
    """

    def __init__(
        self,
        window_size: int = 10,                 # 10 samples = 1.0 s at 10 Hz
        acc_var_threshold: float = 0.04,        # (m/s^2)^2 maximum accel variance
        gyro_norm_threshold: float = 0.045,     # rad/s maximum angular rate norm
        gravity_diff_threshold: float = 0.40,   # m/s^2 maximum deviation from 9.80665
        gravity_ref: float = 9.80665,
        min_duration_samples: int = 3,          # Persistence requirement
    ):
        self.window_size = window_size
        self.acc_var_threshold = acc_var_threshold
        self.gyro_norm_threshold = gyro_norm_threshold
        self.gravity_diff_threshold = gravity_diff_threshold
        self.gravity_ref = gravity_ref
        self.min_duration_samples = min_duration_samples

        self.acc_norm_history = deque(maxlen=window_size)
        self.gyro_norm_history = deque(maxlen=window_size)
        self.consecutive_stationary = 0
        self.total_stationary_time_s = 0.0

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float = 0.1) -> Tuple[bool, float]:
        """
        Processes a single IMU sample in real time.

        Parameters:
            acc: 3D acceleration vector [ax, ay, az] in m/s^2
            gyro: 3D angular rate vector [gx, gy, gz] in rad/s
            dt: Sample time interval in seconds

        Returns:
            is_stationary: True if vehicle is currently stationary (ZUPT active)
            stationary_duration_s: Continuous duration of current stop in seconds
        """
        a_norm = float(np.linalg.norm(acc))
        g_norm = float(np.linalg.norm(gyro))

        self.acc_norm_history.append(a_norm)
        self.gyro_norm_history.append(g_norm)

        # 1. Accelerometer variance check (0.0 if single sample)
        a_var = float(np.var(self.acc_norm_history)) if len(self.acc_norm_history) > 1 else 0.0
        # 2. Gyroscope norm check (mean over recent window)
        g_mean = float(np.mean(self.gyro_norm_history))
        # 3. Gravity deviation check
        g_diff = abs(float(np.mean(self.acc_norm_history)) - self.gravity_ref)

        cond_var = a_var < self.acc_var_threshold
        cond_gyro = g_mean < self.gyro_norm_threshold
        cond_grav = g_diff < self.gravity_diff_threshold

        instant_stationary = cond_var and cond_gyro and cond_grav

        if instant_stationary:
            self.consecutive_stationary += 1
            if self.consecutive_stationary >= self.min_duration_samples:
                self.total_stationary_time_s += dt
                return True, self.total_stationary_time_s
            return False, 0.0
        else:
            self.consecutive_stationary = 0
            self.total_stationary_time_s = 0.0
            return False, 0.0

    def reset(self):
        """Resets detector state."""
        self.acc_norm_history.clear()
        self.gyro_norm_history.clear()
        self.consecutive_stationary = 0
        self.total_stationary_time_s = 0.0


def compute_streaming_zupt(
    df: pd.DataFrame,
    window_size: int = 10,
    acc_var_threshold: float = 0.04,
    gyro_norm_threshold: float = 0.045,
    gravity_diff_threshold: float = 0.40,
    gravity_ref: float = 9.80665,
    min_duration_samples: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized / causal rolling-window evaluation of ZUPT across a full DataFrame.

    Parameters:
        df: DataFrame containing acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, dt

    Returns:
        is_stationary: Boolean array of stationary flags
        stationary_duration: Float array of accumulated stop durations in seconds
    """
    ax = df["acc_x"].values
    ay = df["acc_y"].values
    az = df["acc_z"].values
    gx = df["gyro_x"].values
    gy = df["gyro_y"].values
    gz = df["gyro_z"].values
    dt_arr = df["dt"].values if "dt" in df.columns else np.full(len(df), 0.1)

    detector = ZUPTDetector(
        window_size=window_size,
        acc_var_threshold=acc_var_threshold,
        gyro_norm_threshold=gyro_norm_threshold,
        gravity_diff_threshold=gravity_diff_threshold,
        gravity_ref=gravity_ref,
        min_duration_samples=min_duration_samples,
    )

    n = len(df)
    is_stat = np.zeros(n, dtype=bool)
    durations = np.zeros(n, dtype=float)

    for i in range(n):
        acc = np.array([ax[i], ay[i], az[i]])
        gyro = np.array([gx[i], gy[i], gz[i]])
        stat, dur = detector.update(acc, gyro, dt=dt_arr[i])
        is_stat[i] = stat
        durations[i] = dur

    return is_stat, durations
