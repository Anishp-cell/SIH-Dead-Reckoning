"""
Phase 6 Robust Multi-Signal Stationary Detector with Hysteresis & Dwell Persistence.
Fuses accelerometer variance, angular rate, gravity deviation, and AI speed predictions.
"""

from collections import deque
from typing import Tuple, Optional, Union
import numpy as np


class RobustStationaryDetector:
    """
    Causal real-time stationary detector with hysteresis state machine.
    
    Prevents false triggers during smooth cruising and eliminates chatter during stop-and-go.
    """

    def __init__(
        self,
        window_size: int = 10,                 # 1.0 s at 10 Hz
        acc_var_thresh: float = 0.04,          # (m/s^2)^2
        gyro_norm_thresh: float = 0.045,       # rad/s (~2.58 deg/s)
        grav_diff_thresh: float = 0.40,        # m/s^2 deviation from g
        speed_thresh: float = 0.50,            # m/s
        gravity_ref: float = 9.80665,
        min_enter_samples: int = 5,            # 0.5 s persistence to enter stationary
        min_exit_samples: int = 2,             # 0.2 s persistence to exit stationary
    ):
        self.window_size = window_size
        self.acc_var_thresh = acc_var_thresh
        self.gyro_norm_thresh = gyro_norm_thresh
        self.grav_diff_thresh = grav_diff_thresh
        self.speed_thresh = speed_thresh
        self.gravity_ref = gravity_ref
        self.min_enter_samples = min_enter_samples
        self.min_exit_samples = min_exit_samples

        self.acc_norm_history = deque(maxlen=window_size)
        self.gyro_norm_history = deque(maxlen=window_size)

        # State machine
        self.is_stationary = False
        self.consecutive_stationary_cand = 0
        self.consecutive_moving_cand = 0
        self.stationary_duration_s = 0.0

    def update(
        self,
        f_meas: np.ndarray,
        omega_meas: np.ndarray,
        ai_speed_mps: Optional[float] = None,
        dt: float = 0.1,
    ) -> Tuple[bool, float, float]:
        """
        Updates detector state with single streaming IMU sample.
        
        Returns:
            is_stationary: Current filtered boolean stationary state
            stationary_duration_s: Continuous duration in seconds if stationary, else 0.0
            confidence: Stationary confidence score in [0.0, 1.0]
        """
        a_norm = float(np.linalg.norm(f_meas))
        g_norm = float(np.linalg.norm(omega_meas))

        self.acc_norm_history.append(a_norm)
        self.gyro_norm_history.append(g_norm)

        # 1. Accelerometer rolling variance
        a_var = float(np.var(self.acc_norm_history)) if len(self.acc_norm_history) > 1 else 0.0
        # 2. Gyroscope mean norm
        g_mean = float(np.mean(self.gyro_norm_history))
        # 3. Specific force deviation from 9.80665 m/s^2
        g_diff = abs(float(np.mean(self.acc_norm_history)) - self.gravity_ref)

        c_var = a_var < self.acc_var_thresh
        c_gyro = g_mean < self.gyro_norm_thresh
        c_grav = g_diff < self.grav_diff_thresh

        # 4. AI speed check (if available)
        c_speed = True
        if ai_speed_mps is not None:
            c_speed = float(ai_speed_mps) < self.speed_thresh

        # Instantaneous candidate
        candidate = c_var and c_gyro and c_grav and c_speed

        # Hysteresis state machine
        if candidate:
            self.consecutive_stationary_cand += 1
            self.consecutive_moving_cand = 0
            if not self.is_stationary:
                if self.consecutive_stationary_cand >= self.min_enter_samples:
                    self.is_stationary = True
                    self.stationary_duration_s = self.consecutive_stationary_cand * dt
            else:
                self.stationary_duration_s += dt
        else:
            self.consecutive_moving_cand += 1
            self.consecutive_stationary_cand = 0
            if self.is_stationary:
                if self.consecutive_moving_cand >= self.min_exit_samples:
                    self.is_stationary = False
                    self.stationary_duration_s = 0.0
            else:
                self.stationary_duration_s = 0.0

        # Confidence calculation
        var_score = max(0.0, 1.0 - (a_var / max(self.acc_var_thresh, 1e-6)))
        gyro_score = max(0.0, 1.0 - (g_mean / max(self.gyro_norm_thresh, 1e-6)))
        grav_score = max(0.0, 1.0 - (g_diff / max(self.grav_diff_thresh, 1e-6)))
        raw_conf = 0.4 * var_score + 0.3 * gyro_score + 0.3 * grav_score
        confidence = float(np.clip(raw_conf if self.is_stationary else (raw_conf * 0.5), 0.0, 1.0))

        return self.is_stationary, self.stationary_duration_s, confidence

    def reset(self) -> None:
        """Resets detector history."""
        self.acc_norm_history.clear()
        self.gyro_norm_history.clear()
        self.is_stationary = False
        self.consecutive_stationary_cand = 0
        self.consecutive_moving_cand = 0
        self.stationary_duration_s = 0.0
