"""
Phase 6 Causal Disturbance Detector for Adaptive Virtual Constraint Weighting.
Detects active cornering maneuvers, road shocks/potholes, and chassis vibration.
"""

from dataclasses import dataclass
from typing import Union, List, Optional
import numpy as np


@dataclass
class DisturbanceReport:
    """Diagnostic telemetry regarding dynamic vehicle disturbance."""
    d_lat: float             # Lateral maneuver disturbance index [0, inf)
    d_up: float              # Vertical shock / bump disturbance index [0, inf)
    c_nhc: float             # NHC constraint confidence score [0.0, 1.0]
    is_cornering: bool       # True if active turning maneuver detected
    is_shock: bool           # True if vertical shock / bump detected
    is_pothole_shock: bool = False      # Sharp impulse shock (> pothole threshold)
    shock_magnitude_mps2: float = 0.0   # Vertical deviation from 1g
    r_up_scale_factor: float = 1.0      # Covariance inflation factor for vertical NHC


class DisturbanceDetector:
    """
    Computes causal disturbance scores from raw and conditioned IMU channels.
    Modulates NHC measurement covariance R_NHC to prevent fighting legitimate maneuvers
    and isolates pothole/speed-breaker impulse shocks on Indian roads.
    """

    def __init__(
        self,
        lat_accel_onset: float = 1.50,     # m/s^2 onset of active cornering
        yaw_rate_onset: float = 0.15,      # rad/s (~8.6 deg/s) onset of active turning
        vert_accel_onset: float = 2.00,    # m/s^2 deviation from g for vertical shocks
        pothole_shock_threshold: float = 10.0, # m/s^2 deviation for severe road pothole shock
        vibe_onset: float = 1.00,          # Vibration energy threshold
        gravity_ref: float = 9.80665,
    ):
        self.lat_accel_onset = lat_accel_onset
        self.yaw_rate_onset = yaw_rate_onset
        self.vert_accel_onset = vert_accel_onset
        self.pothole_shock_threshold = pothole_shock_threshold
        self.vibe_onset = vibe_onset
        self.gravity_ref = gravity_ref

    def evaluate(
        self,
        f_meas: np.ndarray,
        omega_meas: np.ndarray,
        vibration_energy: float = 0.0,
    ) -> DisturbanceReport:
        """
        Evaluates disturbance scores from body specific force f_meas and angular rate omega_meas.
        
        Parameters:
            f_meas: (3,) Specific force in Body frame [a_fwd, a_lat, a_up] (m/s^2)
            omega_meas: (3,) Angular rate in Body frame [roll_rate, pitch_rate, yaw_rate] (rad/s)
            vibration_energy: Optional vibration energy channel from feature extractor
        """
        a_fwd = float(f_meas[0])
        a_lat = float(f_meas[1])
        a_up = float(f_meas[2])
        w_yaw = float(omega_meas[2])

        # 1. Lateral cornering disturbance
        term_lat_acc = abs(a_lat) / max(self.lat_accel_onset, 1e-4)
        term_yaw_rate = abs(w_yaw) / max(self.yaw_rate_onset, 1e-4)
        d_lat = max(0.0, (term_lat_acc + term_yaw_rate) - 1.0)
        is_cornering = bool(d_lat > 0.0)

        # 2. Vertical road bump / pothole disturbance
        vert_deviation = abs(a_up - self.gravity_ref)
        term_vert_acc = vert_deviation / max(self.vert_accel_onset, 1e-4)
        term_vibe = max(0.0, vibration_energy) / max(self.vibe_onset, 1e-4)
        d_up = max(0.0, (term_vert_acc + term_vibe) - 1.0)
        is_shock = bool(d_up > 0.0)

        # 3. Pothole / Speed Breaker Impulse Gating
        is_pothole = bool(vert_deviation >= self.pothole_shock_threshold)
        if is_pothole:
            r_scale = float(min(100.0, 1.0 + (vert_deviation / 2.0)**2))
        elif is_shock:
            r_scale = float(min(10.0, 1.0 + (vert_deviation / 4.0)))
        else:
            r_scale = 1.0

        # 4. Overall NHC confidence score in (0.0, 1.0]
        c_nhc = float(1.0 / (1.0 + 0.5 * (d_lat + d_up)))

        return DisturbanceReport(
            d_lat=d_lat,
            d_up=d_up,
            c_nhc=c_nhc,
            is_cornering=is_cornering,
            is_shock=is_shock,
            is_pothole_shock=is_pothole,
            shock_magnitude_mps2=vert_deviation,
            r_up_scale_factor=r_scale,
        )
