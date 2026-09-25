"""
Multi-Grade Sensor Noise Profiles for Industrial & Tactical Inertial Navigation:
Defines exact Allan variance noise characteristics, continuous spectral densities,
and discrete-time noise parameters across:
1. Consumer Smartphone MEMS (Bosch BMI160 / TDK ICM-42688)
2. Tactical Grade MEMS (Analog Devices ADIS16490 / Honeywell HG4930)
3. Aerospace Fiber-Optic Gyroscope (FOG / Honeywell HG9900 / KVH DSP-1760)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


class SensorGrade(str, Enum):
    SMARTPHONE_MEMS = "SMARTPHONE_MEMS"
    TACTICAL_MEMS = "TACTICAL_MEMS"
    AEROSPACE_FOG = "AEROSPACE_FOG"


@dataclass(frozen=True)
class SensorProfile:
    """
    Physical continuous-time and discrete-time noise specification for an IMU sensor grade.
    """
    grade: SensorGrade
    name: str
    nominal_rate_hz: float

    # Continuous Noise Spectral Densities (PSD sqrt)
    # Velocity Random Walk (VRW) / Accel White Noise Density: [m/s^2 / sqrt(Hz)]
    accel_noise_density: float

    # Angular Random Walk (ARW) / Gyro White Noise Density: [rad/s / sqrt(Hz)]
    gyro_noise_density: float

    # Bias Instability (1/f flicker noise floor)
    accel_bias_instability: float       # [m/s^2]
    gyro_bias_instability: float        # [rad/s]

    # Bias Random Walk / Markov Process Driving Noise Density
    accel_bias_rw: float                # [m/s^3 / sqrt(Hz)]
    gyro_bias_rw: float                 # [rad/s^2 / sqrt(Hz)]

    # Dynamic Limits
    max_accel_mps2: float = 160.0       # +-16g full scale
    max_gyro_rads: float = 34.9         # +-2000 deg/s full scale

    def get_discrete_noises(self, dt: float) -> Tuple[float, float, float, float]:
        """
        Converts continuous spectral densities to discrete-time standard deviations
        for an integration interval dt:
        sigma_d = PSD / sqrt(dt).
        
        Returns:
            (sigma_accel_d, sigma_gyro_d, sigma_ba_d, sigma_bg_d)
        """
        dt_safe = max(1e-6, float(dt))
        sqrt_dt = np.sqrt(dt_safe)

        sigma_accel_d = self.accel_noise_density / sqrt_dt
        sigma_gyro_d = self.gyro_noise_density / sqrt_dt

        # Bias random walk discrete std dev
        sigma_ba_d = self.accel_bias_rw * sqrt_dt
        sigma_bg_d = self.gyro_bias_rw * sqrt_dt

        return (
            float(sigma_accel_d),
            float(sigma_gyro_d),
            float(sigma_ba_d),
            float(sigma_bg_d),
        )

    def get_process_noise_covariances(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns continuous-to-discrete process noise diagonal sub-matrices:
        Q_v (3x3), Q_theta (3x3), Q_ba (3x3), Q_bg (3x3).
        """
        dt_safe = max(1e-6, float(dt))
        q_v_val = (self.accel_noise_density ** 2) * dt_safe
        q_th_val = (self.gyro_noise_density ** 2) * dt_safe
        q_ba_val = (self.accel_bias_rw ** 2) * dt_safe
        q_bg_val = (self.gyro_bias_rw ** 2) * dt_safe

        return (
            np.eye(3, dtype=np.float64) * q_v_val,
            np.eye(3, dtype=np.float64) * q_th_val,
            np.eye(3, dtype=np.float64) * q_ba_val,
            np.eye(3, dtype=np.float64) * q_bg_val,
        )

    def simulate_step_noise(
        self,
        dt: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates zero-mean discrete Gaussian white noise samples for one IMU step:
        Returns:
            (delta_accel_mps2_3d, delta_gyro_rads_3d)
        """
        r = rng if rng is not None else np.random
        sigma_a, sigma_w, _, _ = self.get_discrete_noises(dt)
        noise_a = r.normal(0.0, sigma_a, size=3)
        noise_w = r.normal(0.0, sigma_w, size=3)
        return noise_a, noise_w


# Authoritative Industrial Calibration Profiles
SENSOR_PROFILES: Dict[SensorGrade, SensorProfile] = {
    SensorGrade.SMARTPHONE_MEMS: SensorProfile(
        grade=SensorGrade.SMARTPHONE_MEMS,
        name="Bosch BMI160 / TDK ICM-42688 (Smartphone MEMS)",
        nominal_rate_hz=100.0,
        accel_noise_density=1.47e-3,       # 150 ug/rtHz
        gyro_noise_density=1.22e-4,        # 0.007 deg/s/rtHz
        accel_bias_instability=9.81e-4,    # 0.1 mg
        gyro_bias_instability=4.85e-5,     # 10 deg/h
        accel_bias_rw=1.0e-4,
        gyro_bias_rw=5.0e-6,
    ),
    SensorGrade.TACTICAL_MEMS: SensorProfile(
        grade=SensorGrade.TACTICAL_MEMS,
        name="Analog Devices ADIS16490 / Honeywell HG4930 (Tactical MEMS)",
        nominal_rate_hz=200.0,
        accel_noise_density=1.47e-4,       # 15 ug/rtHz
        gyro_noise_density=1.75e-5,        # 0.001 deg/s/rtHz
        accel_bias_instability=1.96e-4,    # 0.02 mg
        gyro_bias_instability=2.42e-6,     # 0.5 deg/h
        accel_bias_rw=1.0e-5,
        gyro_bias_rw=5.0e-7,
    ),
    SensorGrade.AEROSPACE_FOG: SensorProfile(
        grade=SensorGrade.AEROSPACE_FOG,
        name="KVH DSP-1760 / Honeywell HG9900 (Fiber-Optic Gyroscope)",
        nominal_rate_hz=200.0,
        accel_noise_density=4.90e-5,       # 5 ug/rtHz
        gyro_noise_density=1.75e-6,        # 0.0001 deg/s/rtHz
        accel_bias_instability=1.96e-5,    # 0.002 mg
        gyro_bias_instability=4.85e-8,     # 0.01 deg/h
        accel_bias_rw=1.0e-6,
        gyro_bias_rw=5.0e-8,
    ),
}
