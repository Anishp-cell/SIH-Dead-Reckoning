"""
Phase 5 IMU Stochastic Noise Parameters & Discrete Process Covariance Q_k.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class IMUNoiseParameters:
    """
    IMU stochastic error parameters:
        sigma_a: Accelerometer velocity random walk (noise density) [m/s^2 / sqrt(Hz)]
        sigma_g: Gyroscope angular random walk (noise density) [rad/s / sqrt(Hz)]
        sigma_ba: Accelerometer bias instability / random walk [m/s^3 / sqrt(Hz)]
        sigma_bg: Gyroscope bias instability / random walk [rad/s^2 / sqrt(Hz)]
    """
    sigma_a: float = 0.08       # m/s^2 / sqrt(Hz) (estimated from IO-VNBD stationary Vw1)
    sigma_g: float = 0.005      # rad/s / sqrt(Hz)
    sigma_ba: float = 1e-4      # m/s^3 / sqrt(Hz)
    sigma_bg: float = 1e-5      # rad/s^2 / sqrt(Hz)

    def build_discrete_Q(self, dt: float) -> np.ndarray:
        """
        Constructs the 15x15 discrete process noise covariance matrix Q_k over timestep dt.
        
        Blocks:
            Q_dp: (1/3) * sigma_a^2 * dt^3 * I_3
            Q_dv: sigma_a^2 * dt * I_3
            Q_dtheta: sigma_g^2 * dt * I_3
            Q_dba: sigma_ba^2 * dt * I_3
            Q_dbg: sigma_bg^2 * dt * I_3
            Q_dp_dv: (1/2) * sigma_a^2 * dt^2 * I_3
        """
        Q = np.zeros((15, 15), dtype=np.float64)
        I3 = np.eye(3, dtype=np.float64)

        var_a = self.sigma_a ** 2
        var_g = self.sigma_g ** 2
        var_ba = self.sigma_ba ** 2
        var_bg = self.sigma_bg ** 2

        # Position block
        Q[0:3, 0:3] = (1.0 / 3.0) * var_a * (dt ** 3) * I3
        # Velocity block
        Q[3:6, 3:6] = var_a * dt * I3
        # Position-Velocity cross-coupling
        Q[0:3, 3:6] = 0.5 * var_a * (dt ** 2) * I3
        Q[3:6, 0:3] = 0.5 * var_a * (dt ** 2) * I3

        # Attitude block
        Q[6:9, 6:9] = var_g * dt * I3

        # Accelerometer bias block
        Q[9:12, 9:12] = var_ba * dt * I3

        # Gyroscope bias block
        Q[12:15, 12:15] = var_bg * dt * I3

        return Q
