"""
Phase 6 Constraint Noise Models: Adaptive Measurement Covariance Generator.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ConstraintNoiseParameters:
    """Configurable covariance parameters for Phase 6 virtual constraints."""
    sigma_lat_base: float = 0.20       # m/s nominal lateral standard deviation
    sigma_up_base: float = 0.20        # m/s nominal vertical standard deviation
    kappa_dyn: float = 3.0             # Disturbance scaling sensitivity
    sigma_zupt: float = 0.05           # m/s ZUPT velocity clamp uncertainty
    sigma_zaru: float = 0.005          # rad/s ZARU gyro bias observation noise

    def build_nhc_R(self, d_lat: float = 0.0, d_up: float = 0.0) -> np.ndarray:
        """
        Builds adaptive 2x2 measurement covariance matrix for NHC:
            sigma_lat = sigma_lat_base * (1 + kappa_dyn * d_lat)
            sigma_up  = sigma_up_base  * (1 + kappa_dyn * d_up)
            R_NHC = diag(sigma_lat^2, sigma_up^2)
        """
        sig_lat = self.sigma_lat_base * (1.0 + self.kappa_dyn * max(0.0, d_lat))
        sig_up = self.sigma_up_base * (1.0 + self.kappa_dyn * max(0.0, d_up))
        return np.diag([sig_lat**2, sig_up**2]).astype(np.float64)

    def build_zupt_R(self) -> np.ndarray:
        """Builds 3x3 measurement covariance matrix for ZUPT: sigma_zupt^2 * I_3."""
        return np.diag([self.sigma_zupt**2] * 3).astype(np.float64)

    def build_zaru_R(self) -> np.ndarray:
        """Builds 3x3 measurement covariance matrix for ZARU: sigma_zaru^2 * I_3."""
        return np.diag([self.sigma_zaru**2] * 3).astype(np.float64)
