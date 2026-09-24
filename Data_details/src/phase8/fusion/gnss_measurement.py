"""
Phase 8 GNSS Measurement Observation Models & Analytical 3x15 Jacobians
Supports 3D position and 3D velocity observation with full IMU-to-antenna lever-arm compensation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from Data_details.src.phase5.core.quaternion import quaternion_to_rotation_matrix


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Constructs 3x3 skew-symmetric matrix [v]x such that [v]x @ u = v x u.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


@dataclass
class GNSSMeasurementConfig:
    """
    Configuration parameters for GNSS measurement observation and covariance.
    """
    lever_arm_b: np.ndarray = None           # [lx, ly, lz] antenna position in body frame (meters)
    default_horiz_std_m: float = 2.5         # Nominal smartphone horizontal 1-sigma (m)
    vert_to_horiz_ratio: float = 2.0         # GNSS vertical error scaling (~2x horizontal)
    default_vel_std_mps: float = 0.5         # Nominal Doppler velocity 1-sigma (m/s)
    min_pos_std_m: float = 0.5               # Covariance floor to prevent filter overconfidence
    min_vel_std_mps: float = 0.1             # Velocity covariance floor

    def __post_init__(self):
        if self.lever_arm_b is None:
            self.lever_arm_b = np.zeros(3, dtype=np.float64)
        else:
            self.lever_arm_b = np.asarray(self.lever_arm_b, dtype=np.float64).ravel()


class GNSSMeasurementModel:
    """
    Rigorous observation equations and analytical 3x15 Jacobians for GNSS position and velocity.
    State vector error definition:
      delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]^T (15 states)
    """

    def __init__(self, config: Optional[GNSSMeasurementConfig] = None):
        self.config = config or GNSSMeasurementConfig()

    def predict_position(
        self,
        p_enu: np.ndarray,
        q: np.ndarray,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predicts GNSS antenna position in local ENU frame:
        h_p(x) = p_e^n + R_b^n * l^b
        """
        l_b = self.config.lever_arm_b if lever_arm_b is None else lever_arm_b
        R_nb = quaternion_to_rotation_matrix(q)
        return p_enu + (R_nb @ l_b)

    def position_jacobian(
        self,
        q: np.ndarray,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes 3x15 analytical Jacobian H_p = d(h_p)/d(delta_x):
        d(h_p)/d(delta_p) = I_3
        d(h_p)/d(delta_v) = 0_3
        d(h_p)/d(delta_theta) = -[R_b^n * l^b]x
        d(h_p)/d(delta_ba) = 0_3
        d(h_p)/d(delta_bg) = 0_3
        """
        l_b = self.config.lever_arm_b if lever_arm_b is None else lever_arm_b
        R_nb = quaternion_to_rotation_matrix(q)
        arm_n = R_nb @ l_b

        H_p = np.zeros((3, 15), dtype=np.float64)
        H_p[0:3, 0:3] = np.eye(3, dtype=np.float64)
        # Body-frame rotation error: d(R_nb * l_b)/d(delta_theta_b) = -R_nb * [l_b]x
        H_p[0:3, 6:9] = -R_nb @ skew_symmetric(l_b)
        return H_p

    def predict_velocity(
        self,
        v_enu: np.ndarray,
        q: np.ndarray,
        omega_b: np.ndarray,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predicts GNSS antenna velocity in local ENU frame:
        h_v(x) = v_e^n + R_b^n * (omega_b x l^b)
        """
        l_b = self.config.lever_arm_b if lever_arm_b is None else lever_arm_b
        R_nb = quaternion_to_rotation_matrix(q)
        omega_cross_l = np.cross(omega_b, l_b)
        return v_enu + (R_nb @ omega_cross_l)

    def velocity_jacobian(
        self,
        q: np.ndarray,
        omega_b: np.ndarray,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes 3x15 analytical Jacobian H_v = d(h_v)/d(delta_x):
        d(h_v)/d(delta_p) = 0_3
        d(h_v)/d(delta_v) = I_3
        d(h_v)/d(delta_theta_b) = -R_nb * [omega_b x l^b]x
        d(h_v)/d(delta_ba) = 0_3
        d(h_v)/d(delta_bg) = R_nb * [l^b]x
        """
        l_b = self.config.lever_arm_b if lever_arm_b is None else lever_arm_b
        R_nb = quaternion_to_rotation_matrix(q)
        omega_cross_l = np.cross(omega_b, l_b)

        H_v = np.zeros((3, 15), dtype=np.float64)
        H_v[0:3, 3:6] = np.eye(3, dtype=np.float64)
        H_v[0:3, 6:9] = -R_nb @ skew_symmetric(omega_cross_l)
        H_v[0:3, 12:15] = R_nb @ skew_symmetric(l_b)
        return H_v

    def position_covariance(
        self,
        stated_acc_m: float,
        inflation_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Forms 3x3 diagonal position measurement noise matrix R_p.
        sigma_horiz = max(stated_acc_m, min_pos_std_m)
        sigma_vert = sigma_horiz * vert_to_horiz_ratio
        R_p = diag(sigma_horiz^2, sigma_horiz^2, sigma_vert^2) * inflation_factor
        """
        sigma_h = max(self.config.min_pos_std_m, stated_acc_m)
        sigma_v = sigma_h * self.config.vert_to_horiz_ratio

        r_diag = np.array([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2], dtype=np.float64) * inflation_factor
        return np.diag(r_diag)

    def velocity_covariance(
        self,
        vel_acc_mps: Optional[float] = None,
        inflation_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Forms 3x3 diagonal velocity measurement noise matrix R_v.
        """
        sigma_v = max(self.config.min_vel_std_mps, vel_acc_mps if vel_acc_mps is not None else self.config.default_vel_std_mps)
        r_diag = np.array([sigma_v ** 2, sigma_v ** 2, (sigma_v * 1.5) ** 2], dtype=np.float64) * inflation_factor
        return np.diag(r_diag)

    def predict_velocity_2d(
        self,
        v_enu: np.ndarray,
        q: np.ndarray,
        omega_b: np.ndarray,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predicts 2D horizontal GNSS antenna velocity in local ENU frame:
        h_{v,2D}(x) = [v_E, v_N]^T + (R_b^n * (omega_b x l^b))_{1:2}
        Eliminates unmeasured vertical velocity fabrication.
        """
        if len(v_enu) == 4 and len(q) == 3:
            v_enu, q = q, v_enu
        v_3d = self.predict_velocity(v_enu, q, omega_b, lever_arm_b)
        return v_3d[0:2]

    def velocity_2d_jacobian(
        self,
        q: np.ndarray,
        omega_b: np.ndarray,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes 2x15 analytical Jacobian H_{v,2D} for horizontal velocity observations.
        """
        H_3d = self.velocity_jacobian(q, omega_b, lever_arm_b)
        return H_3d[0:2, :]

    def velocity_2d_covariance(
        self,
        vel_acc_mps: Optional[float] = None,
        inflation_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Forms 2x2 diagonal horizontal velocity measurement noise matrix R_{v,2D}.
        """
        sigma_v = max(self.config.min_vel_std_mps, vel_acc_mps if vel_acc_mps is not None else self.config.default_vel_std_mps)
        r_diag = np.array([sigma_v ** 2, sigma_v ** 2], dtype=np.float64) * inflation_factor
        return np.diag(r_diag)

    # Convenience aliases matching test expectations
    compute_position_covariance = position_covariance
    compute_velocity_covariance = velocity_covariance
    compute_velocity_2d_covariance = velocity_2d_covariance


# Export standardized GNSS measurement dataclass and module helpers
from ..gnss.synthetic import GNSSMeasurementSample, GNSSMeasurementSample as GNSSMeasurement

_default_model = GNSSMeasurementModel()
predict_position = _default_model.predict_position
position_jacobian = _default_model.position_jacobian
predict_velocity = _default_model.predict_velocity
velocity_jacobian = _default_model.velocity_jacobian
predict_velocity_2d = _default_model.predict_velocity_2d
velocity_2d_jacobian = _default_model.velocity_2d_jacobian
velocity_2d_covariance = _default_model.velocity_2d_covariance

