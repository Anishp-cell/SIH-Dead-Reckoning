"""
Phase 7 Map Measurements & Analytical Jacobians:
Derives exact analytical observation models and 15-state ESKF Jacobians for:
1. Signed cross-track road constraint:
   z_p = 0, h_p(x) = n^T (p - a) = d_cross
   H_p = [n_E, n_N, 0, 0_{1x3}, 0_{1x3}, 0_{1x3}, 0_{1x3}] (1x15)
2. Road tangent heading constraint:
   nu_psi = wrap(psi_road - psi_veh)
   H_psi = [0_{1x3}, 0_{1x3}, 0, 0, 1, 0_{1x3}, 0_{1x3}] (1x15)
Includes adaptive uncertainty-scaled measurement covariance and Chi-square innovation gating.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.stats import chi2

from .temporal_matcher import MapMatchResult
from ..map.geometry import wrap_angle_rad


@dataclass
class MapMeasurementConfig:
    """
    Configuration parameters for map constraint updates.
    """
    sigma_cross_track_base_m: float = 2.0         # Baseline cross-track std (half-lane width + GPS noise floor)
    sigma_heading_base_rad: float = np.radians(4.0) # Baseline road heading std (curved segment / lane divergence)
    nis_chi2_alpha: float = 0.01                  # Significance level (99% acceptance bound)
    min_velocity_for_heading_mps: float = 1.5     # Minimum vehicle speed to apply heading constraint
    max_cross_track_innovation_m: float = 15.0    # Hard innovation clamp for robustness
    max_heading_innovation_rad: float = np.radians(35.0)


class MapMeasurementModel:
    """
    Computes innovations, analytical Jacobians, adaptive covariances, and NIS gating.
    """

    def __init__(self, config: Optional[MapMeasurementConfig] = None):
        self.cfg = config or MapMeasurementConfig()
        # 1-DOF Chi-square threshold at 1 - alpha (e.g. 0.99 -> 6.635)
        self.chi2_threshold_1d = float(chi2.ppf(1.0 - self.cfg.nis_chi2_alpha, df=1))

    def compute_cross_track_update(
        self,
        match: MapMatchResult,
        P_15x15: np.ndarray,
    ) -> Tuple[float, np.ndarray, float, float, bool]:
        """
        Computes signed cross-track innovation, 1x15 Jacobian, innovation variance S, NIS, and gating decision.

        Returns:
            (nu, H, R, nis, accepted)
        """
        # Cross-track innovation: we want d_cross -> 0
        # h(x) = n^T (p - a) = d_cross
        # nu = z - h(x) = 0 - d_cross = -d_cross
        nu = -float(match.cross_track_error)

        # Innovation clamp to prevent extreme filter shocks
        nu_clamped = float(np.clip(nu, -self.cfg.max_cross_track_innovation_m, self.cfg.max_cross_track_innovation_m))

        # Jacobian H: normal vector in ENU plane
        # Left-pointing normal: n = [-sin(psi_road), cos(psi_road)]
        psi = match.road_heading_enu
        n_E = -np.sin(psi)
        n_N = np.cos(psi)

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 0] = n_E
        H[0, 1] = n_N
        # State components: p(0:3), v(3:6), theta(6:9), ba(9:12), bg(12:15)

        # Adaptive measurement variance scaled inversely by map confidence:
        # R = sigma_base^2 / c_map^2 (as c_map -> 0, R -> infinity)
        c_map = max(1e-4, match.map_confidence)
        R = (self.cfg.sigma_cross_track_base_m / c_map)**2

        # Innovation covariance: S = H P H^T + R
        S = float((H @ P_15x15 @ H.T).item() + R)

        # Normalized Innovation Squared (NIS): computed on RAW unclipped innovation to reject true outliers
        nis = float((nu**2) / S)

        # Innovation gating
        accepted = bool(nis <= self.chi2_threshold_1d and match.accepted)

        # Innovation clamp to prevent extreme filter shocks if accepted
        nu_clamped = float(np.clip(nu, -self.cfg.max_cross_track_innovation_m, self.cfg.max_cross_track_innovation_m))

        return nu_clamped, H, R, nis, accepted

    def compute_heading_update(
        self,
        match: MapMatchResult,
        P_15x15: np.ndarray,
        v_forward_mps: float,
        R_nb: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, float, float, bool]:
        """
        Computes road tangent heading innovation, 1x15 Jacobian, innovation variance S, NIS, and gating decision.

        Returns:
            (nu, H, R, nis, accepted)
        """
        # Road heading constraint is only valid when moving forward
        if abs(v_forward_mps) < self.cfg.min_velocity_for_heading_mps or not match.accepted:
            return 0.0, np.zeros((1, 15)), 1e6, 0.0, False

        # Heading innovation: nu = wrap(psi_road - psi_veh)
        # Note: match.heading_residual is wrap(psi_veh - psi_road), so innovation is -residual
        nu = -float(match.heading_residual)

        # Analytical Jacobian H:
        # State error theta is body-frame rotation perturbation [delta_theta_x, delta_theta_y, delta_theta_z]
        # In ENU frame, psi = atan2(R10, R00).
        # Perturbation yields:
        # d_psi = [(-R00*R12 + R10*R02)/(R00^2 + R10^2)] * delta_theta_y + [(R00*R11 - R10*R01)/(R00^2 + R10^2)] * delta_theta_z
        H = np.zeros((1, 15), dtype=np.float64)
        if R_nb is not None:
            R00 = float(R_nb[0, 0])
            R10 = float(R_nb[1, 0])
            R01 = float(R_nb[0, 1])
            R11 = float(R_nb[1, 1])
            R02 = float(R_nb[0, 2])
            R12 = float(R_nb[1, 2])
            denom = R00**2 + R10**2
            if denom > 1e-8:
                H[0, 7] = (-R00 * R12 + R10 * R02) / denom
                H[0, 8] = (R00 * R11 - R10 * R01) / denom
            else:
                H[0, 8] = 1.0
        else:
            H[0, 8] = 1.0  # Level-vehicle assumption fallback

        c_map = max(1e-4, match.map_confidence)
        R = (self.cfg.sigma_heading_base_rad / c_map)**2

        S = float((H @ P_15x15 @ H.T).item() + R)
        nis = float((nu**2) / S)

        accepted = bool(nis <= self.chi2_threshold_1d and match.accepted)
        nu_clamped = float(np.clip(nu, -self.cfg.max_heading_innovation_rad, self.cfg.max_heading_innovation_rad))

        return nu_clamped, H, R, nis, accepted

    def compute_along_track_update(
        self,
        match: MapMatchResult,
        P_15x15: np.ndarray,
        target_along_m: float,
        variance_along_m2: float = 1.0,
    ) -> Tuple[float, np.ndarray, float, float, bool]:
        """
        Computes signed along-track road spline innovation, 1x15 Jacobian,
        innovation variance S, NIS, and gating decision.

        Along-track innovation: nu = target_along_m - current_along_m
        Jacobian H: tangent vector in ENU plane: [t_E, t_N, 0, 0_{1x12}]
        """
        if not match.accepted or match.matched_segment_id is None:
            return 0.0, np.zeros((1, 15)), 1e6, 0.0, False

        current_along_m = float(match.along_track_position)
        nu = float(target_along_m - current_along_m)

        psi = match.road_heading_enu
        t_E = float(np.cos(psi))
        t_N = float(np.sin(psi))

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 0] = t_E
        H[0, 1] = t_N

        c_map = max(1e-4, match.map_confidence)
        R = max(0.25, float(variance_along_m2) / (c_map**2))

        S = float((H @ P_15x15 @ H.T).item() + R)
        nis = float((nu**2) / S)

        accepted = bool(nis <= self.chi2_threshold_1d)
        nu_clamped = float(np.clip(nu, -10.0, 10.0))

        return nu_clamped, H, R, nis, accepted

