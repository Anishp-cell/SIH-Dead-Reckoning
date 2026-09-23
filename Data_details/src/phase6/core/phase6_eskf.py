"""
Phase 6 15-State Error-State Kalman Filter with Vehicle Physics & Stationary Constraints.
Extends Phase 5 ESKF with Non-Holonomic Constraints (NHC), ZUPT, ZARU, and Adaptive Disturbance Weighting.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from Data_details.src.phase5.core.state import NominalState, NavigationEstimate
from Data_details.src.phase5.core.eskf import ESKF15State
from Data_details.src.phase5.core.quaternion import quaternion_to_rotation_matrix
from Data_details.src.phase5.models.imu_noise import IMUNoiseParameters
from Data_details.src.phase6.constraints.nhc import predict_nhc_velocity, apply_nhc_update
from Data_details.src.phase6.constraints.zupt import apply_zupt_update
from Data_details.src.phase6.constraints.zaru import apply_zaru_update
from Data_details.src.phase6.models.constraint_noise import ConstraintNoiseParameters
from Data_details.src.phase6.detection.stationary_detector import RobustStationaryDetector
from Data_details.src.phase6.detection.disturbance_detector import DisturbanceDetector, DisturbanceReport


@dataclass
class Phase6NavigationEstimate:
    """Comprehensive Phase 6 Navigation Estimate Output."""
    timestamp: float
    p: np.ndarray                 # (3,) Position in ENU (m)
    v: np.ndarray                 # (3,) Velocity in ENU (m/s)
    q: np.ndarray                 # (4,) Quaternion [qw, qx, qy, qz]
    ba: np.ndarray                # (3,) Accel bias (m/s^2)
    bg: np.ndarray                # (3,) Gyro bias (rad/s)
    cov_diagonal: np.ndarray      # (15,) Diagonal elements of P
    v_b: np.ndarray               # (3,) Full body velocity [v_fwd, v_lat, v_up] (m/s)
    speed_innovation: float       # AI Speed residual (m/s)
    speed_accepted: bool          # Whether AI speed update was accepted
    nhc_nu: np.ndarray            # NHC residual [nu_lat, nu_up] (m/s)
    nhc_accepted: bool            # Whether NHC update was applied
    nhc_confidence: float         # Causal disturbance confidence score [0.0, 1.0]
    is_stationary: bool           # Filtered stationary state
    stationary_duration_s: float  # Standstill duration in seconds
    zupt_accepted: bool           # Whether ZUPT was applied
    zaru_accepted: bool           # Whether ZARU was applied
    filter_healthy: bool          # Finite covariance & validity


class Phase6ESKF(ESKF15State):
    """
    Phase 6 Unified Vehicle-Physics Augmented ESKF.
    """

    def __init__(
        self,
        noise_params: Optional[IMUNoiseParameters] = None,
        constraint_params: Optional[ConstraintNoiseParameters] = None,
        nis_speed_thresh: float = 9.0,
        nis_nhc_thresh: float = 9.21,
        nis_zupt_thresh: float = 11.34,
        nis_zaru_thresh: float = 11.34,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_zaru: bool = True,
        enable_disturbance_adaptive: bool = True,
    ):
        super().__init__(noise_params=noise_params, nis_threshold=nis_speed_thresh)
        self.constraint_params = constraint_params or ConstraintNoiseParameters()

        self.nis_speed_thresh = nis_speed_thresh
        self.nis_nhc_thresh = nis_nhc_thresh
        self.nis_zupt_thresh = nis_zupt_thresh
        self.nis_zaru_thresh = nis_zaru_thresh

        self.enable_nhc = enable_nhc
        self.enable_zupt = enable_zupt
        self.enable_zaru = enable_zaru
        self.enable_disturbance_adaptive = enable_disturbance_adaptive

        # Detectors
        self.stationary_detector = RobustStationaryDetector()
        self.disturbance_detector = DisturbanceDetector()

        # Telemetry
        self.last_nhc_nu = np.zeros(2)
        self.last_nhc_accepted = False
        self.last_nhc_conf = 1.0
        self.last_is_stationary = False
        self.last_stat_duration = 0.0
        self.last_zupt_accepted = False
        self.last_zaru_accepted = False

    def update_nhc(self, d_lat: float = 0.0, d_up: float = 0.0) -> bool:
        """Applies Non-Holonomic Constraint (v_lat = 0, v_up = 0) with adaptive covariance."""
        if not self.enable_nhc:
            self.last_nhc_accepted = False
            return False

        if self.enable_disturbance_adaptive:
            R_nhc = self.constraint_params.build_nhc_R(d_lat=d_lat, d_up=d_up)
        else:
            R_nhc = self.constraint_params.build_nhc_R(d_lat=0.0, d_up=0.0)

        res = apply_nhc_update(
            state=self.state,
            P=self.P,
            R_nhc=R_nhc,
            nis_threshold=self.nis_nhc_thresh,
        )

        self.state = res.state_plus
        self.P = res.P_plus
        self.last_nhc_nu = res.nu
        self.last_nhc_accepted = res.accepted
        return res.accepted

    def update_stationary_constraints(self, omega_meas: np.ndarray) -> Tuple[bool, bool]:
        """Applies ZUPT and ZARU updates during confirmed standstill."""
        zupt_ok = False
        zaru_ok = False

        if self.enable_zupt:
            R_zupt = self.constraint_params.build_zupt_R()
            res_zupt = apply_zupt_update(
                state=self.state,
                P=self.P,
                R_zupt=R_zupt,
                nis_threshold=self.nis_zupt_thresh,
            )
            self.state = res_zupt.state_plus
            self.P = res_zupt.P_plus
            zupt_ok = res_zupt.accepted

        if self.enable_zaru:
            R_zaru = self.constraint_params.build_zaru_R()
            res_zaru = apply_zaru_update(
                state=self.state,
                P=self.P,
                omega_meas=omega_meas,
                R_zaru=R_zaru,
                nis_threshold=self.nis_zaru_thresh,
            )
            self.state = res_zaru.state_plus
            self.P = res_zaru.P_plus
            zaru_ok = res_zaru.accepted

        self.last_zupt_accepted = zupt_ok
        self.last_zaru_accepted = zaru_ok
        return zupt_ok, zaru_ok

    def get_phase6_estimate(self, timestamp: float = 0.0) -> Phase6NavigationEstimate:
        """Returns structured Phase 6 navigation estimate."""
        R_nb = quaternion_to_rotation_matrix(self.state.q)
        v_b = R_nb.T @ self.state.v

        cov_diag = np.diag(self.P).copy()
        filter_healthy = bool(np.all(np.isfinite(cov_diag)) and np.all(cov_diag > 0))

        return Phase6NavigationEstimate(
            timestamp=timestamp,
            p=self.state.p.copy(),
            v=self.state.v.copy(),
            q=self.state.q.copy(),
            ba=self.state.ba.copy(),
            bg=self.state.bg.copy(),
            cov_diagonal=cov_diag,
            v_b=v_b,
            speed_innovation=self.last_nu,
            speed_accepted=self.last_accepted,
            nhc_nu=self.last_nhc_nu.copy(),
            nhc_accepted=self.last_nhc_accepted,
            nhc_confidence=self.last_nhc_conf,
            is_stationary=self.last_is_stationary,
            stationary_duration_s=self.last_stat_duration,
            zupt_accepted=self.last_zupt_accepted,
            zaru_accepted=self.last_zaru_accepted,
            filter_healthy=filter_healthy,
        )
