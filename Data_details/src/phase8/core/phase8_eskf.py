"""
Phase 8 Core ESKF Architecture:
Extends Phase 6 ESKF with closed-loop GNSS position & velocity updates,
3-DOF & 2-DOF Chi-Square NIS gating on raw unclipped innovations,
covariance-consistent soft recovery scaling (R_eff = R / alpha),
and single-step physical bounds.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.models.imu_noise import IMUNoiseParameters
from Data_details.src.phase5.core.quaternion import quaternion_to_rotation_matrix
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF, Phase6NavigationEstimate
from Data_details.src.phase6.models.constraint_noise import ConstraintNoiseParameters

from ..gnss.gating import ChiSquareGating3DOF, ChiSquareGating2DOF, GatingDecision
from ..fusion.gnss_measurement import GNSSMeasurementModel, GNSSMeasurementConfig
from ..fusion.gnss_update import (
    evaluate_gnss_position_gate,
    apply_gnss_position_update,
    evaluate_gnss_velocity_2d_gate,
    apply_gnss_velocity_2d_update,
    apply_gnss_velocity_update,
)
from ..fusion.recovery import SmoothRecoveryManager


@dataclass
class Phase8StateEstimate(Phase6NavigationEstimate):
    """
    Extends Phase 6 Navigation Estimate with Phase 8 GNSS fusion and recovery telemetry.
    """
    gnss_pos_accepted: bool = False
    gnss_vel_accepted: bool = False
    gnss_pos_nis: float = 0.0
    gnss_vel_nis: float = 0.0
    gnss_p_step_m: float = 0.0
    recovery_alpha: float = 1.0


class Phase8ESKF(Phase6ESKF):
    """
    Phase 8 Unified GNSS + INS + Physics ESKF.
    """

    def __init__(
        self,
        noise_params: Optional[IMUNoiseParameters] = None,
        constraint_params: Optional[ConstraintNoiseParameters] = None,
        gnss_config: Optional[GNSSMeasurementConfig] = None,
        nis_speed_thresh: float = 9.0,
        nis_nhc_thresh: float = 9.21,
        nis_zupt_thresh: float = 11.34,
        nis_zaru_thresh: float = 11.34,
        nis_gnss_3dof_thresh: float = 11.345,
        nis_gnss_2dof_thresh: float = 9.210,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_zaru: bool = True,
        enable_gnss_pos: bool = True,
        enable_gnss_vel: bool = True,
    ):
        super().__init__(
            noise_params=noise_params,
            constraint_params=constraint_params,
            nis_speed_thresh=nis_speed_thresh,
            nis_nhc_thresh=nis_nhc_thresh,
            nis_zupt_thresh=nis_zupt_thresh,
            nis_zaru_thresh=nis_zaru_thresh,
            enable_nhc=enable_nhc,
            enable_zupt=enable_zupt,
            enable_zaru=enable_zaru,
        )

        self.gnss_config = gnss_config or GNSSMeasurementConfig()
        self.meas_model = GNSSMeasurementModel(self.gnss_config)
        self.gating_3dof = ChiSquareGating3DOF(threshold_accept=nis_gnss_3dof_thresh)
        self.gating_2dof = ChiSquareGating2DOF(threshold_accept=nis_gnss_2dof_thresh)
        self.recovery_manager = SmoothRecoveryManager()

        self.enable_gnss_pos = bool(enable_gnss_pos)
        self.enable_gnss_vel = bool(enable_gnss_vel)

        # GNSS Telemetry
        self.last_gnss_pos_accepted = False
        self.last_gnss_vel_accepted = False
        self.last_gnss_pos_nis = 0.0
        self.last_gnss_vel_nis = 0.0
        self.last_gnss_p_step = 0.0
        self.last_recovery_alpha = 1.0

    @property
    def gnss_model(self) -> GNSSMeasurementModel:
        return self.meas_model

    def evaluate_gnss_position_gate(
        self,
        z_pos_enu: np.ndarray,
        stated_acc_m: float,
        lever_arm_b: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """
        Evaluates 3D position gating without altering the filter state.
        Returns:
            is_accepted: True if NIS passes (decision != REJECT)
            nis: 3-DOF NIS value
        """
        if not self.initialized:
            return False, float("inf")
        R_p = self.meas_model.position_covariance(stated_acc_m)
        gate_res, _, _, _ = evaluate_gnss_position_gate(
            state=self.state,
            P=self.P,
            z_pos_enu=z_pos_enu,
            R_p=R_p,
            lever_arm_b=lever_arm_b if lever_arm_b is not None else self.gnss_config.lever_arm_b,
            gating_3dof=self.gating_3dof,
            meas_model=self.meas_model,
        )
        return (gate_res.decision != GatingDecision.REJECT), float(gate_res.nis)

    def update_gnss_position(
        self,
        z_pos_enu: np.ndarray,
        stated_acc_m: float,
        lever_arm_b: Optional[np.ndarray] = None,
        recovery_alpha: float = 1.0,
        max_step_clamp_m: float = 3.5,
    ) -> bool:
        """
        Executes 3D GNSS position update with 3-DOF Chi-square gating and soft recovery covariance inflation.
        """
        if not self.initialized or not self.enable_gnss_pos:
            self.last_gnss_pos_accepted = False
            return False

        R_p = self.meas_model.position_covariance(stated_acc_m)

        (
            self.state,
            self.P,
            nu,
            S,
            nis,
            accepted,
            actual_step,
        ) = apply_gnss_position_update(
            state=self.state,
            P=self.P,
            z_pos_enu=z_pos_enu,
            R_p=R_p,
            lever_arm_b=lever_arm_b if lever_arm_b is not None else self.gnss_config.lever_arm_b,
            recovery_alpha=recovery_alpha,
            max_step_clamp_m=max_step_clamp_m,
            gating_3dof=self.gating_3dof,
            meas_model=self.meas_model,
        )

        self.last_gnss_pos_accepted = accepted
        self.last_gnss_pos_nis = nis
        self.last_gnss_p_step = actual_step
        self.last_recovery_alpha = recovery_alpha
        return accepted

    def update_gnss_velocity_2d(
        self,
        z_vel_2d: np.ndarray,
        omega_b: np.ndarray,
        vel_acc_mps: Optional[float] = None,
        lever_arm_b: Optional[np.ndarray] = None,
        recovery_alpha: float = 1.0,
    ) -> bool:
        """
        Executes 2D horizontal GNSS velocity update with 2-DOF Chi-square gating and soft recovery scaling.
        Avoids fabricating unmeasured vertical velocity.
        """
        if not self.initialized or not self.enable_gnss_vel:
            self.last_gnss_vel_accepted = False
            return False

        R_v_2d = self.meas_model.velocity_2d_covariance(vel_acc_mps)

        (
            self.state,
            self.P,
            nu,
            S,
            nis,
            accepted,
        ) = apply_gnss_velocity_2d_update(
            state=self.state,
            P=self.P,
            z_vel_2d=z_vel_2d,
            R_v_2d=R_v_2d,
            omega_b=omega_b,
            lever_arm_b=lever_arm_b if lever_arm_b is not None else self.gnss_config.lever_arm_b,
            recovery_alpha=recovery_alpha,
            gating_2dof=self.gating_2dof,
            meas_model=self.meas_model,
        )

        self.last_gnss_vel_accepted = accepted
        self.last_gnss_vel_nis = nis
        return accepted

    def update_gnss_velocity(
        self,
        z_vel_enu: np.ndarray,
        omega_b: np.ndarray,
        vel_acc_mps: Optional[float] = None,
        lever_arm_b: Optional[np.ndarray] = None,
        recovery_alpha: float = 1.0,
    ) -> bool:
        """
        Executes 3D GNSS velocity update with 3-DOF Chi-square gating and soft recovery scaling.
        If vertical velocity is not measured, prefer update_gnss_velocity_2d.
        """
        if not self.initialized or not self.enable_gnss_vel:
            self.last_gnss_vel_accepted = False
            return False

        R_v = self.meas_model.velocity_covariance(vel_acc_mps)

        (
            self.state,
            self.P,
            nu,
            S,
            nis,
            accepted,
        ) = apply_gnss_velocity_update(
            state=self.state,
            P=self.P,
            z_vel_enu=z_vel_enu,
            R_v=R_v,
            omega_b=omega_b,
            lever_arm_b=lever_arm_b if lever_arm_b is not None else self.gnss_config.lever_arm_b,
            recovery_alpha=recovery_alpha,
            gating_3dof=self.gating_3dof,
            meas_model=self.meas_model,
        )

        self.last_gnss_vel_accepted = accepted
        self.last_gnss_vel_nis = nis
        return accepted

    def get_phase8_estimate(self, timestamp: float = 0.0) -> Phase8StateEstimate:
        """Returns complete Phase 8 state estimate snapshot."""
        p6_est = super().get_phase6_estimate(timestamp=timestamp)
        return Phase8StateEstimate(
            timestamp=p6_est.timestamp,
            p=p6_est.p,
            v=p6_est.v,
            q=p6_est.q,
            ba=p6_est.ba,
            bg=p6_est.bg,
            cov_diagonal=p6_est.cov_diagonal,
            v_b=p6_est.v_b,
            speed_innovation=p6_est.speed_innovation,
            speed_accepted=p6_est.speed_accepted,
            nhc_nu=p6_est.nhc_nu,
            nhc_accepted=p6_est.nhc_accepted,
            zupt_applied=p6_est.zupt_applied,
            zaru_applied=p6_est.zaru_applied,
            gnss_pos_accepted=self.last_gnss_pos_accepted,
            gnss_vel_accepted=self.last_gnss_vel_accepted,
            gnss_pos_nis=self.last_gnss_pos_nis,
            gnss_vel_nis=self.last_gnss_vel_nis,
            gnss_p_step_m=self.last_gnss_p_step,
            recovery_alpha=self.last_recovery_alpha,
        )


# Alias for nominal filter state
ESKFState = NominalState
