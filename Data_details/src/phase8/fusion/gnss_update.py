"""
Phase 8 Joseph-Form Stabilized GNSS Position & Velocity Kalman Updates
Enforces strict Chi-Square gating, mathematically consistent soft recovery
covariance inflation (R_eff = R / alpha), and non-linear safety bounds (<= 3.5m)
to prevent unphysical teleportation shocks.
"""

from typing import Optional, Tuple
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import (
    quaternion_to_rotation_matrix,
    inject_small_angle_error,
)
from ..gnss.gating import ChiSquareGating3DOF, ChiSquareGating2DOF, GatingDecision, GatingResult
from .gnss_measurement import GNSSMeasurementModel, GNSSMeasurementConfig, skew_symmetric


def evaluate_gnss_position_gate(
    state: NominalState,
    P: np.ndarray,
    z_pos_enu: np.ndarray,
    R_p: np.ndarray,
    lever_arm_b: Optional[np.ndarray] = None,
    gating_3dof: Optional[ChiSquareGating3DOF] = None,
    meas_model: Optional[GNSSMeasurementModel] = None,
) -> Tuple[GatingResult, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates 3D GNSS position innovation and raw unclipped Chi-Square gate.
    Does not modify estimator state or covariance.

    Returns:
        gate_res: GatingResult (ACCEPT, DOWNWEIGHT, REJECT)
        nu: Raw unclipped innovation vector [3,]
        S: Innovation covariance [3, 3]
        H: Measurement Jacobian [3, 15]
    """
    model = meas_model or GNSSMeasurementModel()
    gater = gating_3dof or ChiSquareGating3DOF()

    p_pred = model.predict_position(state.p, state.q, lever_arm_b)
    nu = np.asarray(z_pos_enu, dtype=np.float64).ravel()[:3] - p_pred

    H = model.position_jacobian(state.q, lever_arm_b)  # (3, 15)

    H_P = H @ P
    S = (H_P @ H.T) + R_p
    S = 0.5 * (S + S.T)

    gate_res = gater.evaluate(nu, S)
    return gate_res, nu, S, H


def apply_gnss_position_update(
    state: NominalState,
    P: np.ndarray,
    z_pos_enu: np.ndarray,
    R_p: np.ndarray,
    lever_arm_b: Optional[np.ndarray] = None,
    recovery_alpha: float = 1.0,
    max_step_clamp_m: float = 3.5,
    gating_3dof: Optional[ChiSquareGating3DOF] = None,
    meas_model: Optional[GNSSMeasurementModel] = None,
) -> Tuple[NominalState, np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
    """
    Applies indirect 3D position Kalman update using Joseph-form covariance stabilization.
    Soft recovery is achieved through mathematically consistent covariance inflation:
        R_eff = R_p / alpha (0 < alpha <= 1)
    ensuring that the Kalman gain K and updated covariance P+ are completely consistent.

    Returns:
        state_plus: Updated NominalState
        P_plus: Updated 15x15 covariance matrix
        nu: Innovation vector [3,]
        S: Innovation covariance [3, 3]
        nis: 3-DOF Normalized Innovation Squared
        accepted: True if update was accepted and applied
        actual_step_m: Magnitude of position correction applied (m)
    """
    model = meas_model or GNSSMeasurementModel()
    gater = gating_3dof or ChiSquareGating3DOF()

    gate_res, nu, S_raw, H = evaluate_gnss_position_gate(
        state=state,
        P=P,
        z_pos_enu=z_pos_enu,
        R_p=R_p,
        lever_arm_b=lever_arm_b,
        gating_3dof=gater,
        meas_model=model,
    )

    if gate_res.decision == GatingDecision.REJECT:
        return state.copy(), P.copy(), nu, S_raw, gate_res.nis, False, 0.0

    # Mathematically consistent soft recovery: inflate measurement covariance
    alpha_clamped = max(0.01, min(1.0, float(recovery_alpha)))
    R_eff = R_p / alpha_clamped

    if gate_res.decision == GatingDecision.DOWNWEIGHT and gate_res.inflation_factor > 1.0:
        R_eff = R_eff * gate_res.inflation_factor

    # S_eff = H P H^T + R_eff
    H_P = H @ P
    S = (H_P @ H.T) + R_eff
    S = 0.5 * (S + S.T)

    # Kalman gain K = P H^T S^{-1} (15 x 3)
    try:
        K_T = np.linalg.solve(S, H_P)  # (3, 15)
        K = K_T.T                      # (15, 3)
    except np.linalg.LinAlgError:
        return state.copy(), P.copy(), nu, S, gate_res.nis, False, 0.0

    # Correction delta_x = K @ nu (consistent with R_eff)
    delta_x = K @ nu  # (15,)

    # Non-linear safety constraint: bound single-step physical position correction
    dp = delta_x[0:3].copy()
    step_norm = float(np.linalg.norm(dp))
    actual_step = step_norm

    if step_norm > max_step_clamp_m and step_norm > 1e-6:
        dp = dp * (max_step_clamp_m / step_norm)
        delta_x[0:3] = dp
        actual_step = max_step_clamp_m

    # Joseph stabilized covariance update using the exact effective observation covariance
    # P+ = (I - K H) P (I - K H)^T + K R_eff K^T
    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - (K @ H)
    P_plus = (I_KH @ P @ I_KH.T) + (K @ R_eff @ K.T)
    P_plus = 0.5 * (P_plus + P_plus.T)

    # Floor diagonal to prevent degenerate covariance
    for i in range(15):
        if P_plus[i, i] < 1e-12:
            P_plus[i, i] = 1e-12

    # Inject error state into nominal state
    dv = delta_x[3:6]
    dtheta = delta_x[6:9]
    dba = delta_x[9:12]
    dbg = delta_x[12:15]

    state_plus = NominalState(
        p=state.p + dp,
        v=state.v + dv,
        q=inject_small_angle_error(state.q, dtheta),
        ba=state.ba + dba,
        bg=state.bg + dbg,
    )

    return state_plus, P_plus, nu, S, gate_res.nis, True, actual_step


def evaluate_gnss_velocity_2d_gate(
    state: NominalState,
    P: np.ndarray,
    z_vel_2d: np.ndarray,
    R_v_2d: np.ndarray,
    omega_b: np.ndarray,
    lever_arm_b: Optional[np.ndarray] = None,
    gating_2dof: Optional[ChiSquareGating2DOF] = None,
    meas_model: Optional[GNSSMeasurementModel] = None,
) -> Tuple[GatingResult, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates 2D horizontal GNSS velocity innovation and 2-DOF Chi-Square gate.
    """
    model = meas_model or GNSSMeasurementModel()
    gater = gating_2dof or ChiSquareGating2DOF()

    v_pred_2d = model.predict_velocity_2d(state.v, state.q, omega_b, lever_arm_b)
    nu = np.asarray(z_vel_2d, dtype=np.float64).ravel()[:2] - v_pred_2d

    H = model.velocity_2d_jacobian(state.q, omega_b, lever_arm_b)  # (2, 15)

    H_P = H @ P
    S = (H_P @ H.T) + R_v_2d
    S = 0.5 * (S + S.T)

    gate_res = gater.evaluate(nu, S)
    return gate_res, nu, S, H


def apply_gnss_velocity_2d_update(
    state: NominalState,
    P: np.ndarray,
    z_vel_2d: np.ndarray,
    R_v_2d: np.ndarray,
    omega_b: np.ndarray,
    lever_arm_b: Optional[np.ndarray] = None,
    recovery_alpha: float = 1.0,
    max_vel_clamp_mps: float = 2.0,
    gating_2dof: Optional[ChiSquareGating2DOF] = None,
    meas_model: Optional[GNSSMeasurementModel] = None,
) -> Tuple[NominalState, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Applies indirect 2D horizontal velocity Kalman update using Joseph-form stabilization.
    Eliminates vertical velocity fabrication when only horizontal speed + course are available.
    """
    model = meas_model or GNSSMeasurementModel()
    gater = gating_2dof or ChiSquareGating2DOF()

    gate_res, nu, S_raw, H = evaluate_gnss_velocity_2d_gate(
        state=state,
        P=P,
        z_vel_2d=z_vel_2d,
        R_v_2d=R_v_2d,
        omega_b=omega_b,
        lever_arm_b=lever_arm_b,
        gating_2dof=gater,
        meas_model=model,
    )

    if gate_res.decision == GatingDecision.REJECT:
        return state.copy(), P.copy(), nu, S_raw, gate_res.nis, False

    alpha_clamped = max(0.01, min(1.0, float(recovery_alpha)))
    R_eff = R_v_2d / alpha_clamped

    if gate_res.decision == GatingDecision.DOWNWEIGHT and gate_res.inflation_factor > 1.0:
        R_eff = R_eff * gate_res.inflation_factor

    H_P = H @ P
    S = (H_P @ H.T) + R_eff
    S = 0.5 * (S + S.T)

    try:
        K_T = np.linalg.solve(S, H_P)
        K = K_T.T  # (15, 2)
    except np.linalg.LinAlgError:
        return state.copy(), P.copy(), nu, S, gate_res.nis, False

    delta_x = K @ nu

    # Nonlinear velocity safety clamp
    dv = delta_x[3:6].copy()
    v_step = float(np.linalg.norm(dv))
    if v_step > max_vel_clamp_mps and v_step > 1e-6:
        dv = dv * (max_vel_clamp_mps / v_step)
        delta_x[3:6] = dv

    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - (K @ H)
    P_plus = (I_KH @ P @ I_KH.T) + (K @ R_eff @ K.T)
    P_plus = 0.5 * (P_plus + P_plus.T)

    for i in range(15):
        if P_plus[i, i] < 1e-12:
            P_plus[i, i] = 1e-12

    dp = delta_x[0:3]
    dtheta = delta_x[6:9]
    dba = delta_x[9:12]
    dbg = delta_x[12:15]

    state_plus = NominalState(
        p=state.p + dp,
        v=state.v + dv,
        q=inject_small_angle_error(state.q, dtheta),
        ba=state.ba + dba,
        bg=state.bg + dbg,
    )

    return state_plus, P_plus, nu, S, gate_res.nis, True


def apply_gnss_velocity_update(
    state: NominalState,
    P: np.ndarray,
    z_vel_enu: np.ndarray,
    R_v: np.ndarray,
    omega_b: np.ndarray,
    lever_arm_b: Optional[np.ndarray] = None,
    recovery_alpha: float = 1.0,
    max_vel_clamp_mps: float = 2.0,
    gating_3dof: Optional[ChiSquareGating3DOF] = None,
    meas_model: Optional[GNSSMeasurementModel] = None,
) -> Tuple[NominalState, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Applies indirect 3D velocity Kalman update with mathematically consistent soft recovery covariance.
    Provided for 3D Doppler receivers.
    """
    model = meas_model or GNSSMeasurementModel()
    gater = gating_3dof or ChiSquareGating3DOF()

    v_pred = model.predict_velocity(state.v, state.q, omega_b, lever_arm_b)
    nu = np.asarray(z_vel_enu, dtype=np.float64).ravel()[:3] - v_pred

    H = model.velocity_jacobian(state.q, omega_b, lever_arm_b)  # (3, 15)

    H_P = H @ P
    S = (H_P @ H.T) + R_v
    S = 0.5 * (S + S.T)

    gate_res = gater.evaluate(nu, S)

    if gate_res.decision == GatingDecision.REJECT:
        return state.copy(), P.copy(), nu, S, gate_res.nis, False

    alpha_clamped = max(0.01, min(1.0, float(recovery_alpha)))
    R_eff = R_v / alpha_clamped

    if gate_res.decision == GatingDecision.DOWNWEIGHT and gate_res.inflation_factor > 1.0:
        R_eff = R_eff * gate_res.inflation_factor
        S = (H_P @ H.T) + R_eff
        S = 0.5 * (S + S.T)

    try:
        K_T = np.linalg.solve(S, H_P)
        K = K_T.T
    except np.linalg.LinAlgError:
        return state.copy(), P.copy(), nu, S, gate_res.nis, False

    delta_x = K @ nu

    dv = delta_x[3:6].copy()
    v_step = float(np.linalg.norm(dv))
    if v_step > max_vel_clamp_mps and v_step > 1e-6:
        dv = dv * (max_vel_clamp_mps / v_step)
        delta_x[3:6] = dv

    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - (K @ H)
    P_plus = (I_KH @ P @ I_KH.T) + (K @ R_eff @ K.T)
    P_plus = 0.5 * (P_plus + P_plus.T)

    for i in range(15):
        if P_plus[i, i] < 1e-12:
            P_plus[i, i] = 1e-12

    dp = delta_x[0:3]
    dtheta = delta_x[6:9]
    dba = delta_x[9:12]
    dbg = delta_x[12:15]

    state_plus = NominalState(
        p=state.p + dp,
        v=state.v + dv,
        q=inject_small_angle_error(state.q, dtheta),
        ba=state.ba + dba,
        bg=state.bg + dbg,
    )

    return state_plus, P_plus, nu, S, gate_res.nis, True
