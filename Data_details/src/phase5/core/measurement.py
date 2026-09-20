"""
Phase 5 AI Speed Measurement Model, Jacobian H, NIS Gating, & Joseph Update.
"""

from typing import Tuple, Dict, Any
import numpy as np
from Data_details.src.phase5.core.state import NominalState, ErrorState
from Data_details.src.phase5.core.quaternion import (
    quaternion_to_rotation_matrix,
    inject_small_angle_error,
)


def predict_forward_speed(state: NominalState) -> Tuple[float, np.ndarray]:
    """
    Computes nominal forward vehicle speed:
        v_b = R_{nb}^T * v_n = [v_fwd, v_lat, v_up]^T
        h(x) = v_fwd = v_b[0]
        
    Returns:
        v_fwd: Forward speed [m/s]
        v_b: Full 3D velocity vector in vehicle body frame [m/s]
    """
    R_nb = quaternion_to_rotation_matrix(state.q)
    v_b = R_nb.T @ state.v
    v_fwd = float(v_b[0])
    return v_fwd, v_b


def compute_speed_jacobian(state: NominalState) -> np.ndarray:
    """
    Analytically derives the 1x15 measurement Jacobian H = dh/d(delta_x).
    
    Structure:
        H = [ 0_{1x3},  e1^T * R_{nb}^T,  e1^T * [v_b]_x,  0_{1x3},  0_{1x3} ]
        where e1^T * [v_b]_x = [ 0,  -v_up,  v_lat ]
    """
    H = np.zeros((1, 15), dtype=np.float64)
    R_nb = quaternion_to_rotation_matrix(state.q)
    _, v_b = predict_forward_speed(state)

    # Velocity block: e1^T * R_{nb}^T = Row 0 of R_{nb}^T = Column 0 of R_{nb}
    H[0, 3:6] = R_nb[:, 0]

    # Attitude block: [0, -v_up, v_lat]
    v_lat = v_b[1]
    v_up = v_b[2]
    H[0, 6:9] = np.array([0.0, -v_up, v_lat], dtype=np.float64)

    return H


def kalman_update_speed(
    nominal_state: NominalState,
    P: np.ndarray,
    z_speed_mps: float,
    sigma_v_mps: float,
    nis_threshold: float = 9.0,
    min_variance: float = 1e-4,
) -> Tuple[NominalState, np.ndarray, float, float, float, bool]:
    """
    Executes an indirect Kalman update with forward speed measurement:
    1. Innovation: nu = z - h(x)
    2. Innovation covariance: S = H * P * H^T + R
    3. NIS gating test: nu^2 / S <= nis_threshold (Chi-square bound)
    4. Kalman Gain: K = P * H^T / S
    5. Joseph-form covariance update: P^+ = (I - K*H) * P * (I - K*H)^T + K * R * K^T
    6. Error injection into nominal state & error reset
    
    Returns:
        state_plus: Updated NominalState
        P_plus: Updated 15x15 Covariance Matrix
        innovation: Scalar innovation nu
        S: Innovation variance
        nis: Normalized Innovation Squared
        accepted: True if update was applied, False if rejected by NIS gate
    """
    v_pred, _ = predict_forward_speed(nominal_state)
    nu = float(z_speed_mps - v_pred)

    # Measurement noise variance R
    R = max(float(sigma_v_mps ** 2), min_variance)

    # Compute Jacobian H (1 x 15)
    H = compute_speed_jacobian(nominal_state)

    # Innovation covariance S (scalar)
    H_P = H @ P  # (1, 15)
    S = float((H_P @ H.T).item() + R)

    if S <= 0.0 or np.isnan(S):
        raise FloatingPointError(f"Singular or invalid innovation variance S = {S}")

    # NIS check
    nis = float((nu ** 2) / S)

    if nis > nis_threshold:
        # Reject outlier measurement: keep prior state and covariance
        return nominal_state.copy(), P.copy(), nu, S, nis, False

    # Kalman Gain K (15 x 1)
    K = (P @ H.T) / S  # (15, 1)

    # Error state estimate delta_x (15,)
    delta_x = (K * nu).ravel()

    # Joseph stabilized covariance update
    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - K @ H  # (15, 15)
    P_plus = I_KH @ P @ I_KH.T + (K * R) @ K.T

    # Enforce symmetry
    P_plus = 0.5 * (P_plus + P_plus.T)

    # Error state injection into nominal state
    dp = delta_x[0:3]
    dv = delta_x[3:6]
    dtheta = delta_x[6:9]
    dba = delta_x[9:12]
    dbg = delta_x[12:15]

    p_new = nominal_state.p + dp
    v_new = nominal_state.v + dv
    q_new = inject_small_angle_error(nominal_state.q, dtheta)
    ba_new = nominal_state.ba + dba
    bg_new = nominal_state.bg + dbg

    state_plus = NominalState(
        p=p_new,
        v=v_new,
        q=q_new,
        ba=ba_new,
        bg=bg_new,
    )

    return state_plus, P_plus, nu, S, nis, True
