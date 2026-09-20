"""
Phase 5 Inertial Navigation System (INS) Propagation & 15x15 Covariance Propagation.
"""

import numpy as np
from typing import Tuple
from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import (
    quaternion_to_rotation_matrix,
    quaternion_integrate,
    skew_symmetric,
)
from Data_details.src.phase5.core.frames import GRAVITY_VECTOR_ENU


def propagate_nominal_ins(
    state: NominalState,
    f_meas: np.ndarray,
    omega_meas: np.ndarray,
    dt: float,
) -> Tuple[NominalState, np.ndarray, np.ndarray]:
    """
    Executes one discrete nominal INS propagation step over interval dt:
        p_{k+1} = p_k + v_k * dt + 0.5 * a_n * dt^2
        v_{k+1} = v_k + a_n * dt
        q_{k+1} = q_k ⊗ dq(omega_corr * dt)
        ba_{k+1} = ba_k
        bg_{k+1} = bg_k
        
    Returns:
        next_state: Updated NominalState
        f_b_corr: Bias-corrected specific force [m/s^2]
        omega_b_corr: Bias-corrected angular velocity [rad/s]
    """
    f_b_corr = np.asarray(f_meas, dtype=np.float64) - state.ba
    omega_b_corr = np.asarray(omega_meas, dtype=np.float64) - state.bg

    # Rotate specific force to navigation frame
    R_nb = quaternion_to_rotation_matrix(state.q)
    f_n = R_nb @ f_b_corr

    # Total navigation acceleration including gravity
    a_n = f_n + GRAVITY_VECTOR_ENU

    # 2nd-order kinematic position and velocity update
    p_next = state.p + state.v * dt + 0.5 * a_n * (dt ** 2)
    v_next = state.v + a_n * dt

    # Exact quaternion integration
    q_next = quaternion_integrate(state.q, omega_b_corr, dt)

    next_state = NominalState(
        p=p_next,
        v=v_next,
        q=q_next,
        ba=state.ba.copy(),
        bg=state.bg.copy(),
    )
    return next_state, f_b_corr, omega_b_corr


def compute_discrete_F_matrix(
    state: NominalState,
    f_b_corr: np.ndarray,
    omega_b_corr: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Constructs the 15x15 discrete state transition Jacobian matrix F_k.
    
    Structure:
        F = [
            I_3,   I_3 * dt,  -0.5 * R_nb * [f_b]_x * dt^2,  -0.5 * R_nb * dt^2,   0_3
            0_3,   I_3,       -R_nb * [f_b]_x * dt,         -R_nb * dt,          0_3
            0_3,   0_3,       I_3 - [omega_b]_x * dt,        0_3,                -I_3 * dt
            0_3,   0_3,       0_3,                          I_3,                 0_3
            0_3,   0_3,       0_3,                          0_3,                 I_3
        ]
    """
    F = np.eye(15, dtype=np.float64)
    R_nb = quaternion_to_rotation_matrix(state.q)
    f_skew = skew_symmetric(f_b_corr)
    omega_skew = skew_symmetric(omega_b_corr)

    dt2 = dt ** 2

    # Row 0: delta_p
    F[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
    F[0:3, 6:9] = -0.5 * (R_nb @ f_skew) * dt2
    F[0:3, 9:12] = -0.5 * R_nb * dt2

    # Row 1: delta_v
    F[3:6, 6:9] = -(R_nb @ f_skew) * dt
    F[3:6, 9:12] = -R_nb * dt

    # Row 2: delta_theta (including 2nd-order Taylor terms for exact dt match)
    F[6:9, 6:9] = np.eye(3, dtype=np.float64) - omega_skew * dt + 0.5 * (omega_skew @ omega_skew) * dt2
    F[6:9, 12:15] = -np.eye(3, dtype=np.float64) * dt + 0.5 * omega_skew * dt2

    # Rows 3 & 4 (biases) are identity (already set by np.eye)
    return F


def propagate_covariance(
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    min_diag: float = 1e-12,
) -> np.ndarray:
    """
    Propagates 15x15 covariance forward: P_{k+1}^- = F * P_k * F^T + Q_k
    Enforces numerical symmetry and non-negative diagonal floor.
    """
    P_next = F @ P @ F.T + Q

    # Enforce exact mathematical symmetry
    P_next = 0.5 * (P_next + P_next.T)

    # Diagonal conditioning check
    diags = np.diag(P_next)
    if np.any(np.isnan(diags)) or np.any(np.isinf(diags)):
        raise FloatingPointError("Covariance matrix exploded into NaN or Inf during propagation.")

    # Floor negative diagonals
    if np.any(diags < min_diag):
        np.fill_diagonal(P_next, np.maximum(diags, min_diag))

    return P_next
