"""
Phase 7 Map Update Engine:
Executes soft indirect Kalman updates for map-derived constraints:
1. Cross-track position constraint
2. Road tangent heading constraint
Applies Joseph-form covariance stabilization, Chi-square NIS gating, and small-angle error injection.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import inject_small_angle_error, quaternion_to_rotation_matrix
from .temporal_matcher import MapMatchResult
from .map_measurement import MapMeasurementModel


def apply_map_cross_track_update(
    state: NominalState,
    P: np.ndarray,
    match: MapMatchResult,
    model: MapMeasurementModel,
) -> Tuple[NominalState, np.ndarray, float, float, float, bool]:
    """
    Applies scalar cross-track road constraint update:
    z_p = 0, h(x) = d_cross -> nu = -d_cross.
    """
    nu, H, R, nis, accepted = model.compute_cross_track_update(match, P)

    if not accepted:
        # Gated or unconfident: keep prior state and covariance
        H_P = H @ P
        S = float((H_P @ H.T).item() + R)
        return state.copy(), P.copy(), nu, S, nis, False

    # Innovation covariance
    H_P = H @ P  # (1, 15)
    S = float((H_P @ H.T).item() + R)

    # Kalman Gain K (15 x 1)
    K = (P @ H.T) / S

    # Indirect error state delta_x (15,)
    delta_x = (K * nu).ravel()

    # Joseph stabilized covariance update
    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - K @ H
    P_plus = I_KH @ P @ I_KH.T + (K * R) @ K.T
    P_plus = 0.5 * (P_plus + P_plus.T)

    # Error injection
    p_new = state.p + delta_x[0:3]
    v_new = state.v + delta_x[3:6]
    q_new = inject_small_angle_error(state.q, delta_x[6:9])
    ba_new = state.ba + delta_x[9:12]
    bg_new = state.bg + delta_x[12:15]

    state_plus = NominalState(p=p_new, v=v_new, q=q_new, ba=ba_new, bg=bg_new)
    return state_plus, P_plus, nu, S, nis, True


def apply_map_heading_update(
    state: NominalState,
    P: np.ndarray,
    match: MapMatchResult,
    model: MapMeasurementModel,
    v_forward_mps: float,
) -> Tuple[NominalState, np.ndarray, float, float, float, bool]:
    """
    Applies scalar road tangent heading constraint update:
    z_psi = psi_road, h(x) = psi_veh -> nu = wrap(psi_road - psi_veh).
    """
    R_nb = quaternion_to_rotation_matrix(state.q)
    nu, H, R, nis, accepted = model.compute_heading_update(match, P, v_forward_mps, R_nb=R_nb)

    if not accepted:
        H_P = H @ P
        S = float((H_P @ H.T).item() + R)
        return state.copy(), P.copy(), nu, S, nis, False

    H_P = H @ P
    S = float((H_P @ H.T).item() + R)

    K = (P @ H.T) / S
    delta_x = (K * nu).ravel()

    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - K @ H
    P_plus = I_KH @ P @ I_KH.T + (K * R) @ K.T
    P_plus = 0.5 * (P_plus + P_plus.T)

    p_new = state.p + delta_x[0:3]
    v_new = state.v + delta_x[3:6]
    q_new = inject_small_angle_error(state.q, delta_x[6:9])
    ba_new = state.ba + delta_x[9:12]
    bg_new = state.bg + delta_x[12:15]

    state_plus = NominalState(p=p_new, v=v_new, q=q_new, ba=ba_new, bg=bg_new)
    return state_plus, P_plus, nu, S, nis, True
