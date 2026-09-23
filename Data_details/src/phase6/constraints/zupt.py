"""
Phase 6 Zero-Velocity Update (ZUPT) Implementation:
Navigation Frame Velocity Clamping (v_n = 0) and Bias Observability Engine.
"""

from typing import Optional
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase6.measurement.robust_update import RobustMeasurementUpdater, UpdateResult


def predict_zupt(state: NominalState) -> np.ndarray:
    """Observation function h_ZUPT(x) = v_n (3D navigation velocity)."""
    return state.v.copy()


def compute_zupt_jacobian(state: NominalState) -> np.ndarray:
    """
    Analytically computes the 3x15 measurement Jacobian H_ZUPT = dh_ZUPT / d(delta_x).
    
    Structure:
        H = [ 0_{3x3},  I_3,  0_{3x3},  0_{3x3},  0_{3x3} ]
    """
    H = np.zeros((3, 15), dtype=np.float64)
    H[0:3, 3:6] = np.eye(3, dtype=np.float64)
    return H


def apply_zupt_update(
    state: NominalState,
    P: np.ndarray,
    R_zupt: Optional[np.ndarray] = None,
    sigma_zupt: float = 0.05,
    nis_threshold: Optional[float] = None,
) -> UpdateResult:
    """
    Applies 3D Zero-Velocity Update (v_n = 0) to clamp standstill drift and isolate accel bias.
    """
    if R_zupt is None:
        R_mat = np.diag([sigma_zupt**2, sigma_zupt**2, sigma_zupt**2]).astype(np.float64)
    else:
        R_mat = R_zupt

    h_zupt = predict_zupt(state)
    z_zupt = np.zeros(3, dtype=np.float64)
    H_zupt = compute_zupt_jacobian(state)

    return RobustMeasurementUpdater.update(
        nominal_state=state,
        P=P,
        z=z_zupt,
        h_val=h_zupt,
        H=H_zupt,
        R=R_mat,
        nis_threshold=nis_threshold,
    )
