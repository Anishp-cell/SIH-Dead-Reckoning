"""
Phase 6 Zero-Angular-Rate Update (ZARU) Implementation:
Stationary Gyroscope Bias Isolation (omega_m = b_g) Without Artificially Resetting Attitude.
"""

from typing import Optional
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase6.measurement.robust_update import RobustMeasurementUpdater, UpdateResult


def predict_zaru(state: NominalState) -> np.ndarray:
    """Observation function h_ZARU(x) = b_g (estimated gyroscope bias)."""
    return state.bg.copy()


def compute_zaru_jacobian(state: NominalState) -> np.ndarray:
    """
    Analytically computes the 3x15 measurement Jacobian H_ZARU = dh_ZARU / d(delta_x).
    
    Structure:
        H = [ 0_{3x3},  0_{3x3},  0_{3x3},  0_{3x3},  I_3 ]
    """
    H = np.zeros((3, 15), dtype=np.float64)
    H[0:3, 12:15] = np.eye(3, dtype=np.float64)
    return H


def apply_zaru_update(
    state: NominalState,
    P: np.ndarray,
    omega_meas: np.ndarray,
    R_zaru: Optional[np.ndarray] = None,
    sigma_zaru: float = 0.005,
    nis_threshold: Optional[float] = None,
) -> UpdateResult:
    """
    Applies 3D Zero-Angular-Rate Update during verified standstill.
    Directly observes and updates gyroscope bias state delta_bg without artificially resetting yaw.
    """
    if R_zaru is None:
        R_mat = np.diag([sigma_zaru**2, sigma_zaru**2, sigma_zaru**2]).astype(np.float64)
    else:
        R_mat = R_zaru

    h_zaru = predict_zaru(state)
    z_zaru = np.asarray(omega_meas, dtype=np.float64).ravel()
    H_zaru = compute_zaru_jacobian(state)

    return RobustMeasurementUpdater.update(
        nominal_state=state,
        P=P,
        z=z_zaru,
        h_val=h_zaru,
        H=H_zaru,
        R=R_mat,
        nis_threshold=nis_threshold,
    )
