"""
Phase 6 Non-Holonomic Constraints (NHC) Implementation:
Lateral (v_y^b = 0) and Vertical (v_z^b = 0) Vehicle Body Velocity Updates.
"""

from typing import Tuple, Optional
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import quaternion_to_rotation_matrix
from Data_details.src.phase6.measurement.robust_update import RobustMeasurementUpdater, UpdateResult


def predict_nhc_velocity(state: NominalState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes body-frame velocity v_b = R_{nb}^T * v_n = [v_fwd, v_lat, v_up]^T.
    
    Returns:
        h_nhc: (2,) array [v_lat, v_up]^T (Observation function h(x))
        v_b: (3,) full 3D body velocity vector [v_fwd, v_lat, v_up]^T
    """
    R_nb = quaternion_to_rotation_matrix(state.q)
    v_b = R_nb.T @ state.v
    h_nhc = np.array([v_b[1], v_b[2]], dtype=np.float64)
    return h_nhc, v_b


def compute_nhc_jacobian(state: NominalState) -> np.ndarray:
    """
    Analytically computes the 2x15 measurement Jacobian H_NHC = dh_NHC / d(delta_x).
    
    Structure:
        H = [ 0_{2x3},  H_v,  H_theta,  0_{2x3},  0_{2x3} ]
        where:
        H_v = [ e2^T * R_{nb}^T ] = [ (Col 1 of R_{nb})^T ]
              [ e3^T * R_{nb}^T ]   [ (Col 2 of R_{nb})^T ]
        H_theta = [ e2^T * [v_b]_x ] = [  v_up,        0,  -v_fwd ]
                  [ e3^T * [v_b]_x ]   [ -v_lat,  v_fwd,       0 ]
    """
    H = np.zeros((2, 15), dtype=np.float64)
    R_nb = quaternion_to_rotation_matrix(state.q)
    _, v_b = predict_nhc_velocity(state)

    v_fwd = v_b[0]
    v_lat = v_b[1]
    v_up = v_b[2]

    # Velocity block (2 x 3)
    H[0, 3:6] = R_nb[:, 1]  # Col 1 of R_nb = Row 1 of R_nb^T
    H[1, 3:6] = R_nb[:, 2]  # Col 2 of R_nb = Row 2 of R_nb^T

    # Attitude block (2 x 3)
    H[0, 6:9] = np.array([v_up, 0.0, -v_fwd], dtype=np.float64)
    H[1, 6:9] = np.array([-v_lat, v_fwd, 0.0], dtype=np.float64)

    return H


def apply_nhc_update(
    state: NominalState,
    P: np.ndarray,
    R_nhc: np.ndarray,
    nis_threshold: Optional[float] = None,
) -> UpdateResult:
    """
    Applies 2D Non-Holonomic Constraint update (v_lat = 0, v_up = 0) with Joseph stabilization.
    """
    h_nhc, _ = predict_nhc_velocity(state)
    z_nhc = np.zeros(2, dtype=np.float64)
    H_nhc = compute_nhc_jacobian(state)

    return RobustMeasurementUpdater.update(
        nominal_state=state,
        P=P,
        z=z_nhc,
        h_val=h_nhc,
        H=H_nhc,
        R=R_nhc,
        nis_threshold=nis_threshold,
    )
