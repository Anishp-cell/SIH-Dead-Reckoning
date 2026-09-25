"""
Phase 6 Non-Holonomic Constraints (NHC) Implementation:
Lateral (v_y^b = 0) and Vertical (v_z^b = 0) Vehicle Body Velocity Updates.
"""

from typing import Tuple, Optional
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import quaternion_to_rotation_matrix
from Data_details.src.phase6.measurement.robust_update import RobustMeasurementUpdater, UpdateResult


def compute_two_wheeler_steady_lean_angle(
    v_forward_mps: float,
    yaw_rate_rads: float,
    gravity_mps2: float = 9.80665,
    max_lean_angle_deg: float = 45.0,
) -> float:
    """
    Estimates steady-state motorcycle / scooter roll/lean angle during cornering:
    tan(phi) = (v_fwd * omega_z) / g.
    """
    if abs(v_forward_mps) < 0.5:
        return 0.0
    ratio = (v_forward_mps * yaw_rate_rads) / gravity_mps2
    phi = float(np.arctan(ratio))
    max_rad = float(np.radians(max_lean_angle_deg))
    return float(np.clip(phi, -max_rad, max_rad))


def predict_nhc_velocity(
    state: NominalState,
    lean_angle_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes body-frame velocity v_b = R_{nb}^T * v_n = [v_fwd, v_lat, v_up]^T
    and projects into the road contact plane when leaning (two-wheelers).
    
    Returns:
        h_nhc: (2,) array [v_contact_lat, v_contact_up]^T
        v_b: (3,) full 3D body velocity vector [v_fwd, v_lat, v_up]^T
    """
    R_nb = quaternion_to_rotation_matrix(state.q)
    v_b = R_nb.T @ state.v
    
    if abs(lean_angle_rad) > 1e-4:
        # Banked turn projection: rotate body transverse velocity into road plane
        c_phi = float(np.cos(lean_angle_rad))
        s_phi = float(np.sin(lean_angle_rad))
        v_contact_lat = v_b[1] * c_phi - v_b[2] * s_phi
        v_contact_up = v_b[1] * s_phi + v_b[2] * c_phi
        h_nhc = np.array([v_contact_lat, v_contact_up], dtype=np.float64)
    else:
        h_nhc = np.array([v_b[1], v_b[2]], dtype=np.float64)

    return h_nhc, v_b


def compute_nhc_jacobian(
    state: NominalState,
    lean_angle_rad: float = 0.0,
) -> np.ndarray:
    """
    Analytically computes the 2x15 measurement Jacobian H_NHC = dh_NHC / d(delta_x).
    Applies lean-angle rotation matrix R_lean when cornering on two-wheelers.
    """
    H = np.zeros((2, 15), dtype=np.float64)
    R_nb = quaternion_to_rotation_matrix(state.q)
    _, v_b = predict_nhc_velocity(state, lean_angle_rad=0.0)

    v_fwd = v_b[0]
    v_lat = v_b[1]
    v_up = v_b[2]

    # Velocity block (2 x 3)
    H[0, 3:6] = R_nb[:, 1]  # Col 1 of R_nb = Row 1 of R_nb^T
    H[1, 3:6] = R_nb[:, 2]  # Col 2 of R_nb = Row 2 of R_nb^T

    # Attitude block (2 x 3)
    H[0, 6:9] = np.array([v_up, 0.0, -v_fwd], dtype=np.float64)
    H[1, 6:9] = np.array([-v_lat, v_fwd, 0.0], dtype=np.float64)

    if abs(lean_angle_rad) > 1e-4:
        c_phi = float(np.cos(lean_angle_rad))
        s_phi = float(np.sin(lean_angle_rad))
        R_lean = np.array([[c_phi, -s_phi], [s_phi, c_phi]], dtype=np.float64)
        H = R_lean @ H

    return H


def apply_nhc_update(
    state: NominalState,
    P: np.ndarray,
    R_nhc: np.ndarray,
    nis_threshold: Optional[float] = None,
    lean_angle_rad: float = 0.0,
) -> UpdateResult:
    """
    Applies 2D Non-Holonomic Constraint update with optional two-wheeler lean compensation.
    """
    h_nhc, _ = predict_nhc_velocity(state, lean_angle_rad=lean_angle_rad)
    z_nhc = np.zeros(2, dtype=np.float64)
    H_nhc = compute_nhc_jacobian(state, lean_angle_rad=lean_angle_rad)

    return RobustMeasurementUpdater.update(
        nominal_state=state,
        P=P,
        z=z_nhc,
        h_val=h_nhc,
        H=H_nhc,
        R=R_nhc,
        nis_threshold=nis_threshold,
    )
