"""
Phase 5 Coordinate Frames & Transformation Utilities.
Navigation frame: ENU (East-North-Up).
Body frame: ISO 8855 Adapted (Forward-Right-Up).
"""

import numpy as np
from typing import Tuple
from Data_details.src.phase5.core.quaternion import (
    quaternion_to_rotation_matrix,
    euler_zyx_to_quaternion,
    quaternion_normalize,
)

GRAVITY_MAGNITUDE = 9.80665  # Standard Earth gravity (m/s^2)
GRAVITY_VECTOR_ENU = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE], dtype=np.float64)


def body_to_nav(v_body: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Transforms a 3D vector from Body frame to Navigation (ENU) frame: v_nav = R_{nb} * v_body."""
    R = quaternion_to_rotation_matrix(q)
    return R @ np.asarray(v_body, dtype=np.float64)


def nav_to_body(v_nav: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Transforms a 3D vector from Navigation (ENU) frame to Body frame: v_body = R_{nb}^T * v_nav."""
    R = quaternion_to_rotation_matrix(q)
    return R.T @ np.asarray(v_nav, dtype=np.float64)


def geographic_to_enu_yaw_rad(psi_gps_rad: float) -> float:
    """
    Converts geographic azimuth heading (clockwise from North) to Cartesian ENU yaw (counter-clockwise from East).
    Relationship: psi_enu = pi/2 - psi_gps.
    """
    enu_yaw = np.pi / 2.0 - psi_gps_rad
    # Wrap to [-pi, pi]
    return float((enu_yaw + np.pi) % (2.0 * np.pi) - np.pi)


def enu_yaw_to_geographic_rad(psi_enu_rad: float) -> float:
    """
    Converts Cartesian ENU yaw (counter-clockwise from East) to geographic azimuth heading (clockwise from North).
    Relationship: psi_gps = pi/2 - psi_enu.
    """
    gps_yaw = np.pi / 2.0 - psi_enu_rad
    # Wrap to [0, 2*pi]
    return float(gps_yaw % (2.0 * np.pi))


def geographic_to_enu_yaw_deg(psi_gps_deg: float) -> float:
    """Converts geographic azimuth in degrees to Cartesian ENU yaw in degrees."""
    gps_rad = np.radians(psi_gps_deg)
    enu_rad = geographic_to_enu_yaw_rad(gps_rad)
    return float(np.degrees(enu_rad))


def enu_yaw_to_geographic_deg(psi_enu_deg: float) -> float:
    """Converts Cartesian ENU yaw in degrees to geographic azimuth in degrees."""
    enu_rad = np.radians(psi_enu_deg)
    gps_rad = enu_yaw_to_geographic_rad(enu_rad)
    return float(np.degrees(gps_rad))


def estimate_tilt_from_static_accel(acc_b: np.ndarray) -> Tuple[float, float]:
    """
    Estimates roll and pitch angles from static accelerometer readings in ISO 8855 frame (X=Fwd, Y=Right, Z=Up).
    When level on flat ground, accelerometer measures upward normal force: [0, 0, +g]^T.
    Returns: (roll_rad, pitch_rad)
    """
    ax, ay, az = acc_b[0], acc_b[1], acc_b[2]
    # roll phi: rotation around forward x-axis (lateral tilt)
    roll = np.arctan2(-ay, az)
    # pitch theta: rotation around lateral right y-axis (longitudinal tilt)
    pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))
    return float(roll), float(pitch)


def initial_leveling_quaternion(acc_b: np.ndarray, initial_gps_heading_deg: float = 0.0) -> np.ndarray:
    """
    Initializes attitude quaternion from static accelerometer gravity vector and initial GPS heading.
    """
    roll, pitch = estimate_tilt_from_static_accel(acc_b)
    yaw_enu = geographic_to_enu_yaw_rad(np.radians(initial_gps_heading_deg))
    return euler_zyx_to_quaternion(roll, pitch, yaw_enu)
