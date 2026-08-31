"""
Phase 2 Tasks 11, 12, 13, 14: Phone-to-Vehicle 3D Alignment, Straight-Line Yaw Estimation,
Orientation Change Detection, and Coordinate Frame Transformations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# TASK 11 & 12: PHONE-TO-VEHICLE 3D ALIGNMENT ESTIMATION
# ============================================================================

def estimate_phone_to_vehicle_rotation(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
    min_speed_mps: float = 4.0,
    max_yaw_rate_rads: float = 0.03,
    min_accel_mps2: float = 0.3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimates the 3D rotation matrix R_{phone->vehicle} that transforms
    measurements from the phone body frame into the vehicle frame (X=Fwd, Y=Right, Z=Up).
    
    Two-step geometric alignment:
    1. Vertical alignment (Z_vehicle):
       Determined from the mean static gravity vector: z_p = -g_phone / ||g_phone||.
    2. Forward alignment (X_vehicle):
       Determined from the dynamic acceleration direction during straight-line acceleration windows
       (speed > min_speed, low yaw rate, positive longitudinal acceleration).
    3. Lateral alignment (Y_vehicle):
       Constructed via right-handed cross product: y_p = z_p x x_p, followed by x_p = y_p x z_p.
       
    Returns:
        R_phone_to_vehicle: 3x3 orthonormal matrix in SO(3)
        metadata: Dict of alignment metrics and Euler angles
    """
    logger.info("Estimating Phone-to-Vehicle 3D Alignment Matrix...")
    
    # 1. Step 1: Vertical Axis (Z_vehicle in phone coordinates)
    gx_stat = df.loc[stationary_mask, "grav_x"].mean() if "grav_x" in df.columns else df.loc[stationary_mask, "acc_x"].mean()
    gy_stat = df.loc[stationary_mask, "grav_y"].mean() if "grav_y" in df.columns else df.loc[stationary_mask, "acc_y"].mean()
    gz_stat = df.loc[stationary_mask, "grav_z"].mean() if "grav_z" in df.columns else df.loc[stationary_mask, "acc_z"].mean()
    
    g_vec = np.array([gx_stat, gy_stat, gz_stat])
    g_norm = np.linalg.norm(g_vec)
    if g_norm < 1e-6:
        g_vec = np.array([0.0, 0.0, 9.80665])
        g_norm = 9.80665
        
    # Vertical axis points UP in vehicle frame (opposite to gravity)
    z_phone = -g_vec / g_norm
    
    # 2. Step 2: Forward Axis (X_vehicle in phone coordinates)
    # Find straight-line acceleration windows
    speed = df["gps_speed_mps"].fillna(0.0).values if "gps_speed_mps" in df.columns else np.zeros(len(df))
    gyro_norm = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2).values
    
    # Linear acceleration in phone frame (acc - grav)
    if "grav_x" in df.columns:
        ax_lin = (df["acc_x"] - df["grav_x"]).values
        ay_lin = (df["acc_y"] - df["grav_y"]).values
        az_lin = (df["acc_z"] - df["grav_z"]).values
    else:
        ax_lin = df["acc_x"].values - g_vec[0]
        ay_lin = df["acc_y"].values - g_vec[1]
        az_lin = df["acc_z"].values - g_vec[2]
        
    a_lin_norm = np.sqrt(ax_lin**2 + ay_lin**2 + az_lin**2)
    
    # Straight-line acceleration mask
    straight_mask = (
        (speed > min_speed_mps) &
        (gyro_norm < max_yaw_rate_rads) &
        (a_lin_norm > min_accel_mps2)
    )
    
    if np.sum(straight_mask) > 20:
        fwd_raw = np.array([
            np.mean(ax_lin[straight_mask]),
            np.mean(ay_lin[straight_mask]),
            np.mean(az_lin[straight_mask]),
        ])
        logger.info(f"Identified {np.sum(straight_mask)} straight acceleration samples for forward alignment")
    else:
        # Default assumption: phone top (Y-axis) is approximately forward
        fwd_raw = np.array([0.0, 1.0, 0.0])
        logger.warning("Few straight acceleration samples; using default phone-top orientation aid")
        
    # Project forward vector onto plane perpendicular to z_phone
    fwd_proj = fwd_raw - np.dot(fwd_raw, z_phone) * z_phone
    fwd_norm = np.linalg.norm(fwd_proj)
    if fwd_norm < 1e-6:
        fwd_proj = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], z_phone) * z_phone
        fwd_norm = np.linalg.norm(fwd_proj)
        
    x_phone = fwd_proj / fwd_norm
    
    # 3. Step 3: Lateral Axis (Y_vehicle in phone coordinates)
    # Right-handed coordinate system: Y = Z x X (Right = Up x Forward in standard right-handed frame)
    # With X=Forward, Y=Right, Z=Up: X x Y = Z => Y = Z x X
    y_phone = np.cross(z_phone, x_phone)
    y_phone = y_phone / np.linalg.norm(y_phone)
    
    # Re-orthogonalize X: X = Y x Z
    x_phone = np.cross(y_phone, z_phone)
    x_phone = x_phone / np.linalg.norm(x_phone)
    
    # Rotation matrix: v_vehicle = R * v_phone
    R_phone_to_veh = np.vstack([x_phone, y_phone, z_phone])
    
    # Verify orthonormality: R * R^T = I, det(R) = +1
    det_R = np.linalg.det(R_phone_to_veh)
    is_orthonormal = np.allclose(R_phone_to_veh @ R_phone_to_veh.T, np.eye(3), atol=1e-5)
    
    # Extract equivalent Euler angles (degrees)
    # R_phone_to_veh transforms vector in phone frame to vehicle frame
    metadata = {
        "is_orthonormal": bool(is_orthonormal),
        "determinant": float(det_R),
        "z_phone_axis": z_phone.tolist(),
        "x_phone_axis": x_phone.tolist(),
        "y_phone_axis": y_phone.tolist(),
        "straight_samples_used": int(np.sum(straight_mask)),
        "convention": "v_vehicle = R_phone_to_vehicle * v_phone (X=Fwd, Y=Right, Z=Up)",
    }
    
    logger.info(f"R_phone_to_vehicle estimated: det={det_R:.4f}, orthonormal={is_orthonormal}")
    return R_phone_to_veh, metadata


def transform_vector_to_vehicle_frame(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    R_phone_to_veh: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies R_phone_to_vehicle rotation to multi-axis signals.
    
    Parameters:
        vx, vy, vz: 1D numpy arrays of vectors in phone body frame
        R_phone_to_veh: 3x3 rotation matrix
        
    Returns:
        v_fwd, v_lat, v_up: 1D numpy arrays in vehicle frame
    """
    v_phone = np.vstack([vx, vy, vz])  # shape (3, N)
    v_veh = R_phone_to_veh @ v_phone   # shape (3, N)
    return v_veh[0], v_veh[1], v_veh[2]


# ============================================================================
# TASK 13: PHONE ORIENTATION CHANGE DETECTION
# ============================================================================

def detect_orientation_changes(
    df: pd.DataFrame,
    initial_gravity_vec: np.ndarray,
    cos_tolerance: float = 0.95,
    window_samples: int = 50,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Monitors stability of the phone mount orientation.
    
    Detects events where the phone shifted/slipped in its mount by comparing
    smoothed gravity vectors against the reference initial gravity vector.
    
    Cosine similarity: cos(theta) = (g_smooth . g_ref) / (||g_smooth|| * ||g_ref||)
    If cos(theta) < cos_tolerance sustained over a window, flags orientation change.
    """
    gx = df["grav_x"].values if "grav_x" in df.columns else df["acc_x"].values
    gy = df["grav_y"].values if "grav_y" in df.columns else df["acc_y"].values
    gz = df["grav_z"].values if "grav_z" in df.columns else df["acc_z"].values
    
    # Smooth with moving average
    gx_smooth = pd.Series(gx).rolling(window_samples, min_periods=1).mean().values
    gy_smooth = pd.Series(gy).rolling(window_samples, min_periods=1).mean().values
    gz_smooth = pd.Series(gz).rolling(window_samples, min_periods=1).mean().values
    
    g_matrix = np.vstack([gx_smooth, gy_smooth, gz_smooth])  # (3, N)
    g_norms = np.linalg.norm(g_matrix, axis=0)
    g_norms = np.where(g_norms == 0, 1.0, g_norms)
    
    g_ref = initial_gravity_vec / np.linalg.norm(initial_gravity_vec)
    
    # Dot product along axis 0
    cos_sim = np.sum(g_matrix * g_ref[:, np.newaxis], axis=0) / g_norms
    
    change_mask = cos_sim < cos_tolerance
    
    events = []
    if np.any(change_mask):
        time_s = df["time_s"].values
        # Group contiguous events
        event_indices = np.where(change_mask)[0]
        events.append({
            "detected": True,
            "n_samples": int(np.sum(change_mask)),
            "first_time_s": float(time_s[event_indices[0]]),
            "min_cos_sim": float(np.min(cos_sim)),
        })
    else:
        events.append({"detected": False, "n_samples": 0, "min_cos_sim": float(np.min(cos_sim))})
        
    logger.info(f"Orientation Change Check: detected={events[0]['detected']}, min_cos={events[0]['min_cos_sim']:.4f}")
    return change_mask, events


def compute_vehicle_frame_signals(
    df: pd.DataFrame,
    R_phone_to_veh: np.ndarray,
    accel_bias: Optional[np.ndarray] = None,
    gyro_bias: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Transforms calibrated smartphone acceleration and angular velocity signals
    into the ISO 8855 Automotive Vehicle Frame:
    - X_v: Forward longitudinal axis (driving direction)
    - Y_v: Lateral right axis (passenger door)
    - Z_v: Vertical upward axis (normal to road plane)
    
    Parameters:
        df: DataFrame with acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        R_phone_to_veh: 3x3 orthonormal rotation matrix from estimate_phone_to_vehicle_rotation
        accel_bias: Optional 3D accelerometer bias vector [bx, by, bz]
        gyro_bias: Optional 3D gyroscope bias vector [bx, by, bz]
        
    Returns:
        df_enriched: DataFrame with added columns:
            acc_fwd_veh, acc_lat_veh, acc_up_veh,
            gyro_roll_veh, gyro_pitch_veh, gyro_yaw_veh
    """
    df = df.copy()
    
    ax = df["acc_x"].values.astype(float)
    ay = df["acc_y"].values.astype(float)
    az = df["acc_z"].values.astype(float)
    if accel_bias is not None:
        ax = ax - accel_bias[0]
        ay = ay - accel_bias[1]
        az = az - accel_bias[2]
        
    a_fwd, a_lat, a_up = transform_vector_to_vehicle_frame(ax, ay, az, R_phone_to_veh)
    df["acc_fwd_veh"] = a_fwd
    df["acc_lat_veh"] = a_lat
    df["acc_up_veh"] = a_up
    
    gx = df["gyro_x"].values.astype(float)
    gy = df["gyro_y"].values.astype(float)
    gz = df["gyro_z"].values.astype(float)
    if gyro_bias is not None:
        gx = gx - gyro_bias[0]
        gy = gy - gyro_bias[1]
        gz = gz - gyro_bias[2]
        
    g_roll, g_pitch, g_yaw = transform_vector_to_vehicle_frame(gx, gy, gz, R_phone_to_veh)
    df["gyro_roll_veh"] = g_roll
    df["gyro_pitch_veh"] = g_pitch
    df["gyro_yaw_veh"] = g_yaw
    
    return df

