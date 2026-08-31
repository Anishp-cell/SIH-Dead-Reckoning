"""
Phase 2 Tasks 7, 8, 9: Quaternion Mathematics, Gravity-Based Tilt Estimation,
Gyroscope Attitude Propagation, and Complementary Filter Attitude Estimator.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# QUATERNION MATHEMATICAL FOUNDATIONS
# ============================================================================

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion [qw, qx, qy, qz] to unit length."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        norm = np.where(norm == 0, 1.0, norm)
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamiltonian quaternion product q = q1 ⊗ q2.
    Format: [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([w, x, y, z], axis=-1)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q* = [qw, -qx, -qy, -qz]."""
    q_conj = q.copy()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [qw, qx, qy, qz] to 3x3 direction cosine matrix R(q).
    Transforms vector from Body frame to Navigation/Reference frame: v_nav = R * v_body.
    """
    q = quaternion_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz] (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
        
    return quaternion_normalize(np.array([qw, qx, qy, qz]))


def quaternion_to_euler_deg(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts Euler angles (Roll, Pitch, Yaw) in degrees (ZYX convention).
    Returns: (roll_deg, pitch_deg, yaw_deg)
    """
    q = quaternion_normalize(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # gimbal lock
    else:
        pitch = np.arcsin(sinp)
        
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def euler_to_quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Converts Euler angles (Roll, Pitch, Yaw in radians) to unit quaternion."""
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return quaternion_normalize(np.array([qw, qx, qy, qz]))


# ============================================================================
# TASK 7: ROLL AND PITCH ESTIMATION FROM GRAVITY
# ============================================================================

def estimate_tilt_from_accel(ax: float, ay: float, az: float) -> Tuple[float, float]:
    """
    Estimates roll and pitch angles from static accelerometer readings.
    
    Convention (Z-up, X-forward, Y-right):
        roll  phi   = atan2(ay, az)
        pitch theta = atan2(-ax, sqrt(ay^2 + az^2))
    
    Returns: (roll_rad, pitch_rad)
    """
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return float(roll), float(pitch)


def accel_to_leveling_quaternion(ax: float, ay: float, az: float, yaw_ref_rad: float = 0.0) -> np.ndarray:
    """Constructs a leveling attitude quaternion from accelerometer gravity vector."""
    roll, pitch = estimate_tilt_from_accel(ax, ay, az)
    return euler_to_quaternion(roll, pitch, yaw_ref_rad)


# ============================================================================
# TASK 8: GYROSCOPE ATTITUDE PROPAGATION
# ============================================================================

def propagate_quaternion_gyro(
    q: np.ndarray,
    omega: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Propagates attitude quaternion forward in time using measured angular velocity.
    
    Algorithm:
        angle = ||omega|| * dt
        dq = [cos(angle/2), (sin(angle/2)/angle) * omega * dt]
        q_{k+1} = q_k ⊗ dq
    """
    omega_norm = np.linalg.norm(omega)
    half_angle = 0.5 * omega_norm * dt
    
    if omega_norm < 1e-8:
        # Small angle approximation (Taylor series)
        dq = np.array([1.0, 0.5 * omega[0] * dt, 0.5 * omega[1] * dt, 0.5 * omega[2] * dt])
    else:
        sinc = np.sin(half_angle) / omega_norm
        dq = np.array([
            np.cos(half_angle),
            sinc * omega[0],
            sinc * omega[1],
            sinc * omega[2],
        ])
        
    q_next = quaternion_multiply(q, dq)
    return quaternion_normalize(q_next)


# ============================================================================
# TASK 9: COMPLEMENTARY FILTER ATTITUDE ESTIMATOR
# ============================================================================

class ComplementaryAttitudeEstimator:
    """
    Lightweight, robust Complementary Filter for attitude estimation (q_body_to_nav).
    
    Fuses:
    - Gyroscope: high-rate dynamic integration (alpha ~ 0.98)
    - Accelerometer: low-rate absolute gravity leveling ((1-alpha) ~ 0.02)
    
    Adaptive gain: suppresses accelerometer correction during strong vehicle acceleration.
    """
    
    def __init__(
        self,
        alpha: float = 0.98,
        gravity_norm_expected: float = 9.80665,
        accel_tolerance: float = 0.6,
    ):
        self.alpha = alpha
        self.g_expected = gravity_norm_expected
        self.accel_tol = accel_tolerance
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.initialized = False
        
    def initialize(self, ax: float, ay: float, az: float, initial_yaw_deg: float = 0.0) -> None:
        """Initializes attitude from static gravity leveling."""
        yaw_rad = np.radians(initial_yaw_deg)
        self.q = accel_to_leveling_quaternion(ax, ay, az, yaw_rad)
        self.initialized = True
        logger.info(f"Attitude Estimator initialized: Euler={quaternion_to_euler_deg(self.q)}")
        
    def update(
        self,
        accel: np.ndarray,
        gyro_bias_corrected: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Executes one timestep update of complementary filter.
        
        Returns:
            Current attitude quaternion q [qw, qx, qy, qz]
        """
        if not self.initialized:
            self.initialize(accel[0], accel[1], accel[2])
            return self.q
            
        # 1. Gyroscope propagation
        q_gyro = propagate_quaternion_gyro(self.q, gyro_bias_corrected, dt)
        
        # 2. Check if accelerometer vector is reliable for gravity leveling
        a_norm = np.linalg.norm(accel)
        is_gravity_reliable = abs(a_norm - self.g_expected) < self.accel_tol
        
        if is_gravity_reliable:
            # Gravity leveling estimate
            # Preserve current estimated yaw, update roll and pitch
            _, _, cur_yaw_deg = quaternion_to_euler_deg(q_gyro)
            q_accel = accel_to_leveling_quaternion(accel[0], accel[1], accel[2], np.radians(cur_yaw_deg))
            
            # SLERP / Linear interpolation of quaternions
            # Ensure shortest path
            if np.dot(q_gyro, q_accel) < 0:
                q_accel = -q_accel
                
            q_fused = self.alpha * q_gyro + (1.0 - self.alpha) * q_accel
            self.q = quaternion_normalize(q_fused)
        else:
            # During dynamic acceleration, rely solely on gyro propagation
            self.q = q_gyro
            
        return self.q


def estimate_sequence_attitude(
    df: pd.DataFrame,
    gyro_bias: Optional[np.ndarray] = None,
    alpha: float = 0.98,
    initial_yaw_deg: float = 0.0,
) -> np.ndarray:
    """
    Runs complementary attitude filter over an entire DataFrame sequence.
    
    Returns:
        Array of unit quaternions shape (N, 4)
    """
    n = len(df)
    quats = np.zeros((n, 4))
    
    if gyro_bias is None:
        gyro_bias = np.zeros(3)
        
    filter_inst = ComplementaryAttitudeEstimator(alpha=alpha)
    
    # Initialize from first 10 samples average
    a_init = df[["acc_x", "acc_y", "acc_z"]].iloc[:10].mean().values
    filter_inst.initialize(a_init[0], a_init[1], a_init[2], initial_yaw_deg)
    
    acc_arr = df[["acc_x", "acc_y", "acc_z"]].values
    gyro_arr = df[["gyro_x", "gyro_y", "gyro_z"]].values
    dt_arr = df["dt"].values
    
    for i in range(n):
        w_corr = gyro_arr[i] - gyro_bias
        quats[i] = filter_inst.update(acc_arr[i], w_corr, dt_arr[i])
        
    return quats
