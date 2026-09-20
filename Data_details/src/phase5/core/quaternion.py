"""
Phase 5 Quaternion Mathematics & SO(3) Kinematics.
Strict scalar-first convention: q = [qw, qx, qy, qz].
"""

import numpy as np
from typing import Tuple


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion [qw, qx, qy, qz] to unit length."""
    q_arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q_arr)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q_arr / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamiltonian quaternion product q = q1 ⊗ q2.
    Format: [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q* = [qw, -qx, -qy, -qz]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [qw, qx, qy, qz] to 3x3 direction cosine matrix R_{nb}.
    Transforms vector from Body frame to Navigation frame: v_nav = R_{nb} * v_body.
    """
    q_u = quaternion_normalize(q)
    w, x, y, z = q_u[0], q_u[1], q_u[2], q_u[3]

    R = np.array([
        [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x**2 + y**2)],
    ], dtype=np.float64)
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz] using Shepperd's method."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return quaternion_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def quaternion_rotate_vector(q: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Rotates 3D vector v_b in body frame to v_n in navigation frame: v_n = R_{nb} * v_b."""
    R = quaternion_to_rotation_matrix(q)
    return R @ np.asarray(v_b, dtype=np.float64)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Constructs 3x3 skew-symmetric matrix [v]_x from 3D vector."""
    vx, vy, vz = v[0], v[1], v[2]
    return np.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0],
    ], dtype=np.float64)


def quaternion_integrate(q: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    """
    Propagates attitude quaternion q forward in time by dt using angular velocity omega_b.
    Uses exact matrix exponential / closed-form rotation vector formula.
    """
    omega = np.asarray(omega_b, dtype=np.float64)
    angle = np.linalg.norm(omega) * dt

    if angle < 1e-8:
        # First-order Taylor approximation
        dq = np.array([1.0, 0.5 * omega[0] * dt, 0.5 * omega[1] * dt, 0.5 * omega[2] * dt], dtype=np.float64)
    else:
        half_angle = 0.5 * angle
        sinc = np.sin(half_angle) / (np.linalg.norm(omega))
        dq = np.array([
            np.cos(half_angle),
            sinc * omega[0],
            sinc * omega[1],
            sinc * omega[2],
        ], dtype=np.float64)

    q_next = quaternion_multiply(q, dq)
    return quaternion_normalize(q_next)


def small_angle_to_quaternion(dtheta: np.ndarray) -> np.ndarray:
    """Converts 3D small-angle error vector dtheta into delta quaternion: delta_q = [1, 0.5 * dtheta]."""
    dth = np.asarray(dtheta, dtype=np.float64)
    dq = np.array([1.0, 0.5 * dth[0], 0.5 * dth[1], 0.5 * dth[2]], dtype=np.float64)
    return quaternion_normalize(dq)


def inject_small_angle_error(q_nominal: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Injects body-frame small-angle rotation error dtheta into nominal quaternion:
        q_true = q_nominal ⊗ [1, 0.5 * dtheta]
    """
    dq = small_angle_to_quaternion(dtheta)
    q_corrected = quaternion_multiply(q_nominal, dq)
    return quaternion_normalize(q_corrected)


def quaternion_to_euler_zyx(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Converts unit quaternion to ZYX Euler angles (roll, pitch, yaw) in radians.
    Returns: (roll, pitch, yaw)
    """
    q_u = quaternion_normalize(q)
    w, x, y, z = q_u[0], q_u[1], q_u[2], q_u[3]

    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def euler_zyx_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Converts ZYX Euler angles (roll, pitch, yaw in radians) to unit quaternion."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return quaternion_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))
