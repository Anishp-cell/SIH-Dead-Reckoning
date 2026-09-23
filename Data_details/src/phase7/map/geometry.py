"""
Phase 7 Road Geometry Engine:
Provides high-performance 2D geometric operations:
- Orthogonal point-to-segment projection
- Signed cross-track distance
- Along-track distance and segment fraction
- Road tangent azimuth and normal vector computation
- Angle wrapping and frame conversions (Cartesian ENU yaw <-> Geographic heading)
"""

from dataclasses import dataclass
from typing import Tuple, Union, Optional
import numpy as np


def wrap_angle_rad(angle_rad: float) -> float:
    """Wraps an angle in radians to [-pi, pi]."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_angle_deg(angle_deg: float) -> float:
    """Wraps an angle in degrees to [-180, 180]."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def compute_segment_azimuth_enu(p_start_2d: np.ndarray, p_end_2d: np.ndarray) -> float:
    """
    Computes Cartesian ENU yaw angle of a directed line segment from start to end.
    Cartesian yaw: 0 at East, +pi/2 at North, counter-clockwise positive in [-pi, pi].
    """
    de = float(p_end_2d[0] - p_start_2d[0])
    dn = float(p_end_2d[1] - p_start_2d[1])
    return float(np.arctan2(dn, de))


def enu_yaw_to_geographic_azimuth_deg(psi_enu_rad: float) -> float:
    """
    Converts Cartesian ENU yaw (rad) to Geographic Azimuth (deg, clockwise from North in [0, 360)).
    psi_gps = pi/2 - psi_enu
    """
    gps_rad = np.pi / 2.0 - psi_enu_rad
    return float((np.degrees(gps_rad) + 360.0) % 360.0)


def geographic_azimuth_to_enu_yaw_rad(psi_gps_deg: float) -> float:
    """
    Converts Geographic Azimuth (deg, clockwise from North) to Cartesian ENU yaw (rad in [-pi, pi]).
    psi_enu = pi/2 - psi_gps
    """
    gps_rad = np.radians(psi_gps_deg)
    enu_rad = np.pi / 2.0 - gps_rad
    return wrap_angle_rad(enu_rad)


@dataclass(frozen=True)
class PointToSegmentResult:
    """
    Result of orthogonal projection of a 2D point onto a directed road segment.
    """
    projected_point_enu: np.ndarray      # [East, North] in meters
    cross_track_m: float                 # Signed lateral distance (positive = to the left of directed road)
    abs_distance_m: float                # Absolute Euclidean distance to closest point on segment
    along_track_m: float                 # Distance along road from segment start in meters
    fraction: float                      # Parameter u in [0.0, 1.0]
    segment_heading_enu_rad: float       # Road tangent heading in Cartesian ENU rad
    unit_tangent: np.ndarray             # [t_E, t_N]
    unit_normal: np.ndarray              # [n_E, n_N] where n = [-t_N, t_E] (left-pointing normal)
    is_clamped: bool                     # True if projected onto segment endpoint (vertex)


def project_point_to_segment(
    p_enu_2d: np.ndarray,
    a_enu_2d: np.ndarray,
    b_enu_2d: np.ndarray,
) -> PointToSegmentResult:
    """
    Projects a 2D ENU point onto a directed line segment from a to b.

    Parameters:
        p_enu_2d: 2D query point [East, North]
        a_enu_2d: 2D segment start node [East, North]
        b_enu_2d: 2D segment end node [East, North]

    Returns:
        PointToSegmentResult containing projected point, signed cross-track distance,
        fraction, tangent, and normal vectors.
    """
    p = np.asarray(p_enu_2d, dtype=np.float64)[:2]
    a = np.asarray(a_enu_2d, dtype=np.float64)[:2]
    b = np.asarray(b_enu_2d, dtype=np.float64)[:2]

    v = b - a  # Segment vector
    length_sq = float(v[0]**2 + v[1]**2)

    # Degenerate zero-length segment
    if length_sq < 1e-12:
        diff = p - a
        dist = float(np.linalg.norm(diff))
        return PointToSegmentResult(
            projected_point_enu=a.copy(),
            cross_track_m=dist,
            abs_distance_m=dist,
            along_track_m=0.0,
            fraction=0.0,
            segment_heading_enu_rad=0.0,
            unit_tangent=np.array([1.0, 0.0], dtype=np.float64),
            unit_normal=np.array([0.0, 1.0], dtype=np.float64),
            is_clamped=True,
        )

    length = np.sqrt(length_sq)
    tangent = v / length
    # Left-hand normal: rotate tangent +90 deg counter-clockwise: [-t_N, t_E]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    w = p - a
    s = float(w[0] * tangent[0] + w[1] * tangent[1])  # Along-track distance from a
    u = s / length  # Normalized fraction

    is_clamped = False
    if u <= 0.0:
        u_clamped = 0.0
        p_proj = a.copy()
        is_clamped = True
    elif u >= 1.0:
        u_clamped = 1.0
        p_proj = b.copy()
        is_clamped = True
    else:
        u_clamped = u
        p_proj = a + u_clamped * v

    # Signed cross-track: projection onto normal vector
    # (p - p_proj) . n
    disp = p - p_proj
    cross_track = float(disp[0] * normal[0] + disp[1] * normal[1])
    abs_dist = float(np.linalg.norm(disp))
    along_track = u_clamped * length

    heading_enu = float(np.arctan2(tangent[1], tangent[0]))

    return PointToSegmentResult(
        projected_point_enu=p_proj,
        cross_track_m=cross_track,
        abs_distance_m=abs_dist,
        along_track_m=along_track,
        fraction=u_clamped,
        segment_heading_enu_rad=heading_enu,
        unit_tangent=tangent,
        unit_normal=normal,
        is_clamped=is_clamped,
    )
