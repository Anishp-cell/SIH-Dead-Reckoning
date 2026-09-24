"""
Geodetic, ECEF, and Local Tangent Plane (ENU) Coordinate Transformations
Compliant with WGS-84 ellipsoid and ISRO SIH26168 navigation conventions.
"""

from typing import Tuple, Union
import numpy as np

# WGS-84 Ellipsoid Parameters
WGS84_A: float = 6378137.0             # Semi-major axis (meters)
WGS84_F: float = 1.0 / 298.257223563   # Flattening
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis (meters) ~ 6356752.314245
WGS84_E2: float = 2.0 * WGS84_F - WGS84_F ** 2  # First eccentricity squared ~ 0.00669437999014
WGS84_EP2: float = (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)  # Second eccentricity squared


def geodetic_to_ecef(
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0,
) -> np.ndarray:
    """
    Transforms WGS-84 geodetic coordinates (lat, lon, alt) to Earth-Centered,
    Earth-Fixed (ECEF) Cartesian coordinates (X, Y, Z).
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Prime vertical radius of curvature
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_phi ** 2)

    x = (n + alt_m) * cos_phi * cos_lam
    y = (n + alt_m) * cos_phi * sin_lam
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_phi

    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(
    ref_lat_deg: float,
    ref_lon_deg: float,
) -> np.ndarray:
    """
    Constructs the 3x3 rotation matrix R_ecef_to_enu from ECEF to local ENU.
    p_enu = R_ecef_to_enu @ (p_ecef - p0_ecef)
    """
    phi0 = np.radians(ref_lat_deg)
    lam0 = np.radians(ref_lon_deg)
    sin_phi0 = np.sin(phi0)
    cos_phi0 = np.cos(phi0)
    sin_lam0 = np.sin(lam0)
    cos_lam0 = np.cos(lam0)

    # Row 1: East (-sin lambda, cos lambda, 0)
    # Row 2: North (-sin phi cos lam, -sin phi sin lam, cos phi)
    # Row 3: Up (cos phi cos lam, cos phi sin lam, sin phi)
    r = np.array([
        [-sin_lam0, cos_lam0, 0.0],
        [-sin_phi0 * cos_lam0, -sin_phi0 * sin_lam0, cos_phi0],
        [cos_phi0 * cos_lam0, cos_phi0 * sin_lam0, sin_phi0],
    ], dtype=np.float64)
    return r


def ecef_to_enu(
    p_ecef: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float = 0.0,
) -> np.ndarray:
    """
    Transforms ECEF Cartesian position [X, Y, Z] to local East-North-Up (ENU)
    relative to a reference geodetic point (ref_lat, ref_lon, ref_alt).
    """
    p0_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    r_mat = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    diff = np.asarray(p_ecef, dtype=np.float64).ravel() - p0_ecef
    return r_mat @ diff


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float = 0.0,
) -> np.ndarray:
    """
    Transforms WGS-84 geodetic coordinates directly to local ENU.
    """
    p_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(p_ecef, ref_lat_deg, ref_lon_deg, ref_alt_m)


def enu_to_ecef(
    p_enu: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float = 0.0,
) -> np.ndarray:
    """
    Transforms local ENU position to ECEF coordinates.
    p_ecef = p0_ecef + R_ecef_to_enu.T @ p_enu
    """
    p0_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    r_mat = ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg)
    return p0_ecef + (r_mat.T @ np.asarray(p_enu, dtype=np.float64).ravel())


def ecef_to_geodetic(
    p_ecef: np.ndarray,
) -> Tuple[float, float, float]:
    """
    High-precision closed-form Bowring transformation from ECEF to Geodetic (lat, lon, alt).
    Accurate to within sub-millimeter precision globally.
    """
    x, y, z = np.asarray(p_ecef, dtype=np.float64).ravel()[:3]
    p = np.sqrt(x ** 2 + y ** 2)

    if p < 1e-6:
        # Near poles
        lat_deg = 90.0 if z >= 0 else -90.0
        lon_deg = 0.0
        alt_m = np.abs(z) - WGS84_B
        return lat_deg, lon_deg, alt_m

    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    phi = np.arctan2(
        z + WGS84_EP2 * WGS84_B * (sin_theta ** 3),
        p - WGS84_E2 * WGS84_A * (cos_theta ** 3),
    )
    lam = np.arctan2(y, x)

    sin_phi = np.sin(phi)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_phi ** 2)
    alt_m = (p / np.cos(phi)) - n

    return float(np.degrees(phi)), float(np.degrees(lam)), float(alt_m)


def enu_to_geodetic(
    p_enu: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Transforms local ENU coordinates back to geodetic (lat_deg, lon_deg, alt_m).
    """
    p_ecef = enu_to_ecef(p_enu, ref_lat_deg, ref_lon_deg, ref_alt_m)
    return ecef_to_geodetic(p_ecef)


def gps_bearing_to_enu_yaw_rad(bearing_deg: float) -> float:
    """
    Converts GNSS compass bearing (degrees clockwise from True North [0, 360))
    to Cartesian ENU yaw (radians counter-clockwise from East [-pi, pi]).
    Formula: psi_enu = wrap_to_pi(pi/2 - deg2rad(bearing_deg))
    """
    rad = np.radians(bearing_deg)
    yaw = (np.pi / 2.0) - rad
    return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


def enu_yaw_to_gps_bearing_deg(yaw_enu_rad: float) -> float:
    """
    Converts Cartesian ENU yaw (radians counter-clockwise from East)
    to GNSS compass bearing (degrees clockwise from True North [0, 360)).
    Formula: bearing = (90 - rad2deg(yaw_enu_rad)) mod 360
    """
    deg = np.degrees(yaw_enu_rad)
    bearing = (90.0 - deg) % 360.0
    return float(bearing)


def gps_vel_to_enu_velocity(
    speed_mps: float,
    bearing_deg: float,
    climb_rate_mps: float = 0.0,
) -> np.ndarray:
    """
    Converts GNSS horizontal speed and bearing into 3D ENU velocity vector [v_E, v_N, v_U].
    East = speed * sin(bearing)
    North = speed * cos(bearing)
    Up = climb_rate
    """
    b_rad = np.radians(bearing_deg)
    v_e = speed_mps * np.sin(b_rad)
    v_n = speed_mps * np.cos(b_rad)
    v_u = climb_rate_mps
    return np.array([v_e, v_n, v_u], dtype=np.float64)
