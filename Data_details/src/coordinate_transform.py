"""
Task 10: Coordinate Transformation — WGS84 Geodetic to local ENU (East-North-Up).
Also provides geodesic distance calculations using pyproj.
"""

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

# WGS84 constants
WGS84_A = 6378137.0           # semi-major axis in meters
WGS84_F = 1.0 / 298.257223563 # flattening
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # eccentricity squared


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts WGS84 geodetic coordinates to Earth-Centered Earth-Fixed (ECEF).
    
    Parameters:
        lat_deg: latitude in degrees
        lon_deg: longitude in degrees  
        alt_m: altitude in meters above ellipsoid
    
    Returns:
        (x, y, z) in meters (ECEF)
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
    
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat
    
    return x, y, z


def ecef_to_enu(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    lat0_deg: float, lon0_deg: float, alt0_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts ECEF coordinates to local ENU (East-North-Up) relative to an origin.
    
    Parameters:
        x, y, z: ECEF coordinates in meters
        lat0_deg, lon0_deg, alt0_m: origin point in geodetic coordinates
    
    Returns:
        (east_m, north_m, up_m) relative to origin
    """
    x0, y0, z0 = geodetic_to_ecef(
        np.array([lat0_deg]), np.array([lon0_deg]), np.array([alt0_m])
    )
    x0, y0, z0 = x0[0], y0[0], z0[0]
    
    dx = x - x0
    dy = y - y0
    dz = z - z0
    
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)
    
    # Rotation matrix ECEF -> ENU
    east  = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return east, north, up


def geodetic_to_enu(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray,
    lat0_deg: float = None, lon0_deg: float = None, alt0_m: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Converts WGS84 geodetic coordinates to local ENU coordinates.
    Uses the first valid point as origin if not specified.
    
    Returns:
        (east_m, north_m, up_m, origin_info)
    """
    # Find valid points (non-NaN)
    valid_mask = ~(np.isnan(lat_deg) | np.isnan(lon_deg))
    
    if lat0_deg is None:
        first_valid = np.argmax(valid_mask)
        lat0_deg = float(lat_deg[first_valid])
        lon0_deg = float(lon_deg[first_valid])
        alt0_m = float(alt_m[first_valid]) if alt_m is not None else 0.0
    
    if alt_m is None:
        alt_m = np.zeros_like(lat_deg)
    
    origin_info = {
        "lat0_deg": lat0_deg,
        "lon0_deg": lon0_deg,
        "alt0_m": alt0_m,
        "method": "WGS84 -> ECEF -> ENU tangent plane",
        "valid_points": int(valid_mask.sum()),
        "total_points": len(lat_deg),
    }
    
    # Convert
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    east, north, up = ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m)
    
    # Set invalid points to NaN
    east[~valid_mask] = np.nan
    north[~valid_mask] = np.nan
    up[~valid_mask] = np.nan
    
    logger.info(f"Coordinate transform: {origin_info['valid_points']}/{origin_info['total_points']} "
                f"valid points, origin=({lat0_deg:.6f}, {lon0_deg:.6f})")
    
    return east, north, up, origin_info


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Calculates the great-circle distance between two points on Earth using the Haversine formula.
    
    Returns:
        distance in meters
    """
    R = 6371000.0  # Earth's mean radius in meters
    
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def cumulative_distance_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculates cumulative geodesic distance along a GPS trajectory.
    
    Returns:
        Array of cumulative distances in meters (first element is 0).
    """
    valid = ~(np.isnan(lat) | np.isnan(lon))
    
    dists = np.zeros(len(lat))
    for i in range(1, len(lat)):
        if valid[i] and valid[i-1]:
            dists[i] = haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i])
        else:
            dists[i] = 0.0
    
    return np.cumsum(dists)


def enu_cumulative_distance(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Calculates cumulative 2D distance in ENU coordinates."""
    de = np.diff(east)
    dn = np.diff(north)
    
    # Handle NaN
    valid = ~(np.isnan(de) | np.isnan(dn))
    seg_dist = np.zeros_like(de)
    seg_dist[valid] = np.sqrt(de[valid]**2 + dn[valid]**2)
    
    return np.concatenate([[0], np.cumsum(seg_dist)])
