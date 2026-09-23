"""
Phase 7 Coordinate Mapper:
Handles conversion between WGS84 Geodetic coordinates (Lat, Lon, Alt) and
local metric East-North-Up (ENU) coordinates.
Directly reuses Data_details.src.coordinate_transform to guarantee exact mathematical
consistency with Phase 5 and Phase 6 navigation states.
"""

from typing import Tuple, Union, Optional, Dict, Any
import numpy as np

from Data_details.src.coordinate_transform import (
    geodetic_to_ecef,
    ecef_to_enu,
    geodetic_to_enu,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    WGS84_F,
)


class CoordinateMapper:
    """
    Manages coordinate transformations relative to a fixed local geodetic origin.
    Default origin is calibrated to the Coventry S1 reference entry:
    lat0 = 52.401660 deg, lon0 = -1.505290 deg, alt0 = 147.5 m.
    """

    def __init__(
        self,
        lat0_deg: float = 52.401660,
        lon0_deg: float = -1.505290,
        alt0_m: float = 147.5,
    ):
        self.lat0_deg = float(lat0_deg)
        self.lon0_deg = float(lon0_deg)
        self.alt0_m = float(alt0_m)

        # Precompute ECEF origin
        x0, y0, z0 = geodetic_to_ecef(
            np.array([self.lat0_deg]), np.array([self.lon0_deg]), np.array([self.alt0_m])
        )
        self.x0 = float(x0[0])
        self.y0 = float(y0[0])
        self.z0 = float(z0[0])

        lat0_rad = np.radians(self.lat0_deg)
        lon0_rad = np.radians(self.lon0_deg)
        self.sin_lat0 = np.sin(lat0_rad)
        self.cos_lat0 = np.cos(lat0_rad)
        self.sin_lon0 = np.sin(lon0_rad)
        self.cos_lon0 = np.cos(lon0_rad)

        # Rotation matrix ECEF -> ENU
        # [e, n, u]^T = R_ecef_to_enu * [dx, dy, dz]^T
        self.R_ecef_to_enu = np.array([
            [-self.sin_lon0, self.cos_lon0, 0.0],
            [-self.sin_lat0 * self.cos_lon0, -self.sin_lat0 * self.sin_lon0, self.cos_lat0],
            [self.cos_lat0 * self.cos_lon0, self.cos_lat0 * self.sin_lon0, self.sin_lat0],
        ], dtype=np.float64)

        # Inverse rotation ENU -> ECEF
        self.R_enu_to_ecef = self.R_ecef_to_enu.T

    def geodetic_to_enu(
        self,
        lat_deg: Union[float, np.ndarray],
        lon_deg: Union[float, np.ndarray],
        alt_m: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts WGS84 Geodetic coordinates to local metric ENU coordinates.
        """
        lat_arr = np.atleast_1d(np.asarray(lat_deg, dtype=np.float64))
        lon_arr = np.atleast_1d(np.asarray(lon_deg, dtype=np.float64))
        if alt_m is None:
            alt_arr = np.zeros_like(lat_arr)
        else:
            alt_arr = np.atleast_1d(np.asarray(alt_m, dtype=np.float64))

        e, n, u, _ = geodetic_to_enu(
            lat_arr, lon_arr, alt_arr,
            lat0_deg=self.lat0_deg, lon0_deg=self.lon0_deg, alt0_m=self.alt0_m
        )

        if np.isscalar(lat_deg) and np.isscalar(lon_deg):
            return float(e[0]), float(n[0]), float(u[0])
        return e, n, u

    def enu_to_ecef(
        self,
        east_m: Union[float, np.ndarray],
        north_m: Union[float, np.ndarray],
        up_m: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts local ENU coordinates to ECEF coordinates.
        """
        e = np.atleast_1d(np.asarray(east_m, dtype=np.float64))
        n = np.atleast_1d(np.asarray(north_m, dtype=np.float64))
        u = np.zeros_like(e) if up_m is None else np.atleast_1d(np.asarray(up_m, dtype=np.float64))

        enu_stack = np.column_stack([e, n, u])  # (N, 3)
        dx_dy_dz = enu_stack @ self.R_ecef_to_enu  # using Transpose: (R.T @ v)^T = v^T @ R

        x = dx_dy_dz[:, 0] + self.x0
        y = dx_dy_dz[:, 1] + self.y0
        z = dx_dy_dz[:, 2] + self.z0

        if np.isscalar(east_m) and np.isscalar(north_m):
            return float(x[0]), float(y[0]), float(z[0])
        return x, y, z

    def enu_to_geodetic(
        self,
        east_m: Union[float, np.ndarray],
        north_m: Union[float, np.ndarray],
        up_m: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts local ENU coordinates back to WGS84 Geodetic (Bowring's method).
        """
        x, y, z = self.enu_to_ecef(east_m, north_m, up_m)
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)
        z_arr = np.atleast_1d(z)

        p = np.sqrt(x_arr**2 + y_arr**2)
        theta = np.arctan2(z_arr * WGS84_A, p * WGS84_B)

        e_prime_sq = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)
        lat = np.arctan2(
            z_arr + e_prime_sq * WGS84_B * np.sin(theta)**3,
            p - WGS84_E2 * WGS84_A * np.cos(theta)**3
        )
        lon = np.arctan2(y_arr, x_arr)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N

        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon)

        if np.isscalar(east_m) and np.isscalar(north_m):
            return float(lat_deg[0]), float(lon_deg[0]), float(alt[0])
        return lat_deg, lon_deg, alt

    @property
    def origin_info(self) -> Dict[str, Any]:
        return {
            "lat0_deg": self.lat0_deg,
            "lon0_deg": self.lon0_deg,
            "alt0_m": self.alt0_m,
            "ecef_origin": (self.x0, self.y0, self.z0),
        }
