"""
Unit tests for Phase 8 WGS-84, ECEF, ENU coordinate transformations and heading conversions.
"""

import numpy as np
import pytest

from Data_details.src.phase8.gnss.coordinate import (
    geodetic_to_ecef,
    ecef_to_enu,
    geodetic_to_enu,
    enu_to_ecef,
    ecef_to_geodetic,
    enu_to_geodetic,
    gps_bearing_to_enu_yaw_rad,
    enu_yaw_to_gps_bearing_deg,
    gps_vel_to_enu_velocity,
)


def test_wgs84_geodetic_to_ecef_and_back():
    """Verifies round-trip Geodetic -> ECEF -> Geodetic precision is sub-millimeter."""
    test_coords = [
        (12.9716, 77.5946, 920.0),   # Bangalore, India
        (0.0, 0.0, 0.0),             # Prime Meridian / Equator
        (45.0, -93.0, 250.0),        # Mid-latitude
        (-33.8688, 151.2093, 50.0),  # Sydney, Australia
    ]

    for lat, lon, alt in test_coords:
        p_ecef = geodetic_to_ecef(lat, lon, alt)
        assert np.all(np.isfinite(p_ecef))

        lat_rec, lon_rec, alt_rec = ecef_to_geodetic(p_ecef)
        assert np.isclose(lat, lat_rec, atol=1e-8), f"Latitude mismatch: {lat} vs {lat_rec}"
        assert np.isclose(lon, lon_rec, atol=1e-8), f"Longitude mismatch: {lon} vs {lon_rec}"
        assert np.isclose(alt, alt_rec, atol=1e-3), f"Altitude mismatch: {alt} vs {alt_rec}"


def test_enu_round_trip():
    """Verifies round-trip ENU -> Geodetic -> ENU precision."""
    ref_lat, ref_lon, ref_alt = 12.9716, 77.5946, 920.0

    p_orig = np.array([1000.0, 2000.0, 15.0], dtype=np.float64)

    # ENU -> Geodetic
    lat, lon, alt = enu_to_geodetic(p_orig, ref_lat, ref_lon, ref_alt)

    # Geodetic -> ENU
    p_rec = geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

    assert np.allclose(p_orig, p_rec, atol=1e-3), f"ENU round trip failed: {p_orig} vs {p_rec}"


def test_heading_conversions():
    """Verifies heading wrap conventions between GPS bearing and Cartesian ENU yaw."""
    # North: 0 deg bearing -> +pi/2 ENU yaw
    yaw_north = gps_bearing_to_enu_yaw_rad(0.0)
    assert np.isclose(yaw_north, np.pi / 2.0, atol=1e-6)
    bearing_north = enu_yaw_to_gps_bearing_deg(yaw_north)
    assert np.isclose(bearing_north, 0.0, atol=1e-6)

    # East: 90 deg bearing -> 0 ENU yaw
    yaw_east = gps_bearing_to_enu_yaw_rad(90.0)
    assert np.isclose(yaw_east, 0.0, atol=1e-6)
    bearing_east = enu_yaw_to_gps_bearing_deg(yaw_east)
    assert np.isclose(bearing_east, 90.0, atol=1e-6)

    # South: 180 deg bearing -> -pi/2 ENU yaw
    yaw_south = gps_bearing_to_enu_yaw_rad(180.0)
    assert np.isclose(yaw_south, -np.pi / 2.0, atol=1e-6)
    bearing_south = enu_yaw_to_gps_bearing_deg(yaw_south)
    assert np.isclose(bearing_south, 180.0, atol=1e-6)

    # West: 270 deg bearing -> pi or -pi ENU yaw
    yaw_west = gps_bearing_to_enu_yaw_rad(270.0)
    assert np.isclose(np.abs(yaw_west), np.pi, atol=1e-6)
    bearing_west = enu_yaw_to_gps_bearing_deg(yaw_west)
    assert np.isclose(bearing_west, 270.0, atol=1e-6)


def test_gps_velocity_to_enu():
    """Verifies speed and bearing conversion to 3D ENU velocity."""
    # Driving East at 20 m/s
    v_east = gps_vel_to_enu_velocity(speed_mps=20.0, bearing_deg=90.0)
    assert np.isclose(v_east[0], 20.0, atol=1e-6)
    assert np.isclose(v_east[1], 0.0, atol=1e-6)
    assert np.isclose(v_east[2], 0.0, atol=1e-6)

    # Driving North at 15 m/s
    v_north = gps_vel_to_enu_velocity(speed_mps=15.0, bearing_deg=0.0)
    assert np.isclose(v_north[0], 0.0, atol=1e-6)
    assert np.isclose(v_north[1], 15.0, atol=1e-6)
