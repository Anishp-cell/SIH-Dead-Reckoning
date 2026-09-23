"""
Unit tests for CoordinateMapper (coordinate_mapper.py).
"""

import numpy as np
import pytest

from Data_details.src.phase7.map.coordinate_mapper import CoordinateMapper


def test_coordinate_mapper_origin():
    mapper = CoordinateMapper(lat0_deg=52.401660, lon0_deg=-1.505290, alt0_m=147.5)
    e, n, u = mapper.geodetic_to_enu(52.401660, -1.505290, 147.5)
    assert np.isclose(e, 0.0, atol=1e-4)
    assert np.isclose(n, 0.0, atol=1e-4)
    assert np.isclose(u, 0.0, atol=1e-4)


def test_coordinate_mapper_roundtrip():
    mapper = CoordinateMapper(lat0_deg=52.401660, lon0_deg=-1.505290, alt0_m=147.5)
    test_lat = 52.410000
    test_lon = -1.520000
    test_alt = 160.0

    e, n, u = mapper.geodetic_to_enu(test_lat, test_lon, test_alt)
    lat_rec, lon_rec, alt_rec = mapper.enu_to_geodetic(e, n, u)

    assert np.isclose(test_lat, lat_rec, atol=1e-7)
    assert np.isclose(test_lon, lon_rec, atol=1e-7)
    assert np.isclose(test_alt, alt_rec, atol=1e-2)


def test_coordinate_mapper_vectorized():
    mapper = CoordinateMapper()
    lats = np.array([52.401660, 52.405000, 52.410000])
    lons = np.array([-1.505290, -1.510000, -1.520000])
    e, n, u = mapper.geodetic_to_enu(lats, lons)

    assert len(e) == 3
    assert np.isclose(e[0], 0.0, atol=1e-4)
    assert np.isclose(n[0], 0.0, atol=1e-4)
