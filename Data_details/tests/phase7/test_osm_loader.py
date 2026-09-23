"""
Unit tests for OSMLoader (osm_loader.py).
"""

from pathlib import Path
import numpy as np
import pytest

from Data_details.src.phase7.map.osm_loader import OSMLoader, parse_speed_limit_mps
from Data_details.src.phase7.map.coordinate_mapper import CoordinateMapper


def test_parse_speed_limit():
    assert np.isclose(parse_speed_limit_mps("30 mph"), 30.0 * 0.44704)
    assert np.isclose(parse_speed_limit_mps("50 km/h"), 50.0 / 3.6)
    assert np.isclose(parse_speed_limit_mps("60"), 60.0 / 3.6)
    assert parse_speed_limit_mps(None) is None
    assert parse_speed_limit_mps("unknown") is None


def test_osm_loader_with_real_cached_data():
    osm_path = Path("Data_details/data/osm/coventry_s1_osm.json")
    if not osm_path.exists():
        pytest.skip("Offline OSM file not present.")

    loader = OSMLoader()
    db = loader.load_from_json(osm_path)

    summary = db.summary()
    assert summary["total_segments"] > 10000
    assert summary["total_ways"] > 1000
    assert summary["total_road_length_km"] > 500.0

    # Verify road classes exist
    assert "residential" in summary["road_classes"]
    assert "trunk" in summary["road_classes"]
    assert "tertiary" in summary["road_classes"]

    # Verify connectivity is established
    connected_counts = sum(len(s.connected_segments) for s in db.segments.values())
    assert connected_counts > 0
