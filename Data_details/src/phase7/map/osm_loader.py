"""
Phase 7 OSM Loader:
Parses offline OpenStreetMap data (JSON format from Overpass/Osmosis),
converts node coordinates into the unified local ENU tangent plane,
extracts drivable road segments, parses metadata (oneway, maxspeed, road class),
and constructs the structured RoadMapDatabase.
100% offline, deterministic, zero live internet dependencies.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import numpy as np

from .geometry import (
    compute_segment_azimuth_enu,
    enu_yaw_to_geographic_azimuth_deg,
)
from .coordinate_mapper import CoordinateMapper
from .map_database import RoadSegment, RoadMapDatabase

logger = logging.getLogger(__name__)

# Standard drivable highway types
VALID_HIGHWAYS: Set[str] = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}


def parse_speed_limit_mps(speed_str: Optional[str]) -> Optional[float]:
    """
    Parses OSM maxspeed tag strings into meters per second.
    Examples: '30 mph' -> 13.41 m/s, '50 km/h' or '50' -> 13.89 m/s.
    Returns None if missing, unparseable, or invalid.
    """
    if not speed_str or not isinstance(speed_str, str):
        return None

    s = speed_str.strip().lower()
    # Check mph
    m_mph = re.match(r"^(\d+(?:\.\d+)?)\s*mph$", s)
    if m_mph:
        return float(m_mph.group(1)) * 0.44704

    # Check km/h or bare number
    m_kmh = re.match(r"^(\d+(?:\.\d+)?)(?:\s*km/?h)?$", s)
    if m_kmh:
        return float(m_kmh.group(1)) / 3.6

    return None


class OSMLoader:
    """
    Loads offline OSM JSON exports into a RoadMapDatabase.
    """

    def __init__(
        self,
        coordinate_mapper: Optional[CoordinateMapper] = None,
        valid_highways: Optional[Set[str]] = None,
        min_segment_length_m: float = 0.5,
    ):
        self.coord_mapper = coordinate_mapper or CoordinateMapper()
        self.valid_highways = valid_highways or VALID_HIGHWAYS
        self.min_segment_length_m = float(min_segment_length_m)

    def load_from_json(self, json_path: Union[str, Path], db_name: str = "coventry_s1") -> RoadMapDatabase:
        """
        Loads an OSM JSON file and returns a populated RoadMapDatabase.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Offline OSM file not found: {path}")

        logger.info(f"Loading offline OSM road network from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        elements = data.get("elements", [])
        return self.process_elements(elements, db_name=db_name)

    def process_elements(self, elements: List[Dict[str, Any]], db_name: str = "coventry_s1") -> RoadMapDatabase:
        """
        Processes elements list (nodes and ways) from OSM JSON structure.
        """
        nodes_dict: Dict[int, Tuple[float, float, float]] = {}  # node_id -> (lat, lon)
        ways_list: List[Dict[str, Any]] = []

        for elem in elements:
            e_type = elem.get("type")
            if e_type == "node":
                nid = elem["id"]
                lat = float(elem["lat"])
                lon = float(elem["lon"])
                nodes_dict[nid] = (lat, lon)
            elif e_type == "way":
                ways_list.append(elem)

        logger.info(f"Parsed {len(nodes_dict)} nodes and {len(ways_list)} ways from OSM source.")

        # Batch convert all nodes to local ENU
        node_ids = list(nodes_dict.keys())
        if node_ids:
            lats = np.array([nodes_dict[nid][0] for nid in node_ids], dtype=np.float64)
            lons = np.array([nodes_dict[nid][1] for nid in node_ids], dtype=np.float64)
            east, north, up = self.coord_mapper.geodetic_to_enu(lats, lons)
            node_enu: Dict[int, np.ndarray] = {
                nid: np.array([east[i], north[i], up[i]], dtype=np.float64)
                for i, nid in enumerate(node_ids)
            }
        else:
            node_enu = {}

        db = RoadMapDatabase(name=db_name)
        total_created = 0

        for way in ways_list:
            tags = way.get("tags", {})
            hw_class = tags.get("highway")
            if not hw_class or hw_class not in self.valid_highways:
                continue

            way_id = way["id"]
            node_refs = way.get("nodes", [])
            if len(node_refs) < 2:
                continue

            # Parse one-way attribute
            oneway_tag = tags.get("oneway", "no").strip().lower()
            is_oneway = oneway_tag in ("yes", "1", "true")
            is_reverse_oneway = oneway_tag in ("-1", "reverse")

            # Parse metadata
            speed_limit = parse_speed_limit_mps(tags.get("maxspeed"))
            road_name = tags.get("name")

            # Build segments between consecutive nodes in the way
            for idx in range(len(node_refs) - 1):
                u_id = node_refs[idx]
                v_id = node_refs[idx + 1]

                if u_id not in node_enu or v_id not in node_enu:
                    continue

                p_u = node_enu[u_id]
                p_v = node_enu[v_id]
                length = float(np.linalg.norm(p_v[:2] - p_u[:2]))

                if length < self.min_segment_length_m:
                    continue  # Ignore tiny micro-segments

                # Forward segment: u -> v
                heading_rad = compute_segment_azimuth_enu(p_u[:2], p_v[:2])
                heading_gps = enu_yaw_to_geographic_azimuth_deg(heading_rad)

                if not is_reverse_oneway:
                    seg_fwd = RoadSegment(
                        segment_id=f"{way_id}_{idx}_fwd",
                        way_id=way_id,
                        sub_idx=idx,
                        start_node=u_id,
                        end_node=v_id,
                        p_start_enu=p_u.copy(),
                        p_end_enu=p_v.copy(),
                        length_m=length,
                        heading_start_rad=heading_rad,
                        heading_end_rad=heading_rad,
                        heading_rad=heading_rad,
                        heading_gps_deg=heading_gps,
                        road_class=hw_class,
                        one_way=is_oneway,
                        speed_limit_mps=speed_limit,
                        road_name=road_name,
                    )
                    db.add_segment(seg_fwd)
                    total_created += 1

                # If bidirectional or reverse one-way, create reverse segment: v -> u
                if not is_oneway or is_reverse_oneway:
                    rev_heading_rad = compute_segment_azimuth_enu(p_v[:2], p_u[:2])
                    rev_heading_gps = enu_yaw_to_geographic_azimuth_deg(rev_heading_rad)

                    seg_rev = RoadSegment(
                        segment_id=f"{way_id}_{idx}_rev",
                        way_id=way_id,
                        sub_idx=idx,
                        start_node=v_id,
                        end_node=u_id,
                        p_start_enu=p_v.copy(),
                        p_end_enu=p_u.copy(),
                        length_m=length,
                        heading_start_rad=rev_heading_rad,
                        heading_end_rad=rev_heading_rad,
                        heading_rad=rev_heading_rad,
                        heading_gps_deg=rev_heading_gps,
                        road_class=hw_class,
                        one_way=False,
                        speed_limit_mps=speed_limit,
                        road_name=road_name,
                    )
                    db.add_segment(seg_rev)
                    total_created += 1

        db.build_connectivity()
        logger.info(f"Constructed RoadMapDatabase with {total_created} directed segments across {len(db.way_to_segments)} ways.")
        return db
