"""
Phase 7 Road Map Database:
Implements a compact, structured offline road database compliant with Section 7 of the Phase 7 specification.
Stores directed road segments, topological connectivity, road classifications, and physical geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class RoadSegment:
    """
    Representation of a directed road segment connecting two adjacent nodes.
    For two-way roads, forward and reverse segments are created explicitly to
    preserve strict directionality in topological transitions and heading compatibility.
    """
    segment_id: str                   # Unique identifier (e.g., "way1234_0_fwd")
    way_id: int                       # Parent OSM Way ID
    sub_idx: int                      # Index of this segment within parent way
    start_node: int                   # OSM Node ID of start vertex
    end_node: int                     # OSM Node ID of end vertex
    p_start_enu: np.ndarray           # [East, North, Up] in meters
    p_end_enu: np.ndarray             # [East, North, Up] in meters
    length_m: float                   # Segment length in meters
    heading_start_rad: float          # Tangent heading at start in Cartesian ENU rad
    heading_end_rad: float            # Tangent heading at end in Cartesian ENU rad
    heading_rad: float                # Overall segment heading in Cartesian ENU rad
    heading_gps_deg: float            # Segment heading in geographic azimuth [0, 360)
    road_class: str                   # OSM highway class ('primary', 'secondary', etc.)
    one_way: bool                     # Whether this segment is part of a one-way way
    speed_limit_mps: Optional[float] = None  # Speed limit in m/s if available, else None
    road_name: Optional[str] = None          # Road name from OSM tags if available
    connected_segments: List[str] = field(default_factory=list)  # Successor segment IDs

    @property
    def midpoint_enu(self) -> np.ndarray:
        """Midpoint of segment in local ENU."""
        return 0.5 * (self.p_start_enu + self.p_end_enu)


class RoadMapDatabase:
    """
    In-memory offline database of road segments with topological indexing.
    Supports deterministic serialization, querying, and summary diagnostics.
    """

    def __init__(self, name: str = "coventry_s1"):
        self.name = name
        self.segments: Dict[str, RoadSegment] = {}
        self.node_outgoing: Dict[int, List[str]] = {}
        self.node_incoming: Dict[int, List[str]] = {}
        self.way_to_segments: Dict[int, List[str]] = {}
        self._bbox_enu: Optional[Tuple[float, float, float, float]] = None

    def add_segment(self, segment: RoadSegment) -> None:
        """Adds a road segment to the database and registers its nodes."""
        self.segments[segment.segment_id] = segment

        # Register outgoing from start_node
        if segment.start_node not in self.node_outgoing:
            self.node_outgoing[segment.start_node] = []
        self.node_outgoing[segment.start_node].append(segment.segment_id)

        # Register incoming to end_node
        if segment.end_node not in self.node_incoming:
            self.node_incoming[segment.end_node] = []
        self.node_incoming[segment.end_node].append(segment.segment_id)

        # Register way
        if segment.way_id not in self.way_to_segments:
            self.way_to_segments[segment.way_id] = []
        self.way_to_segments[segment.way_id].append(segment.segment_id)

        self._bbox_enu = None  # Invalidate cached bbox

    def build_connectivity(self) -> None:
        """
        Builds topological connectivity graph:
        For each segment, connected_segments = all outgoing segments from its end_node.
        """
        for seg in self.segments.values():
            successors = self.node_outgoing.get(seg.end_node, [])
            # Filter out immediate U-turn on the exact same two-way reverse segment if desired,
            # or keep all topologically valid connected edges
            seg.connected_segments = list(successors)

    def get_segment(self, segment_id: str) -> Optional[RoadSegment]:
        return self.segments.get(segment_id)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def total_road_length_km(self) -> float:
        return sum(s.length_m for s in self.segments.values()) / 1000.0

    @property
    def bbox_enu(self) -> Tuple[float, float, float, float]:
        """Returns (min_east, max_east, min_north, max_north) bounding box."""
        if self._bbox_enu is None and len(self.segments) > 0:
            e_coords = []
            n_coords = []
            for s in self.segments.values():
                e_coords.extend([s.p_start_enu[0], s.p_end_enu[0]])
                n_coords.extend([s.p_start_enu[1], s.p_end_enu[1]])
            self._bbox_enu = (
                float(min(e_coords)), float(max(e_coords)),
                float(min(n_coords)), float(max(n_coords))
            )
        return self._bbox_enu or (0.0, 0.0, 0.0, 0.0)

    def summary(self) -> Dict[str, Any]:
        """Returns summary statistics for the road database."""
        class_counts = {}
        for s in self.segments.values():
            c = s.road_class
            class_counts[c] = class_counts.get(c, 0) + 1

        b = self.bbox_enu
        return {
            "name": self.name,
            "total_segments": self.total_segments,
            "total_ways": len(self.way_to_segments),
            "total_nodes": len(self.node_outgoing),
            "total_road_length_km": round(self.total_road_length_km, 2),
            "bbox_enu": {
                "east_min": round(b[0], 2), "east_max": round(b[1], 2),
                "north_min": round(b[2], 2), "north_max": round(b[3], 2),
            },
            "road_classes": class_counts,
        }
