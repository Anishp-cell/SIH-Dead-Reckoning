"""
Phase 7 Road Topology Graph and Transition Model:
Constructs directed topological connectivity graph over RoadMapDatabase segments.
Calculates Markovian transition probabilities P(c_t | c_{t-1}) favoring continuous
along-road driving and connected successor segments, while heavily penalizing
unconnected jumps and parallel road switches.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import networkx as nx

from ..map.map_database import RoadMapDatabase, RoadSegment
from .candidate_generator import RoadCandidate


class RoadTopologyGraph:
    """
    Directed graph representation of road segment connectivity.
    """

    def __init__(self, road_db: RoadMapDatabase):
        self.road_db = road_db
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Constructs networkx directed graph."""
        for seg_id, seg in self.road_db.segments.items():
            self.graph.add_node(seg_id, length_m=seg.length_m, road_class=seg.road_class)

        for seg_id, seg in self.road_db.segments.items():
            for succ_id in seg.connected_segments:
                if succ_id in self.road_db.segments:
                    succ_seg = self.road_db.segments[succ_id]
                    self.graph.add_edge(seg_id, succ_id, weight=succ_seg.length_m)

    def is_connected(self, seg_from_id: str, seg_to_id: str) -> bool:
        """Checks if seg_to is an immediate topological successor of seg_from."""
        return self.graph.has_edge(seg_from_id, seg_to_id)

    def successors(self, seg_id: str) -> List[str]:
        """Returns list of outgoing connected segment IDs."""
        if seg_id in self.graph:
            return list(self.graph.successors(seg_id))
        return []

    def shortest_path_distance(self, seg_from_id: str, seg_to_id: str, cutoff_m: float = 200.0) -> Optional[float]:
        """
        Computes network shortest path distance between segments, or None if disconnected.
        """
        if seg_from_id == seg_to_id:
            return 0.0
        try:
            dist = nx.shortest_path_length(self.graph, source=seg_from_id, target=seg_to_id, weight="weight")
            return float(dist) if dist <= cutoff_m else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


class TransitionModel:
    """
    Computes transition probabilities P(c_t | c_{t-1}) based on topological connectivity
    and vehicle displacement consistency.
    """

    def __init__(
        self,
        topology_graph: RoadTopologyGraph,
        p_stay: float = 0.85,
        p_connected: float = 0.14,
        p_jump_base: float = 1e-4,
        jump_decay_m: float = 15.0,
    ):
        self.topo = topology_graph
        self.p_stay = float(p_stay)
        self.p_connected = float(p_connected)
        self.p_jump_base = float(p_jump_base)
        self.jump_decay_m = float(jump_decay_m)

    def transition_probability(
        self,
        prev_cand: RoadCandidate,
        curr_cand: RoadCandidate,
        delta_dist_m: float = 1.0,
    ) -> float:
        """
        Computes transition probability P(curr | prev).

        Parameters:
            prev_cand: Candidate at time t-1
            curr_cand: Candidate at time t
            delta_dist_m: Vehicle traveled distance between time steps

        Returns:
            Scalar probability P(curr | prev) > 0
        """
        prev_id = prev_cand.segment_id
        curr_id = curr_cand.segment_id

        # Case 1: Continuing on the same road segment
        if prev_id == curr_id:
            # Check along-track progress consistency
            along_delta = curr_cand.along_track_m - prev_cand.along_track_m
            # Forward motion is expected (allow small backward jitter up to 1.5m due to noise)
            if -1.5 <= along_delta <= delta_dist_m + 5.0:
                return self.p_stay
            else:
                # Unexpected jump along same segment
                return self.p_stay * np.exp(-abs(along_delta - delta_dist_m) / self.jump_decay_m)

        # Case 2: Transitioning to an immediate topological successor segment
        if self.topo.is_connected(prev_id, curr_id):
            # Vehicle should be near end of previous segment (e.g. fraction > 0.6)
            # and near start of new segment (fraction < 0.4)
            return self.p_connected

        # Case 3: Disconnected jump (e.g. parallel road, wrong branch, or lost track)
        # Compute Euclidean distance between segment endpoints
        p_prev_end = prev_cand.segment.p_end_enu[:2]
        p_curr_start = curr_cand.segment.p_start_enu[:2]
        gap_m = float(np.linalg.norm(p_curr_start - p_prev_end))

        # Heavily penalize topological jumps
        prob = self.p_jump_base * np.exp(-gap_m / self.jump_decay_m)
        return max(1e-12, float(prob))
