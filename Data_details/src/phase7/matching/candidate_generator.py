"""
Phase 7 Candidate Generator:
Queries spatial index for candidate road segments within an uncertainty-scaled radius,
computes orthogonal projections, signed cross-track errors, along-track parameters,
and heading residuals.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..map.map_database import RoadSegment, RoadMapDatabase
from ..map.spatial_index import SpatialIndex
from ..map.geometry import (
    project_point_to_segment,
    wrap_angle_rad,
    enu_yaw_to_geographic_azimuth_deg,
    PointToSegmentResult,
)


@dataclass
class RoadCandidate:
    """
    Candidate road segment evaluated against the current vehicle navigation state.
    """
    segment: RoadSegment
    segment_id: str
    way_id: int
    projected_point_enu: np.ndarray      # [East, North] in meters
    cross_track_m: float                 # Signed lateral distance (m)
    abs_distance_m: float                # Euclidean distance to road (m)
    along_track_m: float                 # Distance from start node along road (m)
    fraction: float                      # Segment fraction in [0, 1]
    road_heading_enu_rad: float          # Tangent road heading in Cartesian ENU rad
    road_heading_gps_deg: float          # Road heading in geographic azimuth [0, 360)
    heading_residual_rad: float          # wrap(psi_vehicle - psi_road)
    unit_tangent: np.ndarray             # [t_E, t_N]
    unit_normal: np.ndarray              # [n_E, n_N] (left normal)
    is_clamped: bool                     # Whether projection hit an endpoint


class CandidateGenerator:
    """
    Generates structured RoadCandidate objects from ESKF navigation states.
    """

    def __init__(
        self,
        spatial_index: SpatialIndex,
        k_sigma: float = 3.0,
        d_min_m: float = 15.0,
        d_max_clamp_m: float = 150.0,
    ):
        self.spatial_index = spatial_index
        self.k_sigma = float(k_sigma)
        self.d_min_m = float(d_min_m)
        self.d_max_clamp_m = float(d_max_clamp_m)

    def generate_candidates(
        self,
        p_enu: np.ndarray,
        yaw_enu_rad: float,
        pos_cov_2x2: Optional[np.ndarray] = None,
        max_heading_diff_rad: float = np.radians(70.0),
    ) -> Tuple[List[RoadCandidate], float, float]:
        """
        Retrieves candidate road segments and projects the position onto each.

        Parameters:
            p_enu: Estimated 2D/3D position [East, North, (Up)] in meters
            yaw_enu_rad: Estimated Cartesian ENU yaw angle in radians
            pos_cov_2x2: 2x2 position covariance matrix
            max_heading_diff_rad: Max allowable heading divergence before rejection

        Returns:
            (candidates, search_radius_m, query_latency_ms)
        """
        p_2d = np.asarray(p_enu, dtype=np.float64)[:2]
        segments, radius_m, lat_ms = self.spatial_index.query_uncertainty(
            p_enu_2d=p_2d,
            pos_cov_2x2=pos_cov_2x2,
            k_sigma=self.k_sigma,
            d_min_m=self.d_min_m,
            d_max_clamp_m=self.d_max_clamp_m,
        )

        candidates: List[RoadCandidate] = []
        for seg in segments:
            proj = project_point_to_segment(p_2d, seg.p_start_enu[:2], seg.p_end_enu[:2])

            heading_res = wrap_angle_rad(yaw_enu_rad - proj.segment_heading_enu_rad)

            # Check heading divergence filter: allow reasonable turns and lane changes
            # Note: For bidirectional roads, both forward and reverse directed segments
            # exist in the database, so the vehicle heading will naturally align with
            # the segment traveling in the vehicle's direction!
            if abs(heading_res) > max_heading_diff_rad:
                # If diverging too much, we don't immediately drop if radius is small,
                # but we record it for likelihood evaluation. However, if diverging > 90 deg,
                # the vehicle is driving backwards relative to this directed segment.
                if abs(heading_res) > np.radians(95.0):
                    continue

            cand = RoadCandidate(
                segment=seg,
                segment_id=seg.segment_id,
                way_id=seg.way_id,
                projected_point_enu=proj.projected_point_enu,
                cross_track_m=proj.cross_track_m,
                abs_distance_m=proj.abs_distance_m,
                along_track_m=proj.along_track_m,
                fraction=proj.fraction,
                road_heading_enu_rad=proj.segment_heading_enu_rad,
                road_heading_gps_deg=enu_yaw_to_geographic_azimuth_deg(proj.segment_heading_enu_rad),
                heading_residual_rad=heading_res,
                unit_tangent=proj.unit_tangent,
                unit_normal=proj.unit_normal,
                is_clamped=proj.is_clamped,
            )
            candidates.append(cand)

        # Sort candidates by distance
        candidates.sort(key=lambda c: c.abs_distance_m)
        return candidates, radius_m, lat_ms
