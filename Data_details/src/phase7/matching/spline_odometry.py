"""
Phase 7/8 1D Curvilinear Road-Spline Odometry Engine:
Tracks along-track arc length s(t) = s_0 + ∫ v_AI(t) dt along the active OSM road polyline.
Solves the along-track unobservability problem during GNSS blackouts by projecting
neural forward speed predictions strictly along the road tangent geometry.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from ..map.map_database import RoadSegment
from ..map.geometry import project_point_to_segment, wrap_angle_rad
from .temporal_matcher import MapMatchResult


@dataclass
class SplineOdometryConfig:
    """
    Configuration parameters for road-spline along-track odometry.
    """
    sigma_along_track_base_m: float = 1.0       # Base along-track std dev
    sigma_v_growth_rate: float = 0.05           # Error growth per second (m/s)
    max_along_track_innovation_m: float = 8.0   # Innovation clamping threshold
    min_confidence_for_spline: float = 0.30     # Minimum map match confidence to apply
    speed_clamp_min_mps: float = 0.0            # No reverse motion on forward roads
    speed_clamp_max_factor: float = 1.15        # Maximum speed factor above OSM limit


class RoadSplineOdometry:
    """
    Maintains causal along-track arc length integration along OSM road polylines.
    """

    def __init__(self, config: Optional[SplineOdometryConfig] = None):
        self.cfg = config or SplineOdometryConfig()
        self.is_active: bool = False
        self.active_segment_id: Optional[str] = None
        self.s_segment_m: float = 0.0           # Cumulative arc length along current segment
        self.s_total_m: float = 0.0             # Cumulative arc length since blackout start
        self.last_timestamp: Optional[float] = None
        self.accumulated_var_m2: float = 0.25    # Initial position variance floor (0.5m)^2

    def reset(self) -> None:
        """Resets the spline odometry state."""
        self.is_active = False
        self.active_segment_id = None
        self.s_segment_m = 0.0
        self.s_total_m = 0.0
        self.last_timestamp = None
        self.accumulated_var_m2 = 0.25

    def start_outage(
        self,
        initial_match: Optional[MapMatchResult],
        timestamp: float,
    ) -> None:
        """
        Initializes along-track tracking at the onset of a GNSS outage.
        """
        self.reset()
        self.last_timestamp = timestamp
        self.is_active = True
        if initial_match is not None and initial_match.matched_segment_id is not None:
            self.active_segment_id = initial_match.matched_segment_id
            self.s_segment_m = float(initial_match.along_track_position)
        else:
            self.active_segment_id = None
            self.s_segment_m = 0.0

    def step(
        self,
        v_forward_ai_mps: float,
        sigma_v_mps: float,
        timestamp: float,
        match: Optional[MapMatchResult],
        is_stationary: bool = False,
        dt_override: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """
        Advances along-track odometry by one step:
        Δs = v_forward_ai * Δt (if not stationary).
        
        Returns:
            (target_along_m, variance_along_m2, is_valid)
        """
        if not self.is_active:
            return 0.0, 1e6, False

        if dt_override is not None:
            dt = dt_override
        elif self.last_timestamp is not None:
            dt = max(1e-4, timestamp - self.last_timestamp)
        else:
            dt = 0.10

        self.last_timestamp = timestamp

        # Apply speed bounds
        v_clean = max(self.cfg.speed_clamp_min_mps, float(v_forward_ai_mps))
        if is_stationary:
            v_clean = 0.0

        delta_s = v_clean * dt
        self.s_total_m += delta_s

        # Variance accumulation: var += (sigma_v * dt)^2
        step_var = max(0.01, (sigma_v_mps * dt)**2)
        self.accumulated_var_m2 += step_var

        # Handle road segment transitions
        if match is not None and match.matched_segment_id is not None and match.accepted:
            seg_id = match.matched_segment_id
            if self.active_segment_id is None:
                self.active_segment_id = seg_id
                self.s_segment_m = float(match.along_track_position) + delta_s
            elif seg_id == self.active_segment_id:
                self.s_segment_m += delta_s
            else:
                # Transitioned to adjacent segment
                self.active_segment_id = seg_id
                self.s_segment_m = max(0.0, float(match.along_track_position) + delta_s)
            
            return self.s_segment_m, self.accumulated_var_m2, True

        return self.s_total_m, self.accumulated_var_m2, False
