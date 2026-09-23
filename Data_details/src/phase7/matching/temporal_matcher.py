"""
Phase 7 Temporal Causal Map Matcher:
Implements recursive Bayesian candidate tracking (HMM / streaming Viterbi style):
    B_t(c_t) \\propto P(z_t | c_t) \\sum_{c_{t-1}} P(c_t | c_{t-1}) B_{t-1}(c_{t-1})
Strictly causal: zero future frames, zero trajectory smoothing at runtime.
Calculates continuous map confidence score c_map in [0, 1] incorporating candidate
dominance, spatial proximity, heading alignment, and topological consistency.
Gracefully deactivates when off-road or unconfident.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .candidate_generator import RoadCandidate, CandidateGenerator
from .candidate_scoring import CandidateScorer
from .topology import TransitionModel


@dataclass
class MapMatchResult:
    """
    Structured output of the Phase 7 Map Matcher compliant with Section 29 of the specification.
    """
    timestamp: float
    matched_segment_id: Optional[str]
    matched_way_id: Optional[int]
    projected_position_enu: np.ndarray      # [East, North] in meters (or NaN if no match)
    road_heading_enu: float                 # Cartesian ENU yaw rad (or 0.0)
    road_heading_gps_deg: float             # Geographic heading deg (or 0.0)
    cross_track_error: float                # Signed cross-track error in meters
    along_track_position: float             # Along-road distance in meters
    candidate_count: int
    candidate_probability: float            # Top candidate posterior belief B(c*)
    map_confidence: float                   # Continuous confidence score in [0.0, 1.0]
    heading_residual: float                 # wrap(psi_vehicle - psi_road) in rad
    distance_residual: float                # Absolute distance to road in meters
    transition_score: float                 # Transition probability P(c_t | c_{t-1})
    accepted: bool                          # Whether match passes confidence gating
    candidate_details: List[Dict[str, Any]] = field(default_factory=list)


class TemporalMapMatcher:
    """
    Causal streaming map matcher maintaining recursive belief state over candidate roads.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        candidate_scorer: CandidateScorer,
        transition_model: TransitionModel,
        min_confidence_threshold: float = 0.35,
        max_acceptable_distance_m: float = 25.0,
        max_active_candidates: int = 10,
    ):
        self.generator = candidate_generator
        self.scorer = candidate_scorer
        self.transition = transition_model
        self.min_confidence_thresh = float(min_confidence_threshold)
        self.max_dist_m = float(max_acceptable_distance_m)
        self.max_active_cands = int(max_active_candidates)

        # Internal state
        self.prev_candidates: Dict[str, RoadCandidate] = {}
        self.prev_beliefs: Dict[str, float] = {}
        self.last_matched_cand: Optional[RoadCandidate] = None
        self.last_position_enu: Optional[np.ndarray] = None
        self.road_switch_count: int = 0

    def reset(self) -> None:
        """Resets temporal tracking state."""
        self.prev_candidates.clear()
        self.prev_beliefs.clear()
        self.last_matched_cand = None
        self.last_position_enu = None
        self.road_switch_count = 0

    def step(
        self,
        timestamp: float,
        p_enu: np.ndarray,
        yaw_enu_rad: float,
        pos_cov_2x2: Optional[np.ndarray] = None,
        yaw_std_rad: float = np.radians(5.0),
        v_forward_mps: float = 0.0,
        is_stationary: bool = False,
    ) -> MapMatchResult:
        """
        Executes one causal map matching cycle at current navigation state.
        """
        p_2d = np.asarray(p_enu, dtype=np.float64)[:2]

        # Traveled distance since last cycle
        delta_dist_m = 0.0
        if self.last_position_enu is not None:
            delta_dist_m = float(np.linalg.norm(p_2d - self.last_position_enu[:2]))
        self.last_position_enu = p_2d.copy()

        # 1. Retrieve candidates
        candidates, search_radius, query_lat_ms = self.generator.generate_candidates(
            p_enu=p_2d,
            yaw_enu_rad=yaw_enu_rad,
            pos_cov_2x2=pos_cov_2x2,
        )

        if not candidates:
            # Off-road / no nearby candidates
            self.prev_candidates.clear()
            self.prev_beliefs.clear()
            return MapMatchResult(
                timestamp=timestamp,
                matched_segment_id=None,
                matched_way_id=None,
                projected_position_enu=np.array([np.nan, np.nan]),
                road_heading_enu=0.0,
                road_heading_gps_deg=0.0,
                cross_track_error=0.0,
                along_track_position=0.0,
                candidate_count=0,
                candidate_probability=0.0,
                map_confidence=0.0,
                heading_residual=0.0,
                distance_residual=np.nan,
                transition_score=0.0,
                accepted=False,
            )

        # Truncate to top-K candidate roads by distance
        active_cands = candidates[:self.max_active_cands]

        # 2. Score emission likelihoods
        pos_std = float(np.sqrt(np.max(np.linalg.eigvalsh(pos_cov_2x2)))) if pos_cov_2x2 is not None else 5.0
        scored_cands = self.scorer.score_candidates(
            candidates=active_cands,
            pos_std_m=pos_std,
            yaw_std_rad=yaw_std_rad,
            v_forward_mps=v_forward_mps,
            is_stationary=is_stationary,
        )

        # 3. Recursive temporal belief update: B_t(c_t) \propto P(z_t | c_t) \sum_{c_{t-1}} P(c_t | c_{t-1}) B_{t-1}(c_{t-1})
        curr_beliefs: Dict[str, float] = {}
        trans_scores: Dict[str, float] = {}

        if self.prev_beliefs and self.prev_candidates:
            total_unnorm = 0.0
            unnorm_dict = {}

            for cand, p_emit, ll in scored_cands:
                # Accumulate transition probabilities from all previous active candidates
                t_score = 0.0
                for prev_id, prev_b in self.prev_beliefs.items():
                    if prev_id in self.prev_candidates:
                        prev_c = self.prev_candidates[prev_id]
                        p_trans = self.transition.transition_probability(prev_c, cand, delta_dist_m)
                        t_score += p_trans * prev_b

                trans_scores[cand.segment_id] = t_score
                unnorm_b = p_emit * t_score
                unnorm_dict[cand.segment_id] = unnorm_b
                total_unnorm += unnorm_b

            if total_unnorm > 0:
                for seg_id, val in unnorm_dict.items():
                    curr_beliefs[seg_id] = val / total_unnorm
            else:
                for cand, p_emit, _ in scored_cands:
                    curr_beliefs[cand.segment_id] = p_emit
                    trans_scores[cand.segment_id] = 1.0 / len(scored_cands)
        else:
            # Initialization at t=0
            for cand, p_emit, _ in scored_cands:
                curr_beliefs[cand.segment_id] = p_emit
                trans_scores[cand.segment_id] = 1.0

        # 4. Identify top candidate
        cand_dict = {c.segment_id: c for c in active_cands}
        sorted_by_belief = sorted(curr_beliefs.items(), key=lambda x: x[1], reverse=True)
        top_id, top_belief = sorted_by_belief[0]
        top_cand = cand_dict[top_id]

        # Second candidate belief (if exists) for margin calculation
        second_belief = sorted_by_belief[1][1] if len(sorted_by_belief) > 1 else 0.0
        belief_margin = max(0.0, top_belief - second_belief)

        # 5. Continuous Map Confidence Score c_map \in [0, 1]
        # Multiplicative terms:
        # a) Belief dominance
        c_belief = top_belief * (0.5 + 0.5 * belief_margin)

        # b) Spatial distance proximity: smooth decay beyond 10m
        d_road = top_cand.abs_distance_m
        c_dist = float(np.exp(-0.5 * (d_road / 12.0)**2))

        # c) Heading alignment: smooth decay beyond 30 deg
        c_heading = float(np.exp(-0.5 * (top_cand.heading_residual_rad / np.radians(25.0))**2))

        map_confidence = float(np.clip(c_belief * c_dist * c_heading, 0.0, 1.0))

        # 6. Gating decision
        accepted = bool(
            map_confidence >= self.min_confidence_thresh
            and d_road <= self.max_dist_m
            and abs(top_cand.heading_residual_rad) <= np.radians(50.0)
        )

        # Track road segment switches
        if self.last_matched_cand is not None and self.last_matched_cand.segment_id != top_cand.segment_id:
            self.road_switch_count += 1
        self.last_matched_cand = top_cand

        # Update temporal state for next step
        self.prev_candidates = {c.segment_id: c for c in active_cands}
        self.prev_beliefs = curr_beliefs

        cand_details = [
            {
                "segment_id": c_id,
                "belief": round(b, 4),
                "distance_m": round(cand_dict[c_id].abs_distance_m, 2),
                "heading_diff_deg": round(float(np.degrees(cand_dict[c_id].heading_residual_rad)), 1),
            }
            for c_id, b in sorted_by_belief[:5]
        ]

        return MapMatchResult(
            timestamp=timestamp,
            matched_segment_id=top_cand.segment_id,
            matched_way_id=top_cand.way_id,
            projected_position_enu=top_cand.projected_point_enu.copy(),
            road_heading_enu=top_cand.road_heading_enu_rad,
            road_heading_gps_deg=top_cand.road_heading_gps_deg,
            cross_track_error=top_cand.cross_track_m,
            along_track_position=top_cand.along_track_m,
            candidate_count=len(active_cands),
            candidate_probability=top_belief,
            map_confidence=map_confidence,
            heading_residual=top_cand.heading_residual_rad,
            distance_residual=top_cand.abs_distance_m,
            transition_score=trans_scores.get(top_id, 1.0),
            accepted=accepted,
            candidate_details=cand_details,
        )
