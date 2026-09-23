"""
Phase 7 Candidate Scorer:
Computes probabilistic emission likelihoods combining spatial distance, heading compatibility,
and vehicle dynamic motion compatibility.
Supports parameterization for M0-M5 ablation studies.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .candidate_generator import RoadCandidate
from ..map.geometry import wrap_angle_rad


@dataclass
class ScoringWeights:
    """
    Configuration parameters for candidate scoring.
    """
    sigma_road_m: float = 3.0              # Nominal road centerline / half-width uncertainty (m)
    sigma_geom_rad: float = np.radians(8.0)# Road curve / geometry tangent uncertainty (rad)
    w_distance: float = 1.0                # Weight for distance term in ablation
    w_heading: float = 1.0                 # Weight for heading term in ablation
    w_motion: float = 0.5                  # Weight for motion compatibility
    min_prob_floor: float = 1e-12          # Probability floor to prevent numerical underflow


class CandidateScorer:
    """
    Evaluates probabilistic emission likelihoods for candidate road segments.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()

    def score_candidates(
        self,
        candidates: List[RoadCandidate],
        pos_std_m: float = 5.0,
        yaw_std_rad: float = np.radians(5.0),
        v_forward_mps: float = 0.0,
        is_stationary: bool = False,
    ) -> List[Tuple[RoadCandidate, float, float]]:
        """
        Computes emission likelihood p(z_t | c_i) and log-likelihood for each candidate.

        Parameters:
            candidates: List of RoadCandidate objects
            pos_std_m: Estimated 1-sigma position uncertainty from ESKF (m)
            yaw_std_rad: Estimated 1-sigma yaw uncertainty from ESKF (rad)
            v_forward_mps: Estimated forward vehicle velocity (m/s)
            is_stationary: Whether vehicle is currently stationary (ZUPT active)

        Returns:
            List of (candidate, normalized_probability, log_likelihood)
        """
        if not candidates:
            return []

        # Effective standard deviations incorporating map and estimation uncertainty
        sigma_p_eff = np.sqrt(pos_std_m**2 + self.weights.sigma_road_m**2)

        # If vehicle is stationary, heading is not constrained by forward velocity,
        # so we relax heading variance
        if is_stationary or abs(v_forward_mps) < 0.5:
            sigma_psi_eff = np.sqrt(yaw_std_rad**2 + (2.0 * self.weights.sigma_geom_rad)**2 + np.radians(20.0)**2)
        else:
            sigma_psi_eff = np.sqrt(yaw_std_rad**2 + self.weights.sigma_geom_rad**2)

        log_likes: List[float] = []

        for cand in candidates:
            d = cand.abs_distance_m
            d_psi = cand.heading_residual_rad

            # Spatial distance term
            ll_dist = -0.5 * (d / sigma_p_eff)**2

            # Heading compatibility term
            ll_head = -0.5 * (d_psi / sigma_psi_eff)**2

            # Motion compatibility term
            ll_motion = 0.0
            if cand.segment.speed_limit_mps is not None and v_forward_mps > 1.2 * cand.segment.speed_limit_mps:
                # Slight penalty if speed substantially exceeds speed limit
                ll_motion -= 0.5 * ((v_forward_mps - cand.segment.speed_limit_mps) / 5.0)**2

            total_ll = (
                self.weights.w_distance * ll_dist
                + self.weights.w_heading * ll_head
                + self.weights.w_motion * ll_motion
            )
            log_likes.append(float(total_ll))

        # Softmax normalization with numerical stability
        max_ll = max(log_likes)
        unnorm_probs = [np.exp(ll - max_ll) for ll in log_likes]
        total_prob = sum(unnorm_probs)

        results: List[Tuple[RoadCandidate, float, float]] = []
        for cand, prob, ll in zip(candidates, unnorm_probs, log_likes):
            norm_p = prob / total_prob if total_prob > 0 else 1.0 / len(candidates)
            results.append((cand, float(norm_p), float(ll)))

        # Sort by descending normalized probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results
