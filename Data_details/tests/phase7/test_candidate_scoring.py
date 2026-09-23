"""
Unit tests for CandidateScorer (candidate_scoring.py).
"""

import numpy as np
import pytest

from Data_details.src.phase7.map.map_database import RoadSegment
from Data_details.src.phase7.matching.candidate_generator import RoadCandidate
from Data_details.src.phase7.matching.candidate_scoring import CandidateScorer, ScoringWeights


def make_dummy_candidate(seg_id: str, dist_m: float, heading_diff_rad: float) -> RoadCandidate:
    seg = RoadSegment(
        segment_id=seg_id,
        way_id=100,
        sub_idx=0,
        start_node=1,
        end_node=2,
        p_start_enu=np.array([0.0, 0.0, 0.0]),
        p_end_enu=np.array([100.0, 0.0, 0.0]),
        length_m=100.0,
        heading_start_rad=0.0,
        heading_end_rad=0.0,
        heading_rad=0.0,
        heading_gps_deg=90.0,
        road_class="residential",
        one_way=True,
    )
    return RoadCandidate(
        segment=seg,
        segment_id=seg_id,
        way_id=100,
        projected_point_enu=np.array([50.0, 0.0]),
        cross_track_m=dist_m,
        abs_distance_m=abs(dist_m),
        along_track_m=50.0,
        fraction=0.5,
        road_heading_enu_rad=0.0,
        road_heading_gps_deg=90.0,
        heading_residual_rad=heading_diff_rad,
        unit_tangent=np.array([1.0, 0.0]),
        unit_normal=np.array([0.0, 1.0]),
        is_clamped=False,
    )


def test_candidate_scoring_proximity_preference():
    scorer = CandidateScorer()
    # cand1 is close (2m), cand2 is far (20m), same heading alignment
    c1 = make_dummy_candidate("c1", dist_m=2.0, heading_diff_rad=0.0)
    c2 = make_dummy_candidate("c2", dist_m=20.0, heading_diff_rad=0.0)

    scored = scorer.score_candidates([c1, c2], pos_std_m=3.0, yaw_std_rad=0.05)
    assert len(scored) == 2
    # c1 must have significantly higher probability than c2
    assert scored[0][0].segment_id == "c1"
    assert scored[0][1] > 0.90
    assert scored[1][1] < 0.10


def test_candidate_scoring_heading_preference():
    scorer = CandidateScorer()
    # Both at same distance (3m), but c1 is well aligned (0 rad), c2 has 40 deg misalignment
    c1 = make_dummy_candidate("c1", dist_m=3.0, heading_diff_rad=0.0)
    c2 = make_dummy_candidate("c2", dist_m=3.0, heading_diff_rad=np.radians(40.0))

    scored = scorer.score_candidates([c1, c2], pos_std_m=3.0, yaw_std_rad=np.radians(5.0))
    assert scored[0][0].segment_id == "c1"
    assert scored[0][1] > scored[1][1]


def test_candidate_scoring_stationary_relaxation():
    scorer = CandidateScorer()
    c1 = make_dummy_candidate("c1", dist_m=3.0, heading_diff_rad=0.0)
    c2 = make_dummy_candidate("c2", dist_m=3.0, heading_diff_rad=np.radians(20.0))

    # In stationary mode, heading uncertainty is relaxed, so probabilities are closer
    scored_stat = scorer.score_candidates([c1, c2], pos_std_m=3.0, is_stationary=True)
    prob_ratio_stat = scored_stat[0][1] / scored_stat[1][1]

    # In moving mode, heading misalignment is penalized much more heavily
    scored_mov = scorer.score_candidates([c1, c2], pos_std_m=3.0, is_stationary=False, v_forward_mps=10.0)
    prob_ratio_mov = scored_mov[0][1] / scored_mov[1][1]

    assert prob_ratio_mov > prob_ratio_stat
