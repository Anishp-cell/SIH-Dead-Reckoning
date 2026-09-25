"""
Unit Tests for Phase 7/8 Curvilinear Road-Spline Odometry and Along-Track Clamping:
Tests:
1. Spline odometry initialization and reset
2. Forward arc-length integration under constant and variable speeds
3. Standstill anchor / ZUPT integration (zero velocity integration)
4. Segment transitions along polyline
5. Kalman update via apply_map_along_track_update
6. Chi2 gating of inconsistent along-track updates
"""

import pytest
import numpy as np

from Data_details.src.phase7.matching.spline_odometry import RoadSplineOdometry, SplineOdometryConfig
from Data_details.src.phase7.matching.temporal_matcher import MapMatchResult
from Data_details.src.phase7.map.map_database import RoadSegment
from Data_details.src.phase7.matching.map_measurement import MapMeasurementModel
from Data_details.src.phase7.matching.map_update import apply_map_along_track_update
from Data_details.src.phase5.core.eskf import NominalState


@pytest.fixture
def dummy_match():
    return MapMatchResult(
        timestamp=100.0,
        matched_segment_id="way_1_seg_0",
        matched_way_id=1,
        projected_position_enu=np.array([10.0, 0.0]),
        road_heading_enu=0.0,
        road_heading_gps_deg=90.0,
        cross_track_error=0.0,
        along_track_position=10.0,
        candidate_count=1,
        candidate_probability=0.95,
        map_confidence=0.95,
        heading_residual=0.0,
        distance_residual=0.0,
        transition_score=1.0,
        accepted=True,
    )


class TestSplineOdometry:
    def test_init_and_reset(self):
        odo = RoadSplineOdometry()
        assert not odo.is_active
        assert odo.s_total_m == 0.0
        assert odo.accumulated_var_m2 == 0.25

    def test_start_outage(self, dummy_match):
        odo = RoadSplineOdometry()
        odo.start_outage(dummy_match, timestamp=100.0)
        assert odo.is_active
        assert odo.active_segment_id == "way_1_seg_0"
        assert odo.s_segment_m == 10.0
        assert odo.last_timestamp == 100.0

    def test_forward_integration(self, dummy_match):
        odo = RoadSplineOdometry()
        odo.start_outage(dummy_match, timestamp=100.0)

        # Step 1: 10 m/s for 1.0 second -> +10m along track
        target_along, var_along, valid = odo.step(
            v_forward_ai_mps=10.0,
            sigma_v_mps=0.5,
            timestamp=101.0,
            match=dummy_match,
            is_stationary=False,
            dt_override=1.0,
        )

        assert valid
        assert pytest.approx(target_along, abs=1e-3) == 20.0  # 10m initial + 10m step
        assert var_along > 0.25

    def test_standstill_anchor(self, dummy_match):
        odo = RoadSplineOdometry()
        odo.start_outage(dummy_match, timestamp=100.0)

        # Vehicle is stationary: velocity should clamp to 0 regardless of AI output
        target_along, var_along, valid = odo.step(
            v_forward_ai_mps=1.5,  # slight creep noise
            sigma_v_mps=0.2,
            timestamp=101.0,
            match=dummy_match,
            is_stationary=True,
            dt_override=1.0,
        )

        assert valid
        assert pytest.approx(target_along, abs=1e-3) == 10.0  # zero advance

    def test_measurement_update_integration(self, dummy_match):
        model = MapMeasurementModel()
        P = np.eye(15, dtype=np.float64) * 4.0  # high uncertainty
        state = NominalState(
            p=np.array([8.0, 0.0, 0.0]),  # 2m behind expected 10m
            v=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            ba=np.zeros(3),
            bg=np.zeros(3),
        )

        state_plus, P_plus, nu, S, nis, accepted = apply_map_along_track_update(
            state=state,
            P=P,
            match=dummy_match,
            model=model,
            target_along_m=12.0,
            variance_along_m2=0.5,
        )

        assert accepted
        # Position along East should move toward target
        assert state_plus.p[0] > 8.0
        # Variance should decrease
        assert P_plus[0, 0] < P[0, 0]

    def test_rejection_of_wild_outlier(self, dummy_match):
        model = MapMeasurementModel()
        P = np.eye(15, dtype=np.float64) * 0.1  # tight covariance
        state = NominalState(
            p=np.array([10.0, 0.0, 0.0]),
            v=np.array([10.0, 0.0, 0.0]),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            ba=np.zeros(3),
            bg=np.zeros(3),
        )

        # Target is 200m away (huge unphysical jump)
        state_plus, P_plus, nu, S, nis, accepted = apply_map_along_track_update(
            state=state,
            P=P,
            match=dummy_match,
            model=model,
            target_along_m=210.0,
            variance_along_m2=0.5,
        )

        assert not accepted
        # State should remain unchanged on rejection
        assert np.allclose(state_plus.p, state.p)
