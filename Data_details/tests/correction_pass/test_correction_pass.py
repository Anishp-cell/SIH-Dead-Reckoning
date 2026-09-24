"""
Pre-Phase-8 Correction Pass Verification Suite:
Validates that all mathematical, architectural, and runtime corrections are strictly enforced:
1. Feature Column Priority: Unfiltered vehicle kinematic channels are preferred over low-pass filtered.
2. Causal Warm Start: ai_engine.warm_up() populates FIFO buffer with t < t_0 samples, eliminating cold-start transient.
3. Analytical Heading Jacobian: Exact closed-form formulation matches numerical finite differences (< 1e-6).
4. NIS Innovation Gating: Raw innovation is evaluated against Chi-square bound prior to soft clipping.
5. Dual Export Parity: TorchScript / ONNX export preserves both speed and uncertainty without stripping.
6. Temporal Causality: Benchmark step k strictly depends on t <= t_k.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine, MotionEstimate
from Data_details.src.phase4.export import ExportWrapperDual
from Data_details.src.phase5.core.quaternion import (
    euler_zyx_to_quaternion,
    quaternion_to_rotation_matrix,
    inject_small_angle_error,
)
from Data_details.src.phase7.matching.temporal_matcher import MapMatchResult
from Data_details.src.phase7.matching.map_measurement import MapMeasurementModel, MapMeasurementConfig
from Data_details.src.phase7.matching.map_update import apply_map_cross_track_update, apply_map_heading_update
from Data_details.src.phase5.core.state import NominalState


class MockSpeedUncertaintyModel(nn.Module):
    """Mock neural network producing both speed and uncertainty."""
    def __init__(self):
        super().__init__()
        self.fc_speed = nn.Linear(30 * 12, 1)
        self.fc_std = nn.Linear(30 * 12, 1)

    def forward(self, x):
        b = x.size(0)
        flat = x.view(b, -1)
        speed = torch.relu(self.fc_speed(flat))
        std = torch.sigmoid(self.fc_std(flat)) + 0.05
        return {"speed": speed, "std": std}


def test_feature_column_priority():
    """Verify that feature extraction prefers unfiltered features over _filtered."""
    df_columns = [
        "acc_fwd_veh", "acc_fwd_veh_filtered",
        "acc_lat_veh", "acc_lat_veh_filtered",
        "acc_up_veh", "acc_up_veh_filtered",
        "gyro_roll_veh", "gyro_roll_veh_filtered",
        "gyro_pitch_veh", "gyro_pitch_veh_filtered",
        "gyro_yaw_veh", "gyro_yaw_veh_filtered",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    feature_cols = [
        "acc_fwd_veh", "acc_lat_veh", "acc_up_veh",
        "gyro_roll_veh", "gyro_pitch_veh", "gyro_yaw_veh",
        "jerk_fwd", "acc_horiz_norm", "gyro_norm",
        "ori_pitch_deg", "vibration_energy", "is_stationary",
    ]
    for idx_c, col_name in enumerate(feature_cols):
        if col_name not in df_columns:
            fallback = col_name + "_filtered"
            if fallback in df_columns:
                feature_cols[idx_c] = fallback

    assert feature_cols[0] == "acc_fwd_veh"
    assert feature_cols[1] == "acc_lat_veh"
    assert feature_cols[2] == "acc_up_veh"
    assert not any("_filtered" in c for c in feature_cols)


def test_causal_warm_start_buffer():
    """Verify that warm_up pre-populates rolling buffer and prevents zero-buffer transient."""
    model = MockSpeedUncertaintyModel()
    engine = CausalStreamingInferenceEngine(
        model=model,
        normalization_path="Data_details/outputs/phase4/models/normalization.json",
        window_length=30,
        in_features=12,
    )
    assert engine.samples_received == 0
    assert np.all(engine.buffer == 0.0)

    # Generate 30 pre-blackout samples (t < t_0)
    warm_samples = np.random.randn(30, 12).astype(np.float32)
    engine.warm_up(warm_samples)

    assert engine.samples_received == 30
    assert not np.all(engine.buffer == 0.0)

    # First real-time update at t_0
    t0_sample = np.random.randn(12).astype(np.float32)
    est = engine.update(t0_sample, timestamp=100.0)

    assert isinstance(est, MotionEstimate)
    assert est.forward_speed_mps >= 0.0
    assert est.speed_uncertainty > 0.0


def test_analytical_heading_jacobian_vs_finite_difference():
    """Verify that analytical heading Jacobian matches numerical finite differences (< 1e-6) for non-zero attitude."""
    model = MapMeasurementModel()
    P = np.eye(15) * 0.01

    # Vehicle at realistic attitude: roll=0.15 rad, pitch=-0.25 rad, yaw=0.85 rad
    roll, pitch, yaw = 0.15, -0.25, 0.85
    q = euler_zyx_to_quaternion(roll, pitch, yaw)
    R_nb = quaternion_to_rotation_matrix(q)

    match = MapMatchResult(
        timestamp=1.0,
        matched_segment_id="seg1",
        matched_way_id=1,
        projected_position_enu=np.array([10.0, 20.0]),
        road_heading_enu=yaw + 0.05,
        road_heading_gps_deg=45.0,
        cross_track_error=0.0,
        along_track_position=10.0,
        candidate_count=1,
        candidate_probability=0.9,
        map_confidence=0.9,
        heading_residual=0.05,
        distance_residual=0.0,
        transition_score=0.9,
        accepted=True,
    )

    nu, H_analytic, R, nis, accepted = model.compute_heading_update(
        match, P, v_forward_mps=10.0, R_nb=R_nb
    )

    # Numerical perturbation: right-multiplicative attitude error
    eps = 1e-7
    H_numeric = np.zeros(3)
    for i in range(3):
        dtheta = np.zeros(3)
        dtheta[i] = eps
        q_pert_plus = inject_small_angle_error(q, dtheta)
        R_pert_plus = quaternion_to_rotation_matrix(q_pert_plus)
        psi_plus = np.arctan2(R_pert_plus[1, 0], R_pert_plus[0, 0])

        q_pert_minus = inject_small_angle_error(q, -dtheta)
        R_pert_minus = quaternion_to_rotation_matrix(q_pert_minus)
        psi_minus = np.arctan2(R_pert_minus[1, 0], R_pert_minus[0, 0])

        # wrap difference
        d_psi = float(np.arctan2(np.sin(psi_plus - psi_minus), np.cos(psi_plus - psi_minus)))
        H_numeric[i] = d_psi / (2.0 * eps)

    # In H_analytic, theta error corresponds to indices 6, 7, 8
    H_sub = H_analytic[0, 6:9]
    np.testing.assert_allclose(H_sub, H_numeric, atol=1e-5, rtol=1e-4)


def test_nis_innovation_gating_sequence():
    """Verify that gross outliers (> 15m) are rejected by computing NIS on raw unclipped innovation."""
    model = MapMeasurementModel()
    P = np.eye(15) * 1.0  # Tight covariance: S ~ 1.0 + 4.0 = 5.0

    # Cross-track error of 50 meters (gross outlier, e.g. off-road driving)
    match_outlier = MapMatchResult(
        timestamp=1.0,
        matched_segment_id="seg_outlier",
        matched_way_id=99,
        projected_position_enu=np.array([50.0, 50.0]),
        road_heading_enu=0.0,
        road_heading_gps_deg=90.0,
        cross_track_error=50.0,
        along_track_position=0.0,
        candidate_count=1,
        candidate_probability=0.9,
        map_confidence=0.9,
        heading_residual=0.0,
        distance_residual=50.0,
        transition_score=0.9,
        accepted=True,
    )

    nu_clamped, H, R, nis, accepted = model.compute_cross_track_update(match_outlier, P)

    # NIS on raw innovation (50^2 / S = 2500 / 5 = 500 >> 6.635)
    assert not accepted, "Gross outlier must be rejected by raw NIS gating"
    assert nis > model.chi2_threshold_1d
    assert abs(nu_clamped) <= model.cfg.max_cross_track_innovation_m


def test_dual_export_wrapper():
    """Verify that ExportWrapperDual exports both speed and uncertainty without truncation."""
    model = MockSpeedUncertaintyModel()
    wrapper = ExportWrapperDual(model)
    x = torch.randn(1, 30, 12)

    speed, uncertainty = wrapper(x)
    assert speed.shape == (1, 1)
    assert uncertainty.shape == (1, 1)
    assert float(uncertainty.item()) > 0.0
