"""
Unit tests for Phase 8 GNSS Recovery, Joseph update, step clamping, and continuity metrics.
"""

import numpy as np
import pytest

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase8.fusion.gnss_update import apply_gnss_position_update
from Data_details.src.phase8.fusion.recovery import SmoothRecoveryManager


def test_recovery_step_clamping():
    """
    Verifies that a massive position discrepancy (e.g. 50m drift during blackout)
    is clamped to <= 3.5m on the first recovery update step to prevent teleportation.
    """
    state0 = NominalState(
        p=np.array([0.0, 0.0, 0.0]),
        v=np.array([10.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
    )

    # Large covariance after long outage
    P0 = np.eye(15) * 100.0

    # GNSS fix indicates vehicle is 50m ahead
    z_pos = np.array([50.0, 0.0, 0.0])
    R_p = np.eye(3) * 4.0  # std = 2.0m

    state_plus, P_plus, nu, S, nis, accepted, actual_step = apply_gnss_position_update(
        state=state0,
        P=P0,
        z_pos_enu=z_pos,
        R_p=R_p,
        recovery_alpha=0.5,
        max_step_clamp_m=3.5,
    )

    # With high uncertainty and 50m innovation, step is clamped to exactly 3.5m
    assert actual_step <= 3.500001
    delta_p = np.linalg.norm(state_plus.p - state0.p)
    assert delta_p <= 3.500001
    assert np.all(np.isfinite(P_plus))


def test_smooth_recovery_continuity_metrics():
    """
    Verifies that SmoothRecoveryManager correctly flags violations of
    the 4 quantitative continuity metrics:
    p_step <= 3.5m, v_step <= 1.0m/s, yaw_step <= 3.0 deg, pseudo_accel <= 2.5 m/s^2.
    """
    manager = SmoothRecoveryManager(
        max_p_step_m=3.5,
        max_v_step_mps=1.0,
        max_yaw_step_deg=3.0,
        max_pseudo_accel_mps2=2.5,
    )

    # 1. Compliant transition
    p_pre = np.array([100.0, 50.0, 0.0])
    p_post = np.array([102.0, 50.5, 0.0])  # ~2.06m step <= 3.5m
    v_pre = np.array([12.0, 0.0, 0.0])
    v_post = np.array([12.1, 0.05, 0.0])  # ~0.11 m/s step <= 1.0m/s
    q_pre = np.array([1.0, 0.0, 0.0, 0.0])
    q_post = np.array([0.99996, 0.0, 0.0, 0.0087])  # ~1 deg yaw step <= 3 deg

    rep_pass = manager.evaluate_transition(
        timestamp=10.0,
        p_pre=p_pre,
        p_post=p_post,
        v_pre=v_pre,
        v_post=v_post,
        q_pre=q_pre,
        q_post=q_post,
        dt=0.1,
    )
    assert rep_pass.zero_teleportation_passed
    assert rep_pass.p_step_m <= 3.5
    assert rep_pass.v_step_mps <= 1.0
    assert rep_pass.pseudo_accel_mps2 <= 2.5

    # 2. Teleportation violation (e.g. naive state reset)
    p_teleport = np.array([120.0, 50.0, 0.0])  # 20m step > 3.5m
    rep_fail = manager.evaluate_transition(
        timestamp=10.1,
        p_pre=p_pre,
        p_post=p_teleport,
        v_pre=v_pre,
        v_post=v_post,
        q_pre=q_pre,
        q_post=q_post,
        dt=0.1,
    )
    assert not rep_fail.zero_teleportation_passed
    assert rep_fail.p_step_m > 3.5
