"""
Unit tests for Phase 8 GNSS Quality Assessor and 3-DOF Chi-Square Gating.
"""

import numpy as np
import pytest

from Data_details.src.phase8.gnss.quality import GNSSQualityAssessor
from Data_details.src.phase8.gnss.gating import ChiSquareGating3DOF, GatingDecision


def test_quality_assessor_nominal():
    """Verifies that clean, healthy GNSS passes quality pre-filtering."""
    assessor = GNSSQualityAssessor(min_sats=4, max_stated_acc_m=15.0)

    p0 = np.array([100.0, 200.0, 10.0])
    rep = assessor.evaluate(timestamp=0.0, p_enu=p0, stated_acc_m=2.5, num_sats=8, speed_mps=12.0)
    assert rep.is_valid
    assert len(rep.rejection_reasons) == 0
    assert not rep.jump_detected

    # Next clean step at 10 Hz (dt = 0.1s, dx = 1.2m -> 12 m/s)
    p1 = p0 + np.array([1.2, 0.0, 0.0])
    rep1 = assessor.evaluate(timestamp=0.1, p_enu=p1, stated_acc_m=2.5, num_sats=8, speed_mps=12.0)
    assert rep1.is_valid
    assert not rep1.jump_detected
    assert np.isclose(rep1.apparent_vel_mps, 12.0, atol=1e-3)


def test_quality_assessor_low_satellites():
    """Verifies that fix with < 4 satellites is rejected."""
    assessor = GNSSQualityAssessor(min_sats=4)
    rep = assessor.evaluate(timestamp=0.0, p_enu=np.zeros(3), stated_acc_m=2.0, num_sats=3)
    assert not rep.is_valid
    assert any("INSUFFICIENT_SATELLITES" in r for r in rep.rejection_reasons)


def test_quality_assessor_high_uncertainty():
    """Verifies that fix with stated accuracy > 15m is rejected."""
    assessor = GNSSQualityAssessor(max_stated_acc_m=15.0)
    rep = assessor.evaluate(timestamp=0.0, p_enu=np.zeros(3), stated_acc_m=22.0, num_sats=7)
    assert not rep.is_valid
    assert any("HIGH_STATED_UNCERTAINTY" in r for r in rep.rejection_reasons)


def test_quality_assessor_position_step_jump():
    """Verifies that an unphysical jump (e.g. 30m in 0.1s) is flagged and rejected."""
    assessor = GNSSQualityAssessor(max_step_jump_m=20.0, max_physical_speed_mps=45.0)

    p0 = np.array([0.0, 0.0, 0.0])
    assessor.evaluate(timestamp=0.0, p_enu=p0, stated_acc_m=2.0, num_sats=8)

    # Injected 30m jump
    p_jump = np.array([30.0, 0.0, 0.0])
    rep = assessor.evaluate(timestamp=0.1, p_enu=p_jump, stated_acc_m=2.0, num_sats=8)

    assert not rep.is_valid
    assert rep.jump_detected
    assert any("POSITION_STEP_JUMP" in r for r in rep.rejection_reasons)


def test_chisquare_gating_3dof():
    """Verifies 3-DOF Chi-Square Gating thresholds."""
    gater = ChiSquareGating3DOF(threshold_accept=11.345, threshold_reject=25.0)

    S = np.eye(3) * 4.0  # std = 2.0m

    # 1. Clean innovation: nu = [1.0, 1.0, 1.0] -> NIS = (1+1+1)/4 = 0.75 <= 11.345
    nu_clean = np.array([1.0, 1.0, 1.0])
    res_clean = gater.evaluate(nu_clean, S)
    assert res_clean.decision == GatingDecision.ACCEPT
    assert np.isclose(res_clean.nis, 0.75)
    assert res_clean.inflation_factor == 1.0

    # 2. Borderline / moderate multipath: nu = [6.0, 6.0, 6.0] -> NIS = 108/4 = 27.0 -> REJECT
    nu_bad = np.array([6.0, 6.0, 6.0])
    res_bad = gater.evaluate(nu_bad, S)
    assert res_bad.decision == GatingDecision.REJECT

    # 3. Downweight zone: NIS between 11.345 and 25.0
    # Let nu = [4.0, 4.0, 2.0] -> NIS = (16 + 16 + 4)/4 = 36/4 = 9.0 (accept)
    # Let nu = [4.5, 4.5, 4.0] -> NIS = (20.25 + 20.25 + 16)/4 = 56.5/4 = 14.125 -> DOWNWEIGHT
    nu_down = np.array([4.5, 4.5, 4.0])
    res_down = gater.evaluate(nu_down, S)
    assert res_down.decision == GatingDecision.DOWNWEIGHT
    assert res_down.inflation_factor > 1.0
