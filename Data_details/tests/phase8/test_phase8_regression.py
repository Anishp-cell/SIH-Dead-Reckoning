"""
Regression test suite ensuring Phase 8 retains mathematical consistency with Phase 6 and 7 baselines.
"""

import numpy as np
import pytest

from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase8.core.phase8_eskf import Phase8ESKF


def test_phase8_eskf_pure_dr_matches_phase6():
    """
    Verifies that when GNSS updates are disabled (pure dead reckoning / outage mode),
    Phase8ESKF produces identical state propagation to Phase6ESKF.
    """
    p6_eskf = Phase6ESKF(enable_nhc=True, enable_zupt=True, enable_zaru=True)
    p8_eskf = Phase8ESKF(
        enable_nhc=True,
        enable_zupt=True,
        enable_zaru=True,
        enable_gnss_pos=False,
        enable_gnss_vel=False,
    )

    p0 = np.array([10.0, 20.0, 5.0])
    v0 = np.array([12.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    p6_eskf.initialize(p0, v0, q0)
    p8_eskf.initialize(p0, v0, q0)

    f_b = np.array([0.5, 0.1, -9.81])
    omega_b = np.array([0.01, -0.01, 0.02])
    dt = 0.1

    for _ in range(50):
        p6_eskf.predict(f_b, omega_b, dt)
        p8_eskf.predict(f_b, omega_b, dt)

        # Disturbance & NHC
        d6 = p6_eskf.disturbance_detector.evaluate(f_b, omega_b)
        p6_eskf.update_nhc(d_lat=d6.d_lat, d_up=d6.d_up)

        d8 = p8_eskf.disturbance_detector.evaluate(f_b, omega_b)
        p8_eskf.update_nhc(d_lat=d8.d_lat, d_up=d8.d_up)

    assert np.allclose(p6_eskf.state.p, p8_eskf.state.p, atol=1e-10)
    assert np.allclose(p6_eskf.state.v, p8_eskf.state.v, atol=1e-10)
    assert np.allclose(p6_eskf.state.q, p8_eskf.state.q, atol=1e-10)
    assert np.allclose(p6_eskf.P, p8_eskf.P, atol=1e-10)
