"""
Phase 5 Unit Tests: Streaming Navigation Engine.
"""

import pytest
import numpy as np

from Data_details.src.phase5.core.eskf import ESKF15State
from Data_details.src.phase5.streaming import StreamingNavigationEngine


class TestPhase5Streaming:

    def test_streaming_engine_step_execution(self):
        engine = StreamingNavigationEngine(eskf=ESKF15State())

        p0 = np.array([0.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])

        engine.initialize(p0, v0, q0, initial_timestamp=0.0)

        # Feed 20 sequential 12-channel samples
        for i in range(20):
            t = (i + 1) * 0.1
            # Stationary sample: acc_up = -9.80665 (Phase 3 mount convention), others 0
            sample = np.zeros(12)
            sample[2] = -9.80665  # acc_up_veh in Phase 3 feature convention

            est = engine.step(sample, timestamp=t, enable_ai_speed=False)

            assert est.filter_healthy is True
            assert np.all(np.isfinite(est.p))
            assert np.all(np.isfinite(est.v))
            assert np.all(np.isfinite(est.cov_diagonal))

        # After 2 seconds of stationary data, position should remain near zero
        assert np.allclose(est.p, np.zeros(3), atol=1e-6)
