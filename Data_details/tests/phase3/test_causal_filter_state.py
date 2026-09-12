"""
Unit tests for Causal Filter State and No-Future-Data Leakage.
Crucial test: Proves that for causal filters, the output at time step k
is strictly independent of future inputs at steps k+1, k+2, ... N.
"""

import numpy as np
import pytest

from Data_details.src.filter_design import (
    apply_butterworth_causal,
    apply_butterworth_offline,
    RealTimeCausalFilter,
)


class TestCausalFilterState:
    """Verifies causal integrity and zero future-data leakage."""

    def test_causal_filter_no_future_leakage(self):
        """
        Tests causal integrity:
        Given two signals x1 and x2 that are identical up to index k,
        but x2 has completely different (randomized) values after index k:
        - A causal filter's output y1[:k+1] and y2[:k+1] MUST BE IDENTICAL.
        - An offline non-causal filter (filtfilt) WILL NOT be identical because it cheats with future samples.
        """
        fs = 10.0
        n = 100
        k = 50  # Split point

        np.random.seed(42)
        x1 = np.sin(np.linspace(0, 5 * np.pi, n)) + 0.1 * np.random.normal(0, 1.0, n)
        x2 = x1.copy()
        # Corrupt future samples drastically
        x2[k+1:] = x2[k+1:] + 50.0

        # 1. Causal filter test
        y1_causal = apply_butterworth_causal(x1, cutoff_hz=1.5, fs=fs, order=2)
        y2_causal = apply_butterworth_causal(x2, cutoff_hz=1.5, fs=fs, order=2)

        # Output at and before index k MUST be strictly identical
        assert np.allclose(y1_causal[:k+1], y2_causal[:k+1], atol=1e-7), (
            "Causal filter failed: output before index k was affected by future data!"
        )

        # 2. Non-causal filter demonstration: filtfilt WILL leak future samples into the past
        y1_offline = apply_butterworth_offline(x1, cutoff_hz=1.5, fs=fs, order=2)
        y2_offline = apply_butterworth_offline(x2, cutoff_hz=1.5, fs=fs, order=2)

        # Output before index k is contaminated by future corruption
        is_contaminated = not np.allclose(y1_offline[:k+1], y2_offline[:k+1], atol=1e-3)
        assert is_contaminated, "Offline filtfilt unexpectedly did not use future samples."

    def test_stateful_streaming_filter_continuity(self):
        """
        Verifies that streaming chunks through RealTimeCausalFilter
        maintains internal state zi with continuous smooth transitions across chunk boundaries.
        """
        fs = 10.0
        n = 200
        x = np.sin(np.linspace(0, 10, n))

        chunk1 = x[:100]
        chunk2 = x[100:]

        cf = RealTimeCausalFilter(cutoff_hz=1.5, fs=fs, order=2)
        y_chunk1 = cf.process_vector(chunk1)
        y_chunk2 = cf.process_vector(chunk2)

        y_stream_joined = np.concatenate([y_chunk1, y_chunk2])
        y_single_batch = apply_butterworth_causal(x, cutoff_hz=1.5, fs=fs, order=2)

        assert np.allclose(y_stream_joined, y_single_batch, atol=1e-5)
