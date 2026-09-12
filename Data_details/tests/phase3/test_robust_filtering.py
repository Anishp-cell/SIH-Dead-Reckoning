"""
Unit tests for Robust Adaptive Filtering (MAD and Hampel Identifier).
Verifies detection and replacement of artificial impulse spikes while preserving step transitions.
"""

import numpy as np
import pytest

from Data_details.src.adaptive_filtering import (
    compute_mad,
    median_filter_1d,
    hampel_filter,
)


class TestRobustFiltering:
    """Verifies outlier spike rejection and edge preservation."""

    def test_compute_mad_consistency(self):
        """
        Verifies that for standard normal N(0, 1) distribution,
        sigma_mad = 1.4826 * MAD is approximately 1.0.
        """
        np.random.seed(42)
        samples = np.random.normal(0, 1.0, 5000)
        mad = compute_mad(samples)
        sigma_est = 1.4826 * mad

        assert np.isclose(sigma_est, 1.0, atol=0.05)

    def test_hampel_detects_and_replaces_spikes(self):
        """
        Injects 5 extreme isolated spikes (amplitude 30.0 m/s^2) into a smooth signal.
        Hampel filter must detect all 5 spikes and replace them with local median.
        """
        np.random.seed(42)
        n = 200
        clean = np.sin(np.linspace(0, 4 * np.pi, n))

        # Corrupt 5 random isolated positions
        spike_indices = [25, 60, 95, 130, 175]
        noisy = clean.copy()
        for idx in spike_indices:
            noisy[idx] = noisy[idx] + 30.0

        cleaned, outlier_mask, meta = hampel_filter(noisy, window_size=7, n_sigmas=3.0)

        # All 5 spikes must be flagged
        for idx in spike_indices:
            assert outlier_mask[idx] is True or outlier_mask[idx] == 1
            # Replaced value must be close to clean signal
            assert np.abs(cleaned[idx] - clean[idx]) < 1.0

        assert meta["n_outliers"] >= 5

    def test_median_filter_preserves_step_transition(self):
        """
        A sharp step transition (e.g. abrupt braking) must not be blurred
        into a gradual slope by a median filter.
        """
        step_signal = np.concatenate([np.zeros(50), np.full(50, -4.0)])
        filtered = median_filter_1d(step_signal, window_size=5)

        # The plateaus must remain exact
        assert np.allclose(filtered[:45], 0.0)
        assert np.allclose(filtered[55:], -4.0)
