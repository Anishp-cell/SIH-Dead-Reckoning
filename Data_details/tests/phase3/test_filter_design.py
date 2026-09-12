"""
Unit tests for Filter Design module (Moving Average, Butterworth Low-Pass, Causal Streaming).
Verifies frequency attenuation, passband preservation, and filter stability.
"""

import numpy as np
import pytest

from Data_details.src.filter_design import (
    moving_average_filter,
    design_butterworth_lowpass,
    apply_butterworth_causal,
    apply_butterworth_offline,
    RealTimeCausalFilter,
)


class TestFilterDesign:
    """Verifies digital filter design and frequency characteristics."""

    def test_butterworth_passband_preservation(self):
        """
        A 0.2 Hz low-frequency sine wave through a 1.5 Hz Butterworth filter
        must experience less than 2% amplitude attenuation.
        """
        fs = 10.0
        t = np.arange(0, 30.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 0.2 * t)

        y_causal = apply_butterworth_causal(x, cutoff_hz=1.5, fs=fs, order=2)
        # Skip initial 20 samples transient
        amp_raw = np.max(x[30:-30])
        amp_filt = np.max(y_causal[30:-30])

        assert np.isclose(amp_raw, 1.0, atol=0.01)
        assert amp_filt > 0.98  # Less than 2% loss in passband

    def test_butterworth_stopband_attenuation(self):
        """
        A 4.0 Hz high-frequency vibration signal through a 1.5 Hz Butterworth filter
        must be attenuated by at least 80% (> 14 dB attenuation).
        """
        fs = 10.0
        t = np.arange(0, 30.0, 1.0 / fs)
        x_high = np.sin(2 * np.pi * 4.0 * t)

        y_causal = apply_butterworth_causal(x_high, cutoff_hz=1.5, fs=fs, order=2)
        amp_filt = np.max(np.abs(y_causal[30:-30]))

        assert amp_filt < 0.20  # Over 80% attenuation in stopband

    def test_moving_average_causal(self):
        """Verifies causal moving average filter reduces high-frequency noise variance."""
        np.random.seed(42)
        noisy = np.random.normal(0, 1.0, 500)
        smoothed = moving_average_filter(noisy, window_size=5, mode="causal")

        assert len(smoothed) == len(noisy)
        # Variance of moving average of W white noise samples is sigma^2 / W
        assert np.std(smoothed) < np.std(noisy) * 0.60

    def test_streaming_causal_filter_matches_batch(self):
        """
        Verifies that processing sample-by-sample with RealTimeCausalFilter
        produces the exact same numerical result as causal batch processing.
        """
        fs = 10.0
        np.random.seed(42)
        x = np.sin(np.linspace(0, 10, 200)) + 0.2 * np.random.normal(0, 1.0, 200)

        # Batch causal
        y_batch = apply_butterworth_causal(x, cutoff_hz=1.5, fs=fs, order=2)

        # Streaming sample-by-sample
        cf = RealTimeCausalFilter(cutoff_hz=1.5, fs=fs, order=2)
        y_stream = np.array([cf.process_sample(val) for val in x])

        # Must match to floating-point precision
        assert np.allclose(y_batch, y_stream, atol=1e-5)
