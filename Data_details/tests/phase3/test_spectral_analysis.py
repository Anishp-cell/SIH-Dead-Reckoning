"""
Unit tests for Spectral Analysis module (FFT, Welch PSD, frequency characterization).
Uses synthetic multi-frequency sine signals to verify peak recovery and energy calculations.
"""

import numpy as np
import pytest

from Data_details.src.spectral_analysis import (
    compute_fft_spectrum,
    compute_welch_psd,
    extract_spectral_metrics,
)


class TestSpectralAnalysis:
    """Verifies Fourier transform and PSD computations on synthetic signals."""

    def test_fft_frequency_peak_recovery(self):
        """
        Synthesizes x(t) = 1.0 * sin(2*pi*f1*t) + 0.5 * sin(2*pi*f2*t)
        with f1 = 0.8 Hz and f2 = 2.4 Hz at fs = 10.0 Hz.
        Verifies that FFT correctly identifies both dominant peaks.
        """
        fs = 10.0
        duration = 50.0  # 500 samples
        t = np.arange(0, duration, 1.0 / fs)
        f1, f2 = 0.8, 2.4
        x = 1.0 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

        freqs, mag, pwr = compute_fft_spectrum(x, fs=fs)

        # Find top 2 local peaks
        peak_indices = np.argsort(pwr[1:])[-2:] + 1  # exclude DC
        recovered_freqs = sorted(freqs[peak_indices])

        assert np.isclose(recovered_freqs[0], f1, atol=0.05)
        assert np.isclose(recovered_freqs[1], f2, atol=0.05)

    def test_welch_psd_preserves_power(self):
        """Verifies that Welch PSD estimates non-zero power for dynamic signals."""
        fs = 10.0
        t = np.arange(0, 30.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 1.2 * t)

        freqs, psd = compute_welch_psd(x, fs=fs, nperseg=64)
        peak_freq = freqs[np.argmax(psd)]

        assert np.isclose(peak_freq, 1.2, atol=0.2)
        assert np.sum(psd) > 0.0

    def test_spectral_metrics_high_frequency_ratio(self):
        """
        Verifies that high-frequency energy ratio correctly separates low-frequency
        and high-frequency signals based on the 2.5 Hz threshold.
        """
        fs = 10.0
        t = np.arange(0, 20.0, 1.0 / fs)

        # 1. Pure low-frequency signal (0.5 Hz) -> hf_ratio should be near 0%
        x_low = np.sin(2 * np.pi * 0.5 * t)
        m_low = extract_spectral_metrics(x_low, fs=fs, hf_cutoff=2.5)
        assert m_low["hf_ratio_pct"] < 5.0
        assert np.isclose(m_low["dominant_freq_hz"], 0.5, atol=0.1)

        # 2. Pure high-frequency signal (3.5 Hz) -> hf_ratio should be near 100%
        x_high = np.sin(2 * np.pi * 3.5 * t)
        m_high = extract_spectral_metrics(x_high, fs=fs, hf_cutoff=2.5)
        assert m_high["hf_ratio_pct"] > 90.0
        assert np.isclose(m_high["dominant_freq_hz"], 3.5, atol=0.1)
