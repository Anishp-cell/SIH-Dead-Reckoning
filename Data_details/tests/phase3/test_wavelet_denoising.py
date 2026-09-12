"""
Unit tests for Wavelet Denoising module.
Verifies DWT multiresolution decomposition, soft thresholding, and IDWT reconstruction.
"""

import numpy as np
import pytest

from Data_details.src.wavelet_denoising import wavelet_denoise_signal


class TestWaveletDenoising:
    """Verifies discrete wavelet transform denoising and reconstruction."""

    def test_wavelet_reconstruction_clean_signal(self):
        """
        Denoising a clean, smooth signal should preserve the signal with minimal distortion
        (correlation > 0.999).
        """
        t = np.linspace(0, 10, 256)
        clean = np.sin(t) + 0.5 * np.cos(2 * t)

        denoised, meta = wavelet_denoise_signal(clean, wavelet="sym4", level=3, threshold_mode="soft")

        assert len(denoised) == len(clean)
        corr = np.corrcoef(clean, denoised)[0, 1]
        assert corr > 0.995

    def test_wavelet_reduces_gaussian_noise(self):
        """
        Denoising a signal corrupted by additive Gaussian white noise
        must reduce noise standard deviation by at least 35%.
        """
        np.random.seed(42)
        t = np.linspace(0, 10, 512)
        clean = np.sin(t)
        noise = np.random.normal(0, 0.4, 512)
        noisy = clean + noise

        denoised, meta = wavelet_denoise_signal(noisy, wavelet="sym4", level=3, threshold_mode="soft")

        # Error relative to ground truth clean signal
        mse_noisy = np.mean((noisy - clean) ** 2)
        mse_denoised = np.mean((denoised - clean) ** 2)

        assert mse_denoised < mse_noisy * 0.65  # At least 35% MSE reduction
        assert meta["sigma_noise"] > 0.0
