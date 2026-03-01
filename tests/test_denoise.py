"""Tests for sp-denoise."""

import numpy as np
import pytest
from soundplay.core.spectral import SpectralData
from soundplay.tools.denoise import _denoise

SR = 16000
HOP = 512


def _make_sd(stft: np.ndarray):
    return SpectralData(
        stft=stft.astype(np.complex64),
        sample_rate=SR,
        n_fft=2048,
        hop_length=HOP,
        window='hann',
        original_frames=stft.shape[1] * HOP,
    )


class TestDenoise:
    def test_noise_region_attenuated(self):
        # Uniform noise across all frames/bins
        stft = np.full((1, 20, 10), 0.1, dtype=np.complex64)
        sd = _make_sd(stft)
        # noise region = first 5 frames (0 to 5*HOP/SR seconds)
        noise_end_s = 5 * HOP / SR
        result = _denoise(sd, 0.0, noise_end_s, oversubtract=1.0)
        # After subtracting noise profile (0.1), signal (0.1) → should be ~0
        assert np.abs(result.stft).mean() < 0.01

    def test_signal_above_noise_preserved(self):
        # Build: noise in first 5 frames at 0.1, signal in remaining frames at 1.0
        frames = 30
        bins = 10
        stft = np.zeros((1, frames, bins), dtype=np.complex64)
        stft[0, :5, :] = 0.1   # noise region
        stft[0, 5:, :] = 1.0   # signal region
        sd = _make_sd(stft)
        noise_end_s = 5 * HOP / SR
        result = _denoise(sd, 0.0, noise_end_s, oversubtract=1.0)
        # Signal bins should still have high magnitude (1.0 - 0.1 = 0.9)
        signal_mag = np.abs(result.stft[0, 5:, :]).mean()
        assert signal_mag > 0.8

    def test_oversubtract_more_aggressive(self):
        frames = 20
        bins = 5
        stft = np.full((1, frames, bins), 0.5, dtype=np.complex64)
        sd = _make_sd(stft)
        noise_end_s = 5 * HOP / SR
        result1 = _denoise(sd, 0.0, noise_end_s, oversubtract=1.0)
        result2 = _denoise(sd, 0.0, noise_end_s, oversubtract=2.0)
        # More oversubtraction → less signal remaining
        assert np.abs(result2.stft).mean() <= np.abs(result1.stft).mean()

    def test_floor_at_zero(self):
        # Noise > signal → floor at 0, never negative magnitude
        stft = np.full((1, 10, 5), 0.05, dtype=np.complex64)
        sd = _make_sd(stft)
        noise_end_s = 5 * HOP / SR
        result = _denoise(sd, 0.0, noise_end_s, oversubtract=5.0)
        assert np.all(np.abs(result.stft) >= 0.0)

    def test_preserves_metadata(self):
        stft = np.ones((1, 10, 5), dtype=np.complex64)
        sd = _make_sd(stft)
        result = _denoise(sd, 0.0, 5 * HOP / SR, oversubtract=1.0)
        assert result.sample_rate == sd.sample_rate
        assert result.n_fft == sd.n_fft
