"""Tests for sp-stretch."""

import numpy as np
import pytest
from soundplay.core.spectral import SpectralData
from soundplay.tools.stretch import _stretch

SR = 16000
N_FFT = 512
HOP = 128
BINS = N_FFT // 2 + 1


def _make_sd(frames: int = 40):
    stft = np.random.randn(1, frames, BINS).astype(np.float32) + \
           1j * np.random.randn(1, frames, BINS).astype(np.float32)
    stft = stft.astype(np.complex64)
    return SpectralData(
        stft=stft,
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        window='hann',
        original_frames=frames * HOP,
    )


class TestStretch:
    def test_2x_doubles_frames(self):
        sd = _make_sd(frames=40)
        result = _stretch(sd, 2.0)
        assert result.frames == round(40 * 2.0)

    def test_half_halves_frames(self):
        sd = _make_sd(frames=40)
        result = _stretch(sd, 0.5)
        assert result.frames == round(40 * 0.5)

    def test_identity_approx(self):
        sd = _make_sd(frames=40)
        result = _stretch(sd, 1.0)
        assert result.frames == 40
        np.testing.assert_allclose(np.abs(result.stft), np.abs(sd.stft), atol=1e-4)

    def test_original_frames_scaled(self):
        sd = _make_sd(frames=40)
        result = _stretch(sd, 2.0)
        assert result.original_frames == round(sd.original_frames * 2.0)

    def test_preserves_metadata(self):
        sd = _make_sd(frames=40)
        result = _stretch(sd, 1.5)
        assert result.sample_rate == sd.sample_rate
        assert result.n_fft == sd.n_fft
        assert result.hop_length == sd.hop_length
