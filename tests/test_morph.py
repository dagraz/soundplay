"""Tests for sp-morph."""

import numpy as np
import pytest
from soundplay.core.spectral import SpectralData
from soundplay.tools.morph import _morph

SR = 16000
N_FFT = 512
HOP = 128
BINS = N_FFT // 2 + 1


def _make_sd(stft_value: complex, frames: int = 20, channels: int = 1):
    stft = np.full((channels, frames, BINS), stft_value, dtype=np.complex64)
    return SpectralData(
        stft=stft,
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        window='hann',
        original_frames=frames * HOP,
    )


class TestMorph:
    def test_blend_zero_approx_sd1(self):
        sd1 = _make_sd(1.0 + 0j)
        sd2 = _make_sd(0.0 + 0j)
        result = _morph(sd1, sd2, blend_start=0.0, blend_end=0.0)
        np.testing.assert_allclose(np.abs(result.stft), 1.0, atol=1e-5)

    def test_blend_one_approx_sd2(self):
        sd1 = _make_sd(0.0 + 0j)
        sd2 = _make_sd(1.0 + 0j)
        result = _morph(sd1, sd2, blend_start=1.0, blend_end=1.0)
        np.testing.assert_allclose(np.abs(result.stft), 1.0, atol=1e-5)

    def test_crossfade_intermediate(self):
        sd1 = _make_sd(1.0 + 0j)
        sd2 = _make_sd(0.0 + 0j)
        result = _morph(sd1, sd2, blend_start=0.0, blend_end=1.0)
        # First frame ≈ sd1 (alpha=0), last frame ≈ sd2 (alpha=1)
        first_mag = float(np.abs(result.stft[0, 0, :]).mean())
        last_mag = float(np.abs(result.stft[0, -1, :]).mean())
        assert first_mag > 0.9
        assert last_mag < 0.1

    def test_output_frame_count_matches_sd1(self):
        sd1 = _make_sd(1.0 + 0j, frames=20)
        sd2 = _make_sd(0.5 + 0j, frames=40)
        result = _morph(sd1, sd2, blend_start=0.0, blend_end=1.0)
        assert result.frames == sd1.frames

    def test_preserves_metadata(self):
        sd1 = _make_sd(1.0 + 0j)
        sd2 = _make_sd(0.5 + 0j)
        result = _morph(sd1, sd2, blend_start=0.0, blend_end=1.0)
        assert result.sample_rate == sd1.sample_rate
        assert result.n_fft == sd1.n_fft
        assert result.hop_length == sd1.hop_length
