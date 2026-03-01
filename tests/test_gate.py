"""Tests for sp-gate."""

import numpy as np
import pytest
from soundplay.core.spectral import SpectralData
from soundplay.tools.gate import _gate


def _make_sd(stft: np.ndarray, sr: int = 16000, n_fft: int = 2048):
    return SpectralData(
        stft=stft.astype(np.complex64),
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=512,
        window='hann',
        original_frames=sr,
    )


class TestGate:
    def test_quiet_bins_zeroed(self):
        # stft with small values below -40 dB threshold
        stft = np.full((1, 10, 5), 1e-4, dtype=np.complex64)  # < -40 dB
        sd = _make_sd(stft)
        result = _gate(sd, threshold_db=-40.0)
        assert np.all(result.stft == 0.0)

    def test_loud_bins_kept(self):
        # stft with large values well above threshold
        stft = np.full((1, 10, 5), 1.0, dtype=np.complex64)
        sd = _make_sd(stft)
        result = _gate(sd, threshold_db=-40.0)
        np.testing.assert_allclose(np.abs(result.stft), 1.0, atol=1e-6)

    def test_threshold_at_signal_level(self):
        # Signal exactly at threshold — should be zeroed (strict <)
        threshold_db = -20.0
        linear = 10.0 ** (threshold_db / 20.0)
        # slightly below threshold
        stft_below = np.full((1, 5, 3), linear * 0.999, dtype=np.complex64)
        sd = _make_sd(stft_below)
        result = _gate(sd, threshold_db=threshold_db)
        assert np.all(result.stft == 0.0)

        # slightly above threshold
        stft_above = np.full((1, 5, 3), linear * 1.001, dtype=np.complex64)
        sd2 = _make_sd(stft_above)
        result2 = _gate(sd2, threshold_db=threshold_db)
        assert np.all(np.abs(result2.stft) > 0)

    def test_mixed_bins(self):
        stft = np.zeros((1, 1, 4), dtype=np.complex64)
        stft[0, 0, 0] = 0.0001  # below -40 dB
        stft[0, 0, 1] = 0.5     # above -40 dB
        stft[0, 0, 2] = 0.0001
        stft[0, 0, 3] = 1.0
        sd = _make_sd(stft)
        result = _gate(sd, threshold_db=-40.0)
        assert result.stft[0, 0, 0] == 0.0
        assert result.stft[0, 0, 2] == 0.0
        assert abs(result.stft[0, 0, 1]) > 0
        assert abs(result.stft[0, 0, 3]) > 0

    def test_preserves_metadata(self):
        stft = np.ones((1, 5, 5), dtype=np.complex64)
        sd = _make_sd(stft)
        result = _gate(sd, threshold_db=-40.0)
        assert result.sample_rate == sd.sample_rate
        assert result.n_fft == sd.n_fft
        assert result.original_frames == sd.original_frames
