"""Tests for sp-transpose."""

import numpy as np
import pytest
from soundplay.core.spectral import compute_stft
from soundplay.tools.transpose import _parse_semitones, _transpose


SR = 16000
N_FFT = 2048


def _sine_spx(freq: float, duration: float = 1.0):
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    samples = (0.8 * np.sin(2 * np.pi * freq * t)).reshape(-1, 1).astype(np.float32)
    return compute_stft(samples, SR, n_fft=N_FFT)


def _peak_freq(sd) -> float:
    mag = np.abs(sd.stft[0]).mean(axis=0)
    peak_bin = int(np.argmax(mag))
    return peak_bin * SR / N_FFT


class TestParseSemitones:
    def test_integer(self):
        assert _parse_semitones('12') == pytest.approx(12.0)

    def test_negative(self):
        assert _parse_semitones('-7') == pytest.approx(-7.0)

    def test_float(self):
        assert _parse_semitones('3.5') == pytest.approx(3.5)

    def test_cents(self):
        assert _parse_semitones('100c') == pytest.approx(1.0)

    def test_cents_negative(self):
        assert _parse_semitones('-50c') == pytest.approx(-0.5)

    def test_zero_cents(self):
        assert _parse_semitones('0c') == pytest.approx(0.0)


class TestTranspose:
    def test_zero_semitones_identity(self):
        sd = _sine_spx(440)
        result = _transpose(sd, 0.0)
        np.testing.assert_allclose(np.abs(result.stft), np.abs(sd.stft), atol=1e-5)

    def test_twelve_semitones_doubles_freq(self):
        sd = _sine_spx(440)
        result = _transpose(sd, 12.0)
        peak = _peak_freq(result)
        # 440 * 2 = 880 Hz; allow ±1 bin tolerance
        freq_res = SR / N_FFT
        assert abs(peak - 880.0) <= freq_res * 1.5

    def test_negative_semitones_lowers_freq(self):
        sd = _sine_spx(880)
        result = _transpose(sd, -12.0)
        peak = _peak_freq(result)
        freq_res = SR / N_FFT
        assert abs(peak - 440.0) <= freq_res * 1.5

    def test_preserves_shape(self):
        sd = _sine_spx(440)
        result = _transpose(sd, 7.0)
        assert result.stft.shape == sd.stft.shape

    def test_preserves_metadata(self):
        sd = _sine_spx(440)
        result = _transpose(sd, 3.0)
        assert result.sample_rate == sd.sample_rate
        assert result.n_fft == sd.n_fft
        assert result.hop_length == sd.hop_length
