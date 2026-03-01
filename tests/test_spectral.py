"""Tests for soundplay.core.spectral."""

import numpy as np
import pytest

from soundplay.core.spectral import (
    SpectralData, compute_stft, compute_istft,
    save as spx_save, load as spx_load,
)


class TestSpectralDataProperties:
    def test_channels(self, spectral_data):
        assert spectral_data.channels == 1

    def test_bins(self, spectral_data):
        assert spectral_data.bins == spectral_data.n_fft // 2 + 1

    def test_duration(self, spectral_data):
        assert spectral_data.duration == pytest.approx(1.0)


class TestSpxRoundtrip:
    def test_save_load(self, spectral_data, tmp_path):
        p = tmp_path / "rt.spx"
        spx_save(spectral_data, p)
        loaded = spx_load(p)
        assert loaded.sample_rate == spectral_data.sample_rate
        assert loaded.n_fft == spectral_data.n_fft
        assert loaded.hop_length == spectral_data.hop_length
        assert loaded.window == spectral_data.window
        assert loaded.original_frames == spectral_data.original_frames
        np.testing.assert_allclose(
            np.abs(loaded.stft), np.abs(spectral_data.stft), atol=1e-5
        )


class TestStftIstftAccuracy:
    def test_reconstruction(self, mono_audio):
        sd = compute_stft(mono_audio.samples, mono_audio.sample_rate)
        reconstructed = compute_istft(sd)
        # Trim to original length
        orig = mono_audio.samples[:reconstructed.shape[0]]
        reconstructed = reconstructed[:orig.shape[0]]
        np.testing.assert_allclose(reconstructed, orig, atol=1e-3)

    def test_stereo_reconstruction(self, stereo_audio):
        sd = compute_stft(stereo_audio.samples, stereo_audio.sample_rate)
        assert sd.channels == 2
        reconstructed = compute_istft(sd)
        orig = stereo_audio.samples[:reconstructed.shape[0]]
        reconstructed = reconstructed[:orig.shape[0]]
        np.testing.assert_allclose(reconstructed, orig, atol=1e-3)

    def test_440hz_peak_bin(self, mono_audio):
        sd = compute_stft(mono_audio.samples, mono_audio.sample_rate)
        magnitudes = np.abs(sd.stft[0]).mean(axis=0)  # avg over time frames
        peak_bin = np.argmax(magnitudes)
        freq_resolution = mono_audio.sample_rate / sd.n_fft
        peak_freq = peak_bin * freq_resolution
        assert peak_freq == pytest.approx(440.0, abs=freq_resolution)
