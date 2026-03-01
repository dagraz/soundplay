"""Shared fixtures for soundplay tests."""

import numpy as np
import pytest

from soundplay.core.audio import AudioData, save
from soundplay.core.spectral import SpectralData, compute_stft, save as spx_save


SR = 16000  # sample rate used throughout tests


def _sine(freq: float, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate a mono sine wave column vector (frames, 1)."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (0.8 * np.sin(2 * np.pi * freq * t)).reshape(-1, 1).astype(np.float32)


@pytest.fixture
def mono_audio() -> AudioData:
    return AudioData(_sine(440), SR)


@pytest.fixture
def stereo_audio() -> AudioData:
    left = _sine(440)
    right = _sine(880)
    return AudioData(np.hstack([left, right]), SR)


@pytest.fixture
def silence_audio() -> AudioData:
    return AudioData(np.zeros((SR, 1), dtype=np.float32), SR)


@pytest.fixture
def spectral_data(mono_audio) -> SpectralData:
    return compute_stft(mono_audio.samples, mono_audio.sample_rate)


@pytest.fixture
def wav_file(mono_audio, tmp_path):
    p = tmp_path / "test.wav"
    save(mono_audio, p)
    return p


@pytest.fixture
def flac_file(mono_audio, tmp_path):
    p = tmp_path / "test.flac"
    save(mono_audio, p)
    return p


@pytest.fixture
def ogg_file(mono_audio, tmp_path):
    p = tmp_path / "test.ogg"
    save(mono_audio, p)
    return p


@pytest.fixture
def spx_file(spectral_data, tmp_path):
    p = tmp_path / "test.spx"
    spx_save(spectral_data, p)
    return p
