"""Tests for sp-rms."""

import math
import numpy as np
import pytest
from soundplay.core.audio import AudioData
from soundplay.tools.rms import _rms_track, _SILENCE_DB

SR = 16000


def _make_audio(samples: np.ndarray):
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return AudioData(samples.astype(np.float32), SR)


class TestRmsTrack:
    def test_silence_low_rms(self):
        audio = _make_audio(np.zeros(SR))
        rows = _rms_track(audio, window_s=0.1, hop_s=0.1)
        for _, rms_db, peak_db in rows:
            assert rms_db <= -90.0
            assert peak_db <= -90.0

    def test_sine_expected_dbfs(self):
        # 0 dBFS sine (amplitude 1.0): RMS = 1/sqrt(2) → -3.01 dBFS
        t = np.arange(SR, dtype=np.float32) / SR
        samples = np.sin(2 * np.pi * 440.0 * t)
        audio = _make_audio(samples)
        rows = _rms_track(audio, window_s=0.5, hop_s=0.5)
        # Use the first full window
        _, rms_db, peak_db = rows[0]
        assert abs(rms_db - (-3.01)) < 0.5
        assert abs(peak_db - 0.0) < 0.5

    def test_csv_columns(self):
        audio = _make_audio(np.zeros(SR))
        rows = _rms_track(audio, window_s=0.1, hop_s=0.1)
        assert len(rows) > 0
        time_s, rms_db, peak_db = rows[0]
        assert isinstance(time_s, float)
        assert isinstance(rms_db, float)
        assert isinstance(peak_db, float)

    def test_time_progression(self):
        audio = _make_audio(np.zeros(SR))
        rows = _rms_track(audio, window_s=0.1, hop_s=0.1)
        times = [r[0] for r in rows]
        assert times == sorted(times)
        assert times[0] == pytest.approx(0.0)

    def test_window_count(self):
        # 1s audio, 0.1s window, 0.1s hop → ~10 windows
        audio = _make_audio(np.zeros(SR))
        rows = _rms_track(audio, window_s=0.1, hop_s=0.1)
        assert len(rows) == pytest.approx(10, abs=1)

    def test_silence_db_floor(self):
        audio = _make_audio(np.zeros(SR))
        rows = _rms_track(audio, window_s=0.1, hop_s=0.1)
        for _, rms_db, _ in rows:
            assert rms_db >= _SILENCE_DB

    def test_half_amplitude_sine(self):
        # amplitude 0.5 → peak = -6 dBFS, RMS ≈ -9 dBFS
        t = np.arange(SR, dtype=np.float32) / SR
        samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        audio = _make_audio(samples)
        rows = _rms_track(audio, window_s=0.5, hop_s=0.5)
        _, rms_db, peak_db = rows[0]
        assert abs(rms_db - (-9.03)) < 0.5
        assert abs(peak_db - (-6.02)) < 0.5
