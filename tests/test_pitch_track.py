"""Tests for sp-pitch-track."""

import math
import numpy as np
import pytest
from soundplay.core.spectral import compute_stft
from soundplay.tools.pitch_track import _pitch_track, _midi_to_note

SR = 16000
N_FFT = 2048
HOP = 512


def _sine_spx(freq: float, duration: float = 1.0):
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    samples = (0.8 * np.sin(2 * np.pi * freq * t)).reshape(-1, 1).astype(np.float32)
    return compute_stft(samples, SR, n_fft=N_FFT, hop_length=HOP)


class TestMidiToNote:
    def test_a4(self):
        assert _midi_to_note(69.0) == 'A4'

    def test_middle_c(self):
        assert _midi_to_note(60.0) == 'C4'

    def test_c_sharp(self):
        assert _midi_to_note(61.0) == 'C#4'


class TestPitchTrack:
    def test_440hz_peak(self):
        sd = _sine_spx(440.0)
        rows = _pitch_track(sd, fmin=50.0, fmax=2000.0)
        freqs = [r[1] for r in rows if r[1] > 0]
        assert len(freqs) > 0
        median_freq = sorted(freqs)[len(freqs) // 2]
        freq_res = SR / N_FFT
        assert abs(median_freq - 440.0) <= freq_res * 1.5

    def test_csv_columns(self):
        sd = _sine_spx(440.0)
        rows = _pitch_track(sd, fmin=50.0, fmax=2000.0)
        assert len(rows) > 0
        time_s, freq_hz, midi, note = rows[0]
        assert isinstance(time_s, float)
        assert isinstance(freq_hz, float)
        assert isinstance(note, str)

    def test_time_progression(self):
        sd = _sine_spx(440.0)
        rows = _pitch_track(sd, fmin=50.0, fmax=2000.0)
        times = [r[0] for r in rows]
        assert times == sorted(times)
        assert times[0] == pytest.approx(0.0)

    def test_frame_count(self):
        sd = _sine_spx(440.0)
        rows = _pitch_track(sd, fmin=50.0, fmax=2000.0)
        assert len(rows) == sd.frames

    def test_midi_note_for_440(self):
        sd = _sine_spx(440.0)
        rows = _pitch_track(sd, fmin=50.0, fmax=2000.0)
        # Most frames should be close to midi 69 (A4)
        midis = [r[2] for r in rows if not math.isnan(r[2])]
        assert len(midis) > 0
        median_midi = sorted(midis)[len(midis) // 2]
        assert abs(median_midi - 69.0) < 1.0
