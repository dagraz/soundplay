"""Tests for sp-filter tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import AudioData, load, save
from soundplay.tools.filter import main


SR = 16000


def _make_two_tone(f1, f2, sr=SR, duration=1.0):
    """Generate a signal with two frequency components."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    samples = (0.4 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t))
    return AudioData(samples.reshape(-1, 1).astype(np.float32), sr)


def _measure_freq_power(audio, freq, sr=SR):
    """Measure power at a specific frequency using FFT."""
    signal = audio.samples[:, 0]
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    idx = np.argmin(np.abs(freqs - freq))
    return np.abs(fft[idx])


class TestFilterCLI:
    def test_lowpass_attenuates_high(self, tmp_path):
        # 200 Hz + 4000 Hz signal, lowpass at 1000 Hz should keep 200, attenuate 4000
        audio = _make_two_tone(200, 4000)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--type", "lowpass", "--freq", "1000"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        power_low = _measure_freq_power(loaded, 200)
        power_high = _measure_freq_power(loaded, 4000)
        # 4000 Hz should be significantly attenuated relative to 200 Hz
        assert power_high < power_low * 0.1

    def test_highpass_attenuates_low(self, tmp_path):
        audio = _make_two_tone(200, 4000)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--type", "highpass", "--freq", "1000"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        power_low = _measure_freq_power(loaded, 200)
        power_high = _measure_freq_power(loaded, 4000)
        assert power_low < power_high * 0.1

    def test_bandpass(self, tmp_path):
        # Three tones: 100, 1000, 6000 Hz. Bandpass 500-2000 should keep 1000.
        t = np.arange(SR, dtype=np.float32) / SR
        signal = (0.3 * np.sin(2 * np.pi * 100 * t)
                  + 0.3 * np.sin(2 * np.pi * 1000 * t)
                  + 0.3 * np.sin(2 * np.pi * 6000 * t))
        audio = AudioData(signal.reshape(-1, 1).astype(np.float32), SR)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--type", "bandpass",
                                       "--freq", "500", "--freq-hi", "2000"])
        assert result.exit_code == 0
        loaded = load(out)
        p100 = _measure_freq_power(loaded, 100)
        p1000 = _measure_freq_power(loaded, 1000)
        p6000 = _measure_freq_power(loaded, 6000)
        assert p100 < p1000 * 0.1
        assert p6000 < p1000 * 0.1

    def test_missing_freq_hi_for_bandpass(self, wav_file):
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_file), "--type", "bandpass", "--freq", "500"])
        assert result.exit_code != 0
