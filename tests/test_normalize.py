"""Tests for sp-normalize tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import AudioData, load, save
from soundplay.tools.normalize import main


class TestNormalizeCLI:
    def test_peak_normalize(self, mono_audio, tmp_path):
        # Make a quiet version
        quiet = AudioData(mono_audio.samples * 0.2, mono_audio.sample_rate)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(quiet, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        # Peak should be near 1.0 (0 dBFS)
        peak = np.max(np.abs(loaded.samples))
        assert peak > 0.95

    def test_rms_normalize(self, mono_audio, tmp_path):
        quiet = AudioData(mono_audio.samples * 0.1, mono_audio.sample_rate)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(quiet, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--mode", "rms", "--target", "-10"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        rms = np.sqrt(np.mean(loaded.samples ** 2))
        target_rms = 10.0 ** (-10.0 / 20.0)
        # Lossy format tolerance
        np.testing.assert_allclose(rms, target_rms, atol=0.02)

    def test_silence_passthrough(self, silence_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(silence_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        assert np.max(np.abs(loaded.samples)) < 1e-6

    def test_target_minus3(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--target", "-3"])
        assert result.exit_code == 0
        loaded = load(out)
        peak = np.max(np.abs(loaded.samples))
        target = 10.0 ** (-3.0 / 20.0)
        np.testing.assert_allclose(peak, target, atol=0.02)
