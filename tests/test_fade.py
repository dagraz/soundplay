"""Tests for sp-fade tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import load, save
from soundplay.tools.fade import main


class TestFadeCLI:
    def test_fade_in(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--fade-in", "0.5"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        # Beginning should be near-silent
        first_samples = loaded.samples[:100, 0]
        assert np.max(np.abs(first_samples)) < 0.05
        # End should be unchanged
        last_samples = loaded.samples[-1000:, 0]
        orig_last = mono_audio.samples[-1000:, 0]
        np.testing.assert_allclose(last_samples, orig_last, atol=1e-3)

    def test_fade_out(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--fade-out", "0.5"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        # End should be near-silent
        last_samples = loaded.samples[-100:, 0]
        assert np.max(np.abs(last_samples)) < 0.05

    def test_fade_percentage(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--fade-in", "50%"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        # At 25% mark (midpoint of fade), amplitude should be about half
        quarter = loaded.frames // 4
        mid_rms = np.sqrt(np.mean(loaded.samples[quarter-100:quarter+100] ** 2))
        full_rms = np.sqrt(np.mean(mono_audio.samples[quarter-100:quarter+100] ** 2))
        assert mid_rms < full_rms * 0.8

    def test_no_fade_option_errors(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp)])
        assert result.exit_code != 0

    def test_fade_in_and_out(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--fade-in", "0.2", "--fade-out", "0.2"])
        assert result.exit_code == 0
        loaded = load(out)
        assert np.max(np.abs(loaded.samples[:50, 0])) < 0.05
        assert np.max(np.abs(loaded.samples[-50:, 0])) < 0.05
