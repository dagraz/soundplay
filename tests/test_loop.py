"""Tests for sp-loop tool."""

import numpy as np
import pytest
from click.testing import CliRunner

from soundplay.core.audio import load, save
from soundplay.core.spectral import save as spx_save, load as spx_load
from soundplay.tools.loop import main


class TestLoopCLI:
    def test_default_loop_2x(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert loaded.frames == mono_audio.frames * 2

    def test_loop_3x(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "-n", "3"])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.frames == mono_audio.frames * 3

    def test_loop_1x_identity(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "-n", "1"])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.frames == mono_audio.frames
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-3)

    def test_spectral_loop(self, spectral_data, tmp_path):
        inp = tmp_path / "in.spx"
        out = tmp_path / "out.spx"
        spx_save(spectral_data, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "-n", "2"])
        assert result.exit_code == 0
        loaded = spx_load(out)
        assert loaded.frames == spectral_data.frames * 2
        assert loaded.original_frames == spectral_data.original_frames * 2

    def test_duration_multiplied(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "-n", "4"])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.duration == pytest.approx(mono_audio.duration * 4, abs=0.01)
