"""Tests for sp-trim tool."""

import numpy as np
import pytest
from click.testing import CliRunner

from soundplay.core.audio import load, save
from soundplay.tools.trim import main


class TestTrimCLI:
    def test_start_end(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--start", "0.2", "--end", "0.8"])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        expected_frames = int(round(0.6 * mono_audio.sample_rate))
        assert loaded.frames == pytest.approx(expected_frames, abs=2)

    def test_percentage(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--start", "10%", "--end", "90%"])
        assert result.exit_code == 0
        loaded = load(out)
        expected_frames = int(round(0.8 * mono_audio.sample_rate))
        assert loaded.frames == pytest.approx(expected_frames, abs=2)

    def test_negative_end(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        # Drop last 0.2s
        result = runner.invoke(main, [str(inp), str(out), "--end", "-0.2"])
        assert result.exit_code == 0
        loaded = load(out)
        expected_frames = int(round(0.8 * mono_audio.sample_rate))
        assert loaded.frames == pytest.approx(expected_frames, abs=2)

    def test_negative_start(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        # Last 0.3s
        result = runner.invoke(main, [str(inp), str(out), "--start", "-0.3"])
        assert result.exit_code == 0
        loaded = load(out)
        expected_frames = int(round(0.3 * mono_audio.sample_rate))
        assert loaded.frames == pytest.approx(expected_frames, abs=2)

    def test_duration_option(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out), "--start", "0.1", "--duration", "0.5"])
        assert result.exit_code == 0
        loaded = load(out)
        expected_frames = int(round(0.5 * mono_audio.sample_rate))
        assert loaded.frames == pytest.approx(expected_frames, abs=2)

    def test_end_and_duration_mutually_exclusive(self, wav_file):
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_file), "--end", "0.5", "--duration", "0.3"])
        assert result.exit_code != 0
