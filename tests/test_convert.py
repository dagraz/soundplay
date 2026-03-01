"""Tests for sp-convert tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import load, save
from soundplay.tools.convert import main


class TestConvertCLI:
    def test_wav_to_flac(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.flac"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert loaded.sample_rate == mono_audio.sample_rate
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-3)

    def test_wav_to_ogg(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.ogg"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.sample_rate == mono_audio.sample_rate

    def test_flac_to_wav(self, mono_audio, tmp_path):
        inp = tmp_path / "in.flac"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-3)

    def test_format_flag(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), "--format", "flac"])
        assert result.exit_code == 0
        out = tmp_path / "in.flac"
        assert out.exists()

    def test_same_file_errors(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(inp)])
        assert result.exit_code != 0

    def test_no_output_no_format_errors(self, wav_file):
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_file)])
        assert result.exit_code != 0
