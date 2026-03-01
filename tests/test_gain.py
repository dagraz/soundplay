"""Tests for sp-gain tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import AudioData, load, save
from soundplay.tools.gain import main, _parse_gain


class TestParseGain:
    def test_linear(self):
        assert _parse_gain("0.5") == 0.5

    def test_db_negative(self):
        assert _parse_gain("-6dB") == np.float64(10.0 ** (-6.0 / 20.0))

    def test_db_positive(self):
        assert _parse_gain("+3dB") == np.float64(10.0 ** (3.0 / 20.0))

    def test_db_case_insensitive(self):
        assert _parse_gain("-6DB") == _parse_gain("-6dB")


class TestGainCLI:
    def test_half_gain(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, ["0.5", str(inp), str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert np.max(np.abs(loaded.samples)) < np.max(np.abs(mono_audio.samples))

    def test_double_gain(self, mono_audio, tmp_path):
        # Use a quiet signal so doubling doesn't clip
        quiet = AudioData(mono_audio.samples * 0.3, mono_audio.sample_rate)
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(quiet, inp)
        runner = CliRunner()
        result = runner.invoke(main, ["2.0", str(inp), str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert np.max(np.abs(loaded.samples)) > np.max(np.abs(quiet.samples)) * 1.5

    def test_clipping(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, ["10.0", str(inp), str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        assert np.max(np.abs(loaded.samples)) <= 1.0 + 1e-4  # PCM quantization tolerance

    def test_db_gain(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, ["--", "-6dB", str(inp), str(out)])
        assert result.exit_code == 0

    def test_spectral_mode(self, spectral_data, tmp_path):
        from soundplay.core.spectral import save as spx_save, load as spx_load
        inp = tmp_path / "in.spx"
        out = tmp_path / "out.spx"
        spx_save(spectral_data, inp)
        runner = CliRunner()
        result = runner.invoke(main, ["0.5", str(inp), str(out)])
        assert result.exit_code == 0
        loaded = spx_load(out)
        # Magnitude should be roughly half
        ratio = np.mean(np.abs(loaded.stft)) / np.mean(np.abs(spectral_data.stft))
        assert ratio == np.float64(0.5).item().__class__(0.5)
        np.testing.assert_allclose(ratio, 0.5, atol=0.01)
