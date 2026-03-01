"""Tests for sp-reverse tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import load, save
from soundplay.core.spectral import save as spx_save, load as spx_load
from soundplay.tools.reverse import main


class TestReverseCLI:
    def test_audio_reverse(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert loaded.frames == mono_audio.frames
        # Reversed samples should match flipped original (with PCM quantization tolerance)
        np.testing.assert_allclose(loaded.samples, mono_audio.samples[::-1], atol=1e-3)

    def test_double_reverse_identity(self, mono_audio, tmp_path):
        inp = tmp_path / "in.wav"
        mid = tmp_path / "mid.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, inp)
        runner = CliRunner()
        runner.invoke(main, [str(inp), str(mid)])
        runner.invoke(main, [str(mid), str(out)])
        loaded = load(out)
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-3)

    def test_spectral_reverse(self, spectral_data, tmp_path):
        inp = tmp_path / "in.spx"
        out = tmp_path / "out.spx"
        spx_save(spectral_data, inp)
        runner = CliRunner()
        result = runner.invoke(main, [str(inp), str(out)])
        assert result.exit_code == 0
        loaded = spx_load(out)
        # STFT frames should be reversed
        np.testing.assert_allclose(
            np.abs(loaded.stft[0]),
            np.abs(spectral_data.stft[0, ::-1, :]),
            atol=1e-5,
        )
