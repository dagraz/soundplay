"""Tests for sp-mix tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import AudioData, load, save
from soundplay.tools.mix import main


class TestMixCLI:
    def test_equal_mix(self, mono_audio, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, a)
        save(mono_audio, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert loaded.frames == mono_audio.frames

    def test_weighted_mix(self, mono_audio, silence_audio, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, a)
        save(silence_audio, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-w", "1.0,0.0", "-o", str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        # Mixing with silence at weight 0 should give original
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-3)

    def test_shorter_file_padded(self, mono_audio, tmp_path):
        a = tmp_path / "long.wav"
        b = tmp_path / "short.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, a)
        short = AudioData(mono_audio.samples[:8000], mono_audio.sample_rate)
        save(short, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.frames == mono_audio.frames

    def test_weight_count_mismatch_errors(self, mono_audio, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        save(mono_audio, a)
        save(mono_audio, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-w", "0.5,0.3,0.2", "-o", str(tmp_path / "out.wav")])
        assert result.exit_code != 0

    def test_single_file_errors(self, wav_file):
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_file)])
        assert result.exit_code != 0
