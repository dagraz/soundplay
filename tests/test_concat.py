"""Tests for sp-concat tool."""

import numpy as np
from click.testing import CliRunner

from soundplay.core.audio import AudioData, load, save
from soundplay.tools.concat import main


class TestConcatCLI:
    def test_basic_concat(self, mono_audio, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, a)
        save(mono_audio, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0, result.output
        loaded = load(out)
        assert loaded.frames == mono_audio.frames * 2

    def test_three_files(self, mono_audio, tmp_path):
        files = []
        for i in range(3):
            p = tmp_path / f"f{i}.wav"
            save(mono_audio, p)
            files.append(str(p))
        out = tmp_path / "out.wav"
        runner = CliRunner()
        result = runner.invoke(main, files + ["-o", str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.frames == mono_audio.frames * 3

    def test_sample_rate_mismatch(self, mono_audio, tmp_path):
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        save(mono_audio, a)
        different_sr = AudioData(mono_audio.samples, 22050)
        save(different_sr, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-o", str(tmp_path / "out.wav")])
        assert result.exit_code != 0

    def test_mono_stereo_upmix(self, mono_audio, stereo_audio, tmp_path):
        a = tmp_path / "mono.wav"
        b = tmp_path / "stereo.wav"
        out = tmp_path / "out.wav"
        save(mono_audio, a)
        save(stereo_audio, b)
        runner = CliRunner()
        result = runner.invoke(main, [str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0
        loaded = load(out)
        assert loaded.channels == 2
        assert loaded.frames == mono_audio.frames + stereo_audio.frames

    def test_single_file_errors(self, wav_file):
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_file)])
        assert result.exit_code != 0
