"""Tests for soundplay.core.audio."""

import io
import numpy as np
import pytest

from soundplay.core.audio import (
    AudioData, load, save, read_pipe, write_pipe,
)


class TestAudioDataProperties:
    def test_mono_channels(self, mono_audio):
        assert mono_audio.channels == 1

    def test_stereo_channels(self, stereo_audio):
        assert stereo_audio.channels == 2

    def test_frames(self, mono_audio):
        assert mono_audio.frames == 16000

    def test_duration(self, mono_audio):
        assert mono_audio.duration == pytest.approx(1.0)


class TestChannelConversion:
    def test_mono_to_stereo(self, mono_audio):
        st = mono_audio.as_stereo()
        assert st.channels == 2
        np.testing.assert_array_equal(st.samples[:, 0], st.samples[:, 1])

    def test_stereo_to_mono(self, stereo_audio):
        m = stereo_audio.as_mono()
        assert m.channels == 1
        expected = stereo_audio.samples.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(m.samples, expected, atol=1e-6)

    def test_mono_as_mono_returns_self(self, mono_audio):
        assert mono_audio.as_mono() is mono_audio

    def test_stereo_as_stereo_returns_self(self, stereo_audio):
        assert stereo_audio.as_stereo() is stereo_audio


class TestFileIO:
    def test_wav_roundtrip(self, mono_audio, tmp_path):
        p = tmp_path / "rt.wav"
        save(mono_audio, p)
        loaded = load(p)
        assert loaded.sample_rate == mono_audio.sample_rate
        assert loaded.frames == mono_audio.frames
        # PCM_16 quantization limits precision
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-4)

    def test_flac_roundtrip(self, mono_audio, tmp_path):
        p = tmp_path / "rt.flac"
        save(mono_audio, p)
        loaded = load(p)
        assert loaded.frames == mono_audio.frames
        np.testing.assert_allclose(loaded.samples, mono_audio.samples, atol=1e-4)

    def test_ogg_roundtrip(self, mono_audio, tmp_path):
        p = tmp_path / "rt.ogg"
        save(mono_audio, p)
        loaded = load(p)
        assert loaded.sample_rate == mono_audio.sample_rate
        # OGG is lossy so wider tolerance
        assert loaded.frames == pytest.approx(mono_audio.frames, abs=100)

    def test_unsupported_format_raises(self, mono_audio, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            save(mono_audio, tmp_path / "test.xyz")


class TestPipeRoundtrip:
    def test_pipe_roundtrip(self, mono_audio):
        buf = io.BytesIO()
        write_pipe(mono_audio, buf)
        buf.seek(0)
        loaded = read_pipe(buf)
        assert loaded.sample_rate == mono_audio.sample_rate
        assert loaded.channels == mono_audio.channels
        np.testing.assert_array_equal(loaded.samples, mono_audio.samples)

    def test_pipe_roundtrip_stereo(self, stereo_audio):
        buf = io.BytesIO()
        write_pipe(stereo_audio, buf)
        buf.seek(0)
        loaded = read_pipe(buf)
        assert loaded.channels == 2
        np.testing.assert_array_equal(loaded.samples, stereo_audio.samples)

    def test_incomplete_header_raises(self):
        buf = io.BytesIO(b'\x00' * 4)
        with pytest.raises(ValueError, match="Incomplete"):
            read_pipe(buf)

    def test_bad_magic_raises(self):
        buf = io.BytesIO(b'XXXX' + b'\x00' * 12)
        with pytest.raises(ValueError, match="Invalid pipe magic"):
            read_pipe(buf)
