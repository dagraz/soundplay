"""Tests for soundplay.studio.sound — the Sound wrapper class."""

import numpy as np
import pytest

from soundplay.core.audio import AudioData
from soundplay.core.spectral import SpectralData, compute_stft
from soundplay.studio.sound import Sound, load, concat, mix


SR = 16000


def _sine(freq: float, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (0.8 * np.sin(2 * np.pi * freq * t)).reshape(-1, 1).astype(np.float32)


@pytest.fixture
def audio():
    return AudioData(_sine(440), SR)


@pytest.fixture
def spectral(audio):
    return compute_stft(audio.samples, audio.sample_rate)


@pytest.fixture
def sound_from_audio(audio):
    return Sound(audio=audio, name='test')


@pytest.fixture
def sound_from_spectral(spectral):
    return Sound(spectral=spectral, name='test_spx')


# -- Construction -------------------------------------------------------------

class TestConstruction:
    def test_requires_data(self):
        with pytest.raises(ValueError):
            Sound()

    def test_from_audio(self, audio):
        s = Sound(audio=audio, name='a')
        assert s._audio is not None
        assert s._spectral is None

    def test_from_spectral(self, spectral):
        s = Sound(spectral=spectral, name='s')
        assert s._audio is None
        assert s._spectral is not None


# -- Lazy conversion ----------------------------------------------------------

class TestLazyConversion:
    def test_audio_to_spectral(self, sound_from_audio):
        assert sound_from_audio._spectral is None
        sd = sound_from_audio.spectral
        assert isinstance(sd, SpectralData)
        assert sound_from_audio._spectral is not None

    def test_spectral_to_audio(self, sound_from_spectral):
        assert sound_from_spectral._audio is None
        ad = sound_from_spectral.audio
        assert isinstance(ad, AudioData)
        assert sound_from_spectral._audio is not None

    def test_cached_after_first_access(self, sound_from_audio):
        sd1 = sound_from_audio.spectral
        sd2 = sound_from_audio.spectral
        assert sd1 is sd2


# -- Properties ---------------------------------------------------------------

class TestProperties:
    def test_sr(self, sound_from_audio):
        assert sound_from_audio.sr == SR

    def test_duration(self, sound_from_audio):
        assert abs(sound_from_audio.duration - 1.0) < 0.01

    def test_channels_mono(self, sound_from_audio):
        assert sound_from_audio.channels == 1

    def test_samples_shape(self, sound_from_audio):
        assert sound_from_audio.samples.shape == (SR, 1)

    def test_stft_shape(self, sound_from_audio):
        stft = sound_from_audio.stft
        assert stft.ndim == 3  # (channels, frames, bins)


# -- Transforms ---------------------------------------------------------------

class TestTransforms:
    def test_gain(self, sound_from_audio):
        s2 = sound_from_audio.gain(0.5)
        assert isinstance(s2, Sound)
        assert s2 is not sound_from_audio
        np.testing.assert_allclose(
            np.max(np.abs(s2.samples)),
            np.max(np.abs(sound_from_audio.samples)) * 0.5,
            atol=0.01,
        )

    def test_gain_spectral_domain(self, sound_from_spectral):
        s2 = sound_from_spectral.gain(0.5)
        ratio = np.mean(np.abs(s2.stft)) / np.mean(np.abs(sound_from_spectral.stft))
        np.testing.assert_allclose(ratio, 0.5, atol=0.01)

    def test_normalize(self, sound_from_audio):
        s2 = sound_from_audio.normalize(target_db=-6.0, mode='peak')
        peak = np.max(np.abs(s2.samples))
        expected = 10.0 ** (-6.0 / 20.0)
        np.testing.assert_allclose(peak, expected, atol=0.02)

    def test_fade(self, sound_from_audio):
        s2 = sound_from_audio.fade(fade_in=0.1, fade_out=0.1)
        assert isinstance(s2, Sound)
        # First sample should be near zero (faded in)
        assert abs(s2.samples[0, 0]) < 0.01

    def test_reverse(self, sound_from_audio):
        s2 = sound_from_audio.reverse()
        np.testing.assert_allclose(
            s2.samples[:, 0],
            sound_from_audio.samples[::-1, 0],
            atol=1e-6,
        )

    def test_trim(self, sound_from_audio):
        s2 = sound_from_audio.trim(0.2, 0.5)
        expected_dur = 0.3
        assert abs(s2.duration - expected_dur) < 0.01

    def test_loop(self, sound_from_audio):
        s2 = sound_from_audio.loop(3)
        assert abs(s2.duration - 3.0 * sound_from_audio.duration) < 0.01

    def test_pad(self, sound_from_audio):
        s2 = sound_from_audio.pad(start=0.5, end=0.5)
        assert abs(s2.duration - 2.0) < 0.01

    def test_filter_lowpass(self, sound_from_audio):
        s2 = sound_from_audio.filter('lowpass', 200.0)
        assert isinstance(s2, Sound)

    def test_transpose(self, sound_from_audio):
        s2 = sound_from_audio.transpose(12)
        assert isinstance(s2, Sound)
        assert s2._spectral is not None

    def test_gate(self, sound_from_audio):
        s2 = sound_from_audio.gate(-20.0)
        assert isinstance(s2, Sound)

    def test_denoise(self, sound_from_audio):
        s2 = sound_from_audio.denoise(0.0, 0.2, 1.0)
        assert isinstance(s2, Sound)

    def test_stretch(self, sound_from_audio):
        s2 = sound_from_audio.stretch(2.0)
        assert abs(s2.duration - 2.0 * sound_from_audio.duration) < 0.1

    def test_morph(self, sound_from_audio):
        other = Sound(audio=AudioData(_sine(880), SR), name='other')
        morphed = sound_from_audio.morph(other, 0.0, 1.0)
        assert isinstance(morphed, Sound)

    def test_chaining(self, sound_from_audio):
        s2 = sound_from_audio.trim(0.1, 0.5).gain(0.5).reverse()
        assert isinstance(s2, Sound)
        assert abs(s2.duration - 0.4) < 0.01


# -- Analysis -----------------------------------------------------------------

class TestAnalysis:
    def test_pitch_track(self, sound_from_audio):
        rows = sound_from_audio.pitch_track(fmin=400, fmax=500)
        assert len(rows) > 0
        # First entry is (time_s, freq_hz, midi, note)
        assert len(rows[0]) == 4

    def test_rms(self, sound_from_audio):
        rows = sound_from_audio.rms(window=0.1)
        assert len(rows) > 0
        assert len(rows[0]) == 3  # (time_s, rms_db, peak_db)

    def test_decompose(self, sound_from_audio):
        parts = sound_from_audio.decompose(max_notes=2)
        assert isinstance(parts, list)
        for p in parts:
            assert isinstance(p, Sound)


# -- Operators ----------------------------------------------------------------

class TestOperators:
    def test_mul(self, sound_from_audio):
        s2 = sound_from_audio * 0.5
        np.testing.assert_allclose(
            np.max(np.abs(s2.samples)),
            np.max(np.abs(sound_from_audio.samples)) * 0.5,
            atol=0.01,
        )

    def test_rmul(self, sound_from_audio):
        s2 = 0.5 * sound_from_audio
        assert isinstance(s2, Sound)

    def test_add(self, sound_from_audio):
        s2 = sound_from_audio + sound_from_audio
        assert isinstance(s2, Sound)
        assert abs(s2.duration - sound_from_audio.duration) < 0.01

    def test_repr(self, sound_from_audio):
        r = repr(sound_from_audio)
        assert 'test' in r
        assert '1ch' in r
        assert f'{SR}Hz' in r


# -- I/O ----------------------------------------------------------------------

class TestIO:
    def test_load_wav(self, tmp_path):
        from soundplay.core.audio import save
        ad = AudioData(_sine(440), SR)
        p = tmp_path / 'test.wav'
        save(ad, p)
        s = load(str(p))
        assert isinstance(s, Sound)
        assert abs(s.duration - 1.0) < 0.01

    def test_load_spx(self, tmp_path):
        from soundplay.core.spectral import save as spx_save
        sd = compute_stft(_sine(440), SR)
        p = tmp_path / 'test.spx'
        spx_save(sd, p)
        s = load(str(p))
        assert isinstance(s, Sound)
        assert s._spectral is not None
        assert s._audio is None

    def test_save_wav(self, sound_from_audio, tmp_path):
        p = tmp_path / 'out.wav'
        sound_from_audio.save(str(p))
        assert p.exists()

    def test_save_spx(self, sound_from_audio, tmp_path):
        p = tmp_path / 'out.spx'
        sound_from_audio.save(str(p))
        assert p.exists()


# -- Module-level functions ---------------------------------------------------

class TestModuleFunctions:
    def test_concat(self, sound_from_audio):
        s2 = concat(sound_from_audio, sound_from_audio)
        assert abs(s2.duration - 2.0 * sound_from_audio.duration) < 0.01

    def test_mix_equal(self, sound_from_audio):
        s2 = mix(sound_from_audio, sound_from_audio)
        assert abs(s2.duration - sound_from_audio.duration) < 0.01

    def test_mix_weights(self, sound_from_audio):
        s2 = mix(sound_from_audio, sound_from_audio, weights=[0.7, 0.3])
        assert isinstance(s2, Sound)

    def test_mix_empty_raises(self):
        with pytest.raises(ValueError):
            mix()
