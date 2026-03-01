"""Sound: unified wrapper around AudioData and SpectralData with lazy conversion."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from soundplay.core.audio import AudioData
from soundplay.core.audio import load as audio_load
from soundplay.core.audio import save as audio_save
from soundplay.core.spectral import (
    SpectralData,
    compute_stft,
    compute_istft,
    load as spx_load,
)


class Sound:
    __slots__ = ('_audio', '_spectral', '_name')

    def __init__(self, audio: AudioData | None = None,
                 spectral: SpectralData | None = None,
                 name: str | None = None):
        if audio is None and spectral is None:
            raise ValueError("Sound requires at least one of audio or spectral")
        self._audio = audio
        self._spectral = spectral
        self._name = name

    # -- Lazy conversion properties ------------------------------------------

    @property
    def audio(self) -> AudioData:
        if self._audio is None:
            samples = compute_istft(self._spectral)
            self._audio = AudioData(samples, self._spectral.sample_rate)
        return self._audio

    @property
    def spectral(self) -> SpectralData:
        if self._spectral is None:
            self._spectral = compute_stft(self._audio.samples, self._audio.sample_rate)
        return self._spectral

    # -- Convenience properties -----------------------------------------------

    @property
    def samples(self) -> np.ndarray:
        return self.audio.samples

    @property
    def stft(self) -> np.ndarray:
        return self.spectral.stft

    @property
    def sr(self) -> int:
        if self._audio is not None:
            return self._audio.sample_rate
        return self._spectral.sample_rate

    @property
    def duration(self) -> float:
        if self._audio is not None:
            return self._audio.duration
        return self._spectral.duration

    @property
    def channels(self) -> int:
        if self._audio is not None:
            return self._audio.channels
        return self._spectral.channels

    # -- Dual-domain transforms -----------------------------------------------

    def gain(self, factor: float) -> Sound:
        from soundplay.tools.gain import _gain_audio, _gain_spectral
        if self._spectral is not None:
            return Sound(spectral=_gain_spectral(self.spectral, factor), name=self._name)
        return Sound(audio=_gain_audio(self.audio, factor), name=self._name)

    def normalize(self, target_db: float = 0.0, mode: str = 'peak') -> Sound:
        from soundplay.tools.normalize import _normalize_audio, _normalize_spectral
        if self._spectral is not None:
            return Sound(spectral=_normalize_spectral(self.spectral, target_db, mode), name=self._name)
        return Sound(audio=_normalize_audio(self.audio, target_db, mode), name=self._name)

    def fade(self, fade_in: float = 0.0, fade_out: float = 0.0) -> Sound:
        from soundplay.tools.fade import _fade_audio, _fade_spectral
        if self._spectral is not None:
            return Sound(spectral=_fade_spectral(self.spectral, fade_in, fade_out), name=self._name)
        return Sound(audio=_fade_audio(self.audio, fade_in, fade_out), name=self._name)

    def reverse(self) -> Sound:
        from soundplay.tools.reverse import _reverse_audio, _reverse_spectral
        if self._spectral is not None:
            return Sound(spectral=_reverse_spectral(self.spectral), name=self._name)
        return Sound(audio=_reverse_audio(self.audio), name=self._name)

    def trim(self, start: float = 0.0, end: float | None = None) -> Sound:
        from soundplay.tools.trim import _trim_audio, _trim_spectral
        end = end if end is not None else self.duration
        if self._spectral is not None:
            return Sound(spectral=_trim_spectral(self.spectral, start, end), name=self._name)
        return Sound(audio=_trim_audio(self.audio, start, end), name=self._name)

    def loop(self, times: int = 2) -> Sound:
        from soundplay.tools.loop import _loop_audio, _loop_spectral
        if self._spectral is not None:
            return Sound(spectral=_loop_spectral(self.spectral, times), name=self._name)
        return Sound(audio=_loop_audio(self.audio, times), name=self._name)

    def pad(self, start: float = 0.0, end: float = 0.0) -> Sound:
        audio = self.audio
        sr = audio.sample_rate
        ch = audio.channels
        parts = []
        if start > 0:
            parts.append(np.zeros((int(round(start * sr)), ch), dtype=np.float32))
        parts.append(audio.samples)
        if end > 0:
            parts.append(np.zeros((int(round(end * sr)), ch), dtype=np.float32))
        return Sound(audio=AudioData(np.concatenate(parts, axis=0), sr), name=self._name)

    def filter(self, type: str, freq: float, freq_hi: float | None = None,
               order: int = 4) -> Sound:
        from soundplay.tools.filter import _apply_filter, _filter_spectral
        if self._spectral is not None:
            return Sound(spectral=_filter_spectral(self.spectral, type, freq, freq_hi), name=self._name)
        filtered = _apply_filter(self.audio.samples, self.audio.sample_rate,
                                 type, freq, freq_hi, order)
        return Sound(audio=AudioData(filtered, self.audio.sample_rate), name=self._name)

    # -- Spectral-only transforms ---------------------------------------------

    def transpose(self, semitones: float) -> Sound:
        from soundplay.tools.transpose import _transpose
        return Sound(spectral=_transpose(self.spectral, semitones), name=self._name)

    def gate(self, threshold_db: float = -40.0) -> Sound:
        from soundplay.tools.gate import _gate
        return Sound(spectral=_gate(self.spectral, threshold_db), name=self._name)

    def denoise(self, noise_start: float = 0.0, noise_end: float = 0.5,
                oversubtract: float = 1.0) -> Sound:
        from soundplay.tools.denoise import _denoise
        return Sound(spectral=_denoise(self.spectral, noise_start, noise_end, oversubtract), name=self._name)

    def stretch(self, factor: float) -> Sound:
        from soundplay.tools.stretch import _stretch
        return Sound(spectral=_stretch(self.spectral, factor), name=self._name)

    def morph(self, other: Sound, blend_start: float = 0.0,
              blend_end: float = 1.0) -> Sound:
        from soundplay.tools.morph import _morph
        return Sound(spectral=_morph(self.spectral, other.spectral, blend_start, blend_end), name=self._name)

    # -- Analysis -------------------------------------------------------------

    def decompose(self, max_notes: int = 12, min_freq: float = 50.0,
                  prominence: float = 15.0, bin_window: int = 2,
                  max_harmonics: int = 16) -> list[Sound]:
        from soundplay.tools.decompose import (
            _mean_magnitude, _find_fundamentals, _harmonic_bins, _apply_mask,
        )
        sd = self.spectral
        nyquist = sd.sample_rate / 2.0
        freq_bins = np.linspace(0, nyquist, sd.bins)
        mean_mag = _mean_magnitude(sd)
        fundamentals = _find_fundamentals(mean_mag, freq_bins, min_freq, max_notes, prominence)
        parts = []
        claimed = np.zeros(sd.bins, dtype=bool)
        for hz, _ in fundamentals:
            mask = _harmonic_bins(hz, freq_bins, bin_window, max_harmonics)
            mask = mask & ~claimed
            claimed |= mask
            component = _apply_mask(sd, mask)
            parts.append(Sound(spectral=component, name=f"{hz:.1f}Hz"))
        return parts

    def pitch_track(self, fmin: float = 50.0, fmax: float = 2000.0) -> list[tuple]:
        from soundplay.tools.pitch_track import _pitch_track
        return _pitch_track(self.spectral, fmin, fmax)

    def rms(self, window: float = 0.1, hop: float | None = None) -> list[tuple]:
        from soundplay.tools.rms import _rms_track
        hop_s = hop if hop is not None else window
        return _rms_track(self.audio, window, hop_s)

    # -- I/O ------------------------------------------------------------------

    def save(self, path: str, format: str | None = None) -> None:
        p = Path(path)
        if p.suffix.lower() == '.spx':
            from soundplay.core.spectral import save as spx_save
            spx_save(self.spectral, p)
        else:
            audio_save(self.audio, p, format=format)

    # -- Playback -------------------------------------------------------------

    def play(self, start: float = 0.0) -> None:
        from soundplay.studio.playback import get_player
        player = get_player()
        player.play(self.audio, start)

    def pause(self) -> None:
        from soundplay.studio.playback import get_player
        get_player().pause()

    def stop(self) -> None:
        from soundplay.studio.playback import get_player
        get_player().stop()

    # -- Visualization --------------------------------------------------------

    def plot(self, **kwargs) -> None:
        from soundplay.studio.viz import show_spectrogram
        show_spectrogram(self, **kwargs)

    def waveform(self, **kwargs) -> None:
        from soundplay.studio.viz import show_waveform
        show_waveform(self, **kwargs)

    # -- Operators ------------------------------------------------------------

    def __mul__(self, factor: float) -> Sound:
        return self.gain(factor)

    def __rmul__(self, factor: float) -> Sound:
        return self.gain(factor)

    def __add__(self, other: Sound) -> Sound:
        return mix(self, other)

    def __repr__(self) -> str:
        name = self._name or 'untitled'
        return f'Sound("{name}", {self.channels}ch, {self.sr}Hz, {self.duration:.3f}s)'


# -- Module-level convenience functions ---------------------------------------

def load(path: str) -> Sound:
    p = Path(path)
    if p.suffix.lower() == '.spx':
        sd = spx_load(p)
        return Sound(spectral=sd, name=p.name)
    ad = audio_load(p)
    return Sound(audio=ad, name=p.name)


def concat(*sounds: Sound) -> Sound:
    from soundplay.tools.concat import _concat_audio
    audios = [s.audio for s in sounds]
    return Sound(audio=_concat_audio(audios), name='concat')


def mix(*sounds: Sound, weights: list[float] | None = None) -> Sound:
    if len(sounds) == 0:
        raise ValueError("mix requires at least one Sound")
    if weights is None:
        w = [1.0 / len(sounds)] * len(sounds)
    else:
        w = weights

    audios = [s.audio for s in sounds]
    sr = audios[0].sample_rate
    max_ch = max(a.channels for a in audios)

    aligned = []
    for a in audios:
        if a.channels < max_ch:
            aligned.append(a.as_stereo() if max_ch == 2 else a)
        else:
            aligned.append(a)

    max_frames = max(a.frames for a in aligned)
    acc = np.zeros((max_frames, max_ch), dtype=np.float32)
    for a, weight in zip(aligned, w):
        acc[:a.frames, :] += a.samples * weight

    result = AudioData(np.clip(acc, -1.0, 1.0).astype(np.float32), sr)
    return Sound(audio=result, name='mix')
