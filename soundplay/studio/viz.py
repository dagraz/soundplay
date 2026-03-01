"""Visualization helpers for Sound objects."""

from __future__ import annotations

import numpy as np


def show_spectrogram(sound, channel: int = -1, db_range: float = 80.0,
                     fmin: float = 20.0, fmax: float | None = None,
                     colormap: str = 'inferno', notes: bool = False,
                     note_labels: bool = False, all_notes: bool = False,
                     title: str | None = None, figsize: tuple = (12, 6)):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from soundplay.tools.plot import note_guidelines

    sd = sound.spectral

    if channel == -1:
        mag_ch = np.abs(sd.stft).mean(axis=0)
    else:
        mag_ch = np.abs(sd.stft[channel])

    db = 20.0 * np.log10(np.maximum(mag_ch, 1e-10))
    db_max = db.max()
    db_min = db_max - db_range
    db = np.clip(db, db_min, db_max)

    nyquist = sd.sample_rate / 2.0
    freq_bins = np.linspace(0, nyquist, sd.bins)
    time_axis = np.arange(sd.frames) * sd.hop_length / sd.sample_rate

    f_max = min(fmax, nyquist) if fmax is not None else nyquist

    freq_mask = (freq_bins >= fmin) & (freq_bins <= f_max)
    freq_display = freq_bins[freq_mask]
    db_display = db[:, freq_mask]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        time_axis, freq_display, db_display.T,
        shading='auto', cmap=colormap,
        vmin=db_min, vmax=db_max, rasterized=True,
    )
    ax.set_yscale('log')
    ax.set_ylim(max(fmin, freq_display[freq_display > 0].min()), f_max)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 100 else f"{v:.1f}"
    ))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title or repr(sound))
    fig.colorbar(im, ax=ax, pad=0.02).set_label('dB')

    if notes or note_labels or all_notes:
        guide_notes = note_guidelines(fmin, f_max)
        for hz, name, is_c in guide_notes:
            if not all_notes and not is_c:
                ax.axhline(hz, color='white', linewidth=0.3, alpha=0.25, linestyle='--')
                continue
            lw = 0.8 if is_c else 0.4
            alpha = 0.6 if is_c else 0.4
            ax.axhline(hz, color='white', linewidth=lw, alpha=alpha, linestyle='--')
            if note_labels or all_notes:
                ax.text(time_axis[-1], hz, f' {name}', color='white', fontsize=6,
                        va='center', ha='left', clip_on=True)

    fig.tight_layout()
    plt.show()


def show_waveform(sound, channel: int = -1, title: str | None = None,
                  figsize: tuple = (12, 4)):
    import matplotlib.pyplot as plt

    audio = sound.audio
    sr = audio.sample_rate
    samples = audio.samples
    time_axis = np.arange(audio.frames) / sr

    if channel == -1 and audio.channels > 1:
        data = samples.mean(axis=1)
        ch_label = 'mono mix'
    elif channel >= 0 and channel < audio.channels:
        data = samples[:, channel]
        ch_label = f'ch {channel}'
    else:
        data = samples[:, 0]
        ch_label = 'ch 0'

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, data, linewidth=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title or f'{repr(sound)} [{ch_label}]')
    ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    plt.show()
