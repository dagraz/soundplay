"""sp-plot: Visualize a .spx spectral file as a PNG spectrogram."""

import sys
import numpy as np
import click
from soundplay.core.spectral import load, read_pipe
from soundplay.core.audio import is_pipe

# Equal temperament note names (chromatic scale)
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_hz(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz. A4 = MIDI 69 = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def midi_to_name(midi: int) -> str:
    """Convert MIDI note number to note name, e.g. 69 -> 'A4'."""
    octave = (midi // 12) - 1
    name = _NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def note_guidelines(fmin: float, fmax: float):
    """
    Return list of (frequency, name, is_c) for all equal-temperament notes
    within [fmin, fmax]. MIDI range 0-127 covers ~8 Hz to ~12 kHz.
    """
    notes = []
    for midi in range(0, 128):
        hz = midi_to_hz(midi)
        if hz < fmin:
            continue
        if hz > fmax:
            break
        is_c = (midi % 12) == 0
        notes.append((hz, midi_to_name(midi), is_c))
    return notes


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--channel', default=0, show_default=True,
              help='Channel index to display, or -1 to average all channels.')
@click.option('--notes', is_flag=True,
              help='Draw horizontal guidelines at standard note frequencies.')
@click.option('--note-labels', is_flag=True,
              help='Label note guidelines (implies --notes). C notes are always labeled; '
                   'use with --all-notes to label every note.')
@click.option('--all-notes', is_flag=True,
              help='Draw/label every note, not just C notes.')
@click.option('--db-range', default=80.0, show_default=True,
              help='Dynamic range in dB to display (top N dB of peak).')
@click.option('--fmin', default=20.0, show_default=True,
              help='Minimum frequency to display (Hz).')
@click.option('--fmax', default=None, type=float,
              help='Maximum frequency to display (Hz). Defaults to Nyquist.')
@click.option('--colormap', default='inferno', show_default=True,
              help='Matplotlib colormap name.')
@click.option('--title', default=None,
              help='Plot title. Defaults to input filename.')
@click.option('--dpi', default=150, show_default=True,
              help='Output image DPI.')
@click.option('--width', default=12.0, show_default=True,
              help='Figure width in inches.')
@click.option('--height', default=6.0, show_default=True,
              help='Figure height in inches.')
def main(input, output, channel, notes, note_labels, all_notes, db_range,
         fmin, fmax, colormap, title, dpi, width, height):
    """Visualize a .spx spectral file as a PNG spectrogram.

    \b
    INPUT   .spx file or '-' for stdin pipe.
    OUTPUT  PNG file. Defaults to INPUT with .png extension.

    \b
    Examples:
      sp-plot song.spx song.png
      sp-plot --notes --note-labels song.spx song.png
      sp-plot --all-notes --fmin 80 --fmax 8000 song.spx song.png
      cat song.spx | sp-plot - song.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Load spectral data
    if input == '-' or (input is None and is_pipe(sys.stdin)):
        sd = read_pipe()
        src_name = 'stdin'
    elif input is None:
        raise click.UsageError("No input: provide an INPUT file or pipe via stdin")
    else:
        sd = load(input)
        src_name = input

    # Determine output path
    if output is None:
        if src_name == 'stdin':
            raise click.UsageError("Specify an OUTPUT file when reading from stdin")
        from pathlib import Path
        output = str(Path(src_name).with_suffix('.png'))

    # Select channel
    if channel == -1:
        mag_ch = np.abs(sd.stft).mean(axis=0)  # (frames, bins)
    else:
        if channel >= sd.channels:
            raise click.BadParameter(
                f"Channel {channel} out of range (file has {sd.channels} channel(s))",
                param_hint='--channel'
            )
        mag_ch = np.abs(sd.stft[channel])  # (frames, bins)

    # Convert to dB
    db = 20.0 * np.log10(np.maximum(mag_ch, 1e-10))
    db_max = db.max()
    db_min = db_max - db_range
    db = np.clip(db, db_min, db_max)

    # Frequency and time axes
    nyquist = sd.sample_rate / 2.0
    freq_bins = np.linspace(0, nyquist, sd.bins)
    # Time axis: centre of each frame
    time_axis = np.arange(sd.frames) * sd.hop_length / sd.sample_rate

    f_max = fmax if fmax is not None else nyquist
    f_max = min(f_max, nyquist)

    # Trim to frequency range for display
    freq_mask = (freq_bins >= fmin) & (freq_bins <= f_max)
    freq_display = freq_bins[freq_mask]
    db_display = db[:, freq_mask]  # (frames, display_bins)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(width, height))

    # pcolormesh: X=time, Y=freq, C=magnitude
    # db_display is (frames, bins) → we need (bins, frames) for pcolormesh
    im = ax.pcolormesh(
        time_axis, freq_display, db_display.T,
        shading='auto', cmap=colormap,
        vmin=db_min, vmax=db_max,
        rasterized=True,
    )

    ax.set_yscale('log')
    ax.set_ylim(max(fmin, freq_display[freq_display > 0].min()), f_max)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 100 else f"{v:.1f}"
    ))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    plot_title = title or src_name
    if sd.channels > 1 and channel >= 0:
        plot_title += f'  [ch {channel}]'
    ax.set_title(plot_title)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('dB')

    # -----------------------------------------------------------------------
    # Note guidelines
    # -----------------------------------------------------------------------
    if notes or note_labels or all_notes:
        guide_notes = note_guidelines(fmin, f_max)

        for hz, name, is_c in guide_notes:
            if not all_notes and not is_c:
                # Draw thin line for every note, but only label C notes
                ax.axhline(hz, color='white', linewidth=0.3, alpha=0.25, linestyle='--')
                continue

            # C notes (or all notes if --all-notes): more visible line
            lw = 0.8 if is_c else 0.4
            alpha = 0.6 if is_c else 0.4
            ax.axhline(hz, color='white', linewidth=lw, alpha=alpha, linestyle='--')

            # Label
            if note_labels or all_notes:
                ax.text(
                    time_axis[-1], hz, f' {name}',
                    color='white', fontsize=6,
                    va='center', ha='left',
                    clip_on=True,
                )

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    click.echo(f"Wrote spectrogram → {output}", err=True)
