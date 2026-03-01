"""sp-pitch-track: Detect dominant pitch frame-by-frame from a .spx file."""

import sys
import math
import numpy as np
import click
from pathlib import Path
from soundplay.core.spectral import SpectralData, load, read_pipe
from soundplay.core.audio import is_pipe

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _midi_to_note(midi: float) -> str:
    n = round(midi)
    octave = (n // 12) - 1
    name = _NOTE_NAMES[n % 12]
    return f"{name}{octave}"


def _pitch_track(sd: SpectralData, fmin: float, fmax: float) -> list[tuple[float, float, float, str]]:
    sr = sd.sample_rate
    n_fft = sd.n_fft
    hop = sd.hop_length
    bins = sd.bins

    freq_resolution = sr / n_fft
    bin_min = max(0, int(math.ceil(fmin / freq_resolution)))
    bin_max = min(bins - 1, int(math.floor(fmax / freq_resolution)))

    # Mean magnitude across channels
    mag = np.abs(sd.stft).mean(axis=0)  # (frames, bins)

    rows = []
    for frame_idx in range(sd.frames):
        time_s = frame_idx * hop / sr
        frame_mag = mag[frame_idx, bin_min:bin_max + 1]
        if frame_mag.max() == 0:
            freq_hz = 0.0
            midi = float('nan')
            note = '—'
        else:
            peak_offset = int(np.argmax(frame_mag))
            peak_bin = bin_min + peak_offset
            freq_hz = peak_bin * freq_resolution
            if freq_hz > 0:
                midi = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
                note = _midi_to_note(midi)
            else:
                midi = float('nan')
                note = '—'
        rows.append((time_s, freq_hz, midi, note))
    return rows


@click.command()
@click.argument('input', default=None, required=False)
@click.option('--output', default=None, help='Output file. Defaults to stdout.')
@click.option('--fmin', default=50.0, show_default=True, help='Minimum frequency (Hz).')
@click.option('--fmax', default=2000.0, show_default=True, help='Maximum frequency (Hz).')
@click.option('--format', 'fmt', default='csv', show_default=True,
              type=click.Choice(['csv', 'tsv']), help='Output format.')
def main(input, output, fmin, fmax, fmt):
    """Detect dominant pitch frame-by-frame from a .spx file.

    Outputs a table with columns: time_s, freq_hz, midi_note, note_name.

    \b
    INPUT   .spx file or '-' for stdin pipe.

    \b
    Examples:
      sp-pitch-track song.spx
      sp-pitch-track song.spx --output pitches.csv
      sp-pitch-track --fmin 80 --fmax 1000 --format tsv song.spx
      cat song.spx | sp-pitch-track
    """
    if input == '-' or (input is None and is_pipe(sys.stdin)):
        sd = read_pipe()
    elif input is None:
        raise click.UsageError("No input: provide an INPUT file or pipe via stdin")
    else:
        sd = load(input)

    rows = _pitch_track(sd, fmin, fmax)

    sep = ',' if fmt == 'csv' else '\t'
    header = sep.join(['time_s', 'freq_hz', 'midi_note', 'note_name'])
    lines = [header]
    for time_s, freq_hz, midi, note in rows:
        midi_str = f"{midi:.2f}" if not math.isnan(midi) else 'nan'
        lines.append(f"{time_s:.4f}{sep}{freq_hz:.2f}{sep}{midi_str}{sep}{note}")

    text = '\n'.join(lines) + '\n'

    if output:
        Path(output).write_text(text)
        click.echo(f"Wrote {len(rows)} frames → {output}", err=True)
    else:
        sys.stdout.write(text)
