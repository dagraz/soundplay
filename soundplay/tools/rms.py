"""sp-rms: Measure RMS and peak levels over sliding windows."""

import sys
import math
import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import AudioData, load, read_pipe, is_pipe

_SILENCE_DB = -96.0


def _db(linear: float) -> float:
    if linear <= 0.0:
        return _SILENCE_DB
    return max(_SILENCE_DB, 20.0 * math.log10(linear))


def _rms_track(audio: AudioData, window_s: float, hop_s: float) -> list[tuple[float, float, float]]:
    sr = audio.sample_rate
    win_frames = max(1, int(round(window_s * sr)))
    hop_frames = max(1, int(round(hop_s * sr)))
    total = audio.frames
    # Mono mix for measurement
    if audio.channels > 1:
        mono = audio.samples.mean(axis=1)
    else:
        mono = audio.samples[:, 0]

    rows = []
    pos = 0
    while pos < total:
        segment = mono[pos:pos + win_frames]
        time_s = pos / sr
        rms_lin = float(np.sqrt(np.mean(segment ** 2)))
        peak_lin = float(np.max(np.abs(segment)))
        rows.append((time_s, _db(rms_lin), _db(peak_lin)))
        pos += hop_frames
    return rows


@click.command()
@click.argument('input', default=None, required=False)
@click.option('--output', default=None, help='Output file. Defaults to stdout.')
@click.option('--window', default=0.1, show_default=True,
              help='Analysis window size (seconds).')
@click.option('--hop', default=None, type=float,
              help='Hop between windows (seconds). Defaults to --window.')
@click.option('--format', 'fmt', default='csv', show_default=True,
              type=click.Choice(['csv', 'tsv']), help='Output format.')
def main(input, output, window, hop, fmt):
    """Measure RMS and peak levels over sliding windows.

    Outputs a table with columns: time_s, rms_db, peak_db.
    Silence is reported as -96.0 dBFS.

    \b
    INPUT   Audio file (wav, flac, etc.) or '-' for stdin pipe.

    \b
    Examples:
      sp-rms song.wav
      sp-rms --window 0.5 --hop 0.1 song.wav --output levels.csv
      cat song.wav | sp-rms --format tsv
    """
    if input == '-' or (input is None and is_pipe(sys.stdin)):
        audio = read_pipe()
    elif input is None:
        raise click.UsageError("No input: provide an INPUT file or pipe via stdin")
    else:
        audio = load(input)

    hop_s = hop if hop is not None else window

    rows = _rms_track(audio, window, hop_s)

    sep = ',' if fmt == 'csv' else '\t'
    header = sep.join(['time_s', 'rms_db', 'peak_db'])
    lines = [header]
    for time_s, rms_db, peak_db in rows:
        lines.append(f"{time_s:.4f}{sep}{rms_db:.2f}{sep}{peak_db:.2f}")

    text = '\n'.join(lines) + '\n'

    if output:
        Path(output).write_text(text)
        click.echo(f"Wrote {len(rows)} windows → {output}", err=True)
    else:
        sys.stdout.write(text)
