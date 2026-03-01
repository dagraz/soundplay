"""sp-transpose: Pitch-shift a .spx file by shifting frequency bins."""

import sys
import re
import numpy as np
import click
from pathlib import Path
from scipy.interpolate import interp1d
from soundplay.core.spectral import SpectralData, load, read_pipe, save, write_pipe
from soundplay.core.audio import is_pipe


def _parse_semitones(value: str) -> float:
    """Parse semitone amount: '12', '-3', '100c' (cents), '-50c'."""
    s = str(value).strip()
    m = re.fullmatch(r'(-?\d+(?:\.\d+)?)c', s)
    if m:
        return float(m.group(1)) / 100.0
    return float(s)


def _transpose(sd: SpectralData, semitones: float) -> SpectralData:
    factor = 2.0 ** (semitones / 12.0)
    bins = sd.bins
    src_bins = np.arange(bins, dtype=np.float64)
    # Output bin b samples input at b / factor
    dst_bins = src_bins / factor

    channels, frames, _ = sd.stft.shape
    new_stft = np.zeros_like(sd.stft)

    for ch in range(channels):
        real = sd.stft[ch].real  # (frames, bins)
        imag = sd.stft[ch].imag
        interp_r = interp1d(src_bins, real, axis=1, bounds_error=False, fill_value=0.0)
        interp_i = interp1d(src_bins, imag, axis=1, bounds_error=False, fill_value=0.0)
        new_stft[ch] = (interp_r(dst_bins) + 1j * interp_i(dst_bins)).astype(np.complex64)

    return SpectralData(
        stft=new_stft,
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('semitones')
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
def main(semitones, input, output):
    """Pitch-shift a .spx file by shifting frequency bins.

    \b
    SEMITONES  Float semitones (e.g. 12, -3.5) or cents with 'c' suffix (e.g. 100c).
    INPUT      .spx file or '-' for stdin pipe.
    OUTPUT     .spx file or '-' for stdout pipe.
               Default: INPUT with '_t+N' or '_t-N' suffix.

    \b
    Examples:
      sp-transpose 12 song.spx octave_up.spx
      sp-transpose -7 song.spx - | sp-resynth - out.wav
      sp-transpose 100c song.spx song_100c.spx
    """
    try:
        st = _parse_semitones(semitones)
    except ValueError:
        raise click.UsageError(f"Invalid SEMITONES value: {semitones!r}")

    if input == '-' or (input is None and is_pipe(sys.stdin)):
        sd = read_pipe()
        stem = 'stdin'
        src_dir = Path('.')
    elif input is None:
        raise click.UsageError("No input: provide an INPUT file or pipe via stdin")
    else:
        sd = load(input)
        stem = Path(input).stem
        src_dir = Path(input).parent

    result = _transpose(sd, st)

    if output is None:
        sign = '+' if st >= 0 else ''
        suffix = f"_t{sign}{st:g}"
        output = str(src_dir / f"{stem}{suffix}.spx")

    if output == '-':
        write_pipe(result)
    else:
        save(result, output)
        click.echo(f"Transposed {st:+g} semitones → {output}", err=True)
