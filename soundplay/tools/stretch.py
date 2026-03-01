"""sp-stretch: Time-stretch a .spx file without pitch change."""

import sys
import numpy as np
import click
from pathlib import Path
from scipy.interpolate import interp1d
from soundplay.core.spectral import SpectralData, load, read_pipe, save, write_pipe
from soundplay.core.audio import is_pipe


def _stretch(sd: SpectralData, factor: float) -> SpectralData:
    frames = sd.frames
    new_frames = max(1, round(frames * factor))

    src = np.linspace(0, 1, frames)
    dst = np.linspace(0, 1, new_frames)

    channels, _, bins = sd.stft.shape
    new_stft = np.zeros((channels, new_frames, bins), dtype=np.complex64)

    for ch in range(channels):
        real = sd.stft[ch].real  # (frames, bins)
        imag = sd.stft[ch].imag
        interp_r = interp1d(src, real, axis=0, bounds_error=False,
                            fill_value=(real[0], real[-1]))
        interp_i = interp1d(src, imag, axis=0, bounds_error=False,
                            fill_value=(imag[0], imag[-1]))
        new_stft[ch] = (interp_r(dst) + 1j * interp_i(dst)).astype(np.complex64)

    new_original_frames = round(sd.original_frames * factor)

    return SpectralData(
        stft=new_stft,
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=new_original_frames,
    )


@click.command()
@click.argument('factor', type=float)
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
def main(factor, input, output):
    """Time-stretch a .spx file without pitch change.

    Resamples the STFT frame axis by FACTOR. Values >1 slow the audio
    down; values <1 speed it up.

    \b
    FACTOR  Stretch factor (2.0 = half speed, 0.5 = double speed).
    INPUT   .spx file or '-' for stdin pipe.
    OUTPUT  .spx file or '-' for stdout pipe.
            Default: INPUT with '_s{factor}x' suffix.

    \b
    Examples:
      sp-stretch 2.0 song.spx slow.spx
      sp-stretch 0.5 song.spx fast.spx
      cat song.spx | sp-stretch 1.5 - out.spx
    """
    if factor <= 0:
        raise click.UsageError("FACTOR must be positive")

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

    result = _stretch(sd, factor)

    if output is None:
        output = str(src_dir / f"{stem}_s{factor:g}x.spx")

    if output == '-':
        write_pipe(result)
    else:
        save(result, output)
        click.echo(
            f"Stretched ×{factor:g} ({sd.duration:.3f}s → {result.duration:.3f}s) → {output}",
            err=True,
        )
