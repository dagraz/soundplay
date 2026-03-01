"""sp-morph: Interpolate between two .spx files for timbral transitions."""

import sys
import numpy as np
import click
from pathlib import Path
from scipy.interpolate import interp1d
from soundplay.core.spectral import SpectralData, load, save, write_pipe
from soundplay.core.audio import is_pipe


def _morph(sd1: SpectralData, sd2: SpectralData,
           blend_start: float, blend_end: float) -> SpectralData:
    frames1 = sd1.frames
    frames2 = sd2.frames

    # Resample sd2 to sd1's frame count
    src_frames = np.linspace(0, 1, frames2)
    dst_frames = np.linspace(0, 1, frames1)

    # Interpolate sd2's real/imag along frames axis per channel
    channels = sd1.channels
    bins = sd1.bins
    sd2_resampled = np.zeros((channels, frames1, bins), dtype=np.complex64)

    for ch in range(channels):
        real2 = sd2.stft[ch].real  # (frames2, bins)
        imag2 = sd2.stft[ch].imag
        interp_r = interp1d(src_frames, real2, axis=0, bounds_error=False,
                            fill_value=(real2[0], real2[-1]))
        interp_i = interp1d(src_frames, imag2, axis=0, bounds_error=False,
                            fill_value=(imag2[0], imag2[-1]))
        sd2_resampled[ch] = (interp_r(dst_frames) + 1j * interp_i(dst_frames)).astype(np.complex64)

    # Build linear alpha envelope over frames
    alpha = np.linspace(blend_start, blend_end, frames1)  # (frames,)
    alpha = alpha[np.newaxis, :, np.newaxis]  # (1, frames, 1) for broadcasting

    new_stft = ((1.0 - alpha) * sd1.stft + alpha * sd2_resampled).astype(np.complex64)

    return SpectralData(
        stft=new_stft,
        sample_rate=sd1.sample_rate,
        n_fft=sd1.n_fft,
        hop_length=sd1.hop_length,
        window=sd1.window,
        original_frames=sd1.original_frames,
    )


@click.command()
@click.argument('input1')
@click.argument('input2')
@click.option('--output', default=None, help='.spx output file or - for stdout.')
@click.option('--blend-start', default=0.0, show_default=True,
              help='Blend weight of INPUT2 at the start (0=INPUT1, 1=INPUT2).')
@click.option('--blend-end', default=1.0, show_default=True,
              help='Blend weight of INPUT2 at the end.')
def main(input1, input2, output, blend_start, blend_end):
    """Interpolate between two .spx files for timbral transitions.

    Resamples INPUT2 to INPUT1's frame count, then blends with a linear
    alpha envelope from blend-start to blend-end.

    Both inputs must share n_fft, hop_length, channels, and sample_rate.

    \b
    Examples:
      sp-morph a.spx b.spx --output morphed.spx
      sp-morph a.spx b.spx --blend-start 0.2 --blend-end 0.8 --output partial.spx
      sp-morph a.spx b.spx --output - | sp-resynth - out.wav
    """
    sd1 = load(input1)
    sd2 = load(input2)

    if sd1.n_fft != sd2.n_fft:
        raise click.UsageError("Inputs must have the same n_fft")
    if sd1.hop_length != sd2.hop_length:
        raise click.UsageError("Inputs must have the same hop_length")
    if sd1.channels != sd2.channels:
        raise click.UsageError("Inputs must have the same channel count")
    if sd1.sample_rate != sd2.sample_rate:
        raise click.UsageError("Inputs must have the same sample rate")

    result = _morph(sd1, sd2, blend_start, blend_end)

    if output is None:
        stem = Path(input1).stem
        src_dir = Path(input1).parent
        output = str(src_dir / f"{stem}_morph.spx")

    if output == '-':
        write_pipe(result)
    else:
        save(result, output)
        click.echo(
            f"Morphed ({blend_start:.2f}→{blend_end:.2f}) → {output}",
            err=True,
        )
