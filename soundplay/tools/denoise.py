"""sp-denoise: Spectral subtraction noise reduction."""

import sys
import numpy as np
import click
from pathlib import Path
from soundplay.core.spectral import SpectralData, load, read_pipe, save, write_pipe
from soundplay.core.audio import is_pipe
from soundplay.core.timeutil import TIME, resolve as resolve_time


def _denoise(sd: SpectralData, noise_start_s: float, noise_end_s: float,
             oversubtract: float) -> SpectralData:
    sr = sd.sample_rate
    hop = sd.hop_length
    # Convert time to frame indices
    frame_start = int(round(noise_start_s * sr / hop))
    frame_end = int(round(noise_end_s * sr / hop))
    frame_start = max(0, min(frame_start, sd.frames - 1))
    frame_end = max(frame_start + 1, min(frame_end, sd.frames))

    # Noise profile: mean magnitude over noise region, averaged across channels
    noise_region = np.abs(sd.stft[:, frame_start:frame_end, :])  # (ch, frames, bins)
    noise_profile = noise_region.mean(axis=(0, 1))  # (bins,)

    # Spectral subtraction per channel
    magnitudes = np.abs(sd.stft)  # (ch, frames, bins)
    phases = np.angle(sd.stft)

    new_mag = np.maximum(magnitudes - oversubtract * noise_profile[np.newaxis, np.newaxis, :], 0.0)
    new_stft = (new_mag * np.exp(1j * phases)).astype(np.complex64)

    return SpectralData(
        stft=new_stft,
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--noise-start', default='0', type=TIME, show_default=True,
              help='Start of noise reference region (seconds or %).')
@click.option('--noise-end', default='0.5', type=TIME, show_default=True,
              help='End of noise reference region (seconds or %).')
@click.option('--oversubtract', default=1.0, show_default=True,
              help='Oversubtraction factor (>1 = more aggressive).')
def main(input, output, noise_start, noise_end, oversubtract):
    """Spectral subtraction noise reduction.

    Estimates a noise profile from a quiet region, then subtracts it
    from the entire file's magnitude spectrum.

    \b
    INPUT   .spx file or '-' for stdin pipe.
    OUTPUT  .spx file or '-' for stdout pipe.
            Default: INPUT with '_denoise' suffix.

    \b
    Examples:
      sp-denoise recording.spx clean.spx
      sp-denoise --noise-start 0 --noise-end 1.0 recording.spx clean.spx
      sp-denoise --oversubtract 1.5 noisy.spx cleaner.spx
    """
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

    total = sd.duration
    ns = resolve_time(noise_start, total)
    ne = resolve_time(noise_end, total)

    if ns >= ne:
        raise click.UsageError("--noise-start must be less than --noise-end")

    result = _denoise(sd, ns, ne, oversubtract)

    if output is None:
        output = str(src_dir / f"{stem}_denoise.spx")

    if output == '-':
        write_pipe(result)
    else:
        save(result, output)
        click.echo(
            f"Denoised (noise region {ns:.3f}s–{ne:.3f}s, oversubtract={oversubtract}) → {output}",
            err=True,
        )
