"""sp-spectralize: Convert an audio file to spectral (.spx) format."""

import sys
import click
from soundplay.core.audio import load_input, is_pipe
from soundplay.core.spectral import compute_stft, save, write_pipe


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--n-fft', default=2048, show_default=True,
              help='FFT window size (samples). Larger = more frequency resolution, less time resolution.')
@click.option('--hop-length', default=512, show_default=True,
              help='Hop length (samples). Smaller = more time frames, more overlap.')
@click.option('--window', default='hann', show_default=True,
              help='Window function (hann, hamming, blackman, etc.)')
def main(input, output, n_fft, hop_length, window):
    """Convert an audio file to spectral (.spx) format.

    \b
    INPUT   Audio file (wav, flac, mp3) or '-' for stdin pipe.
    OUTPUT  .spx file or '-' for stdout pipe.

    \b
    Examples:
      sp-spectralize song.mp3 song.spx
      sp-spectralize song.wav - | sp-resynth - out.wav
      cat song.wav | sp-spectralize | sp-resynth > out.wav
    """
    audio = load_input(input)

    sd = compute_stft(audio.samples, audio.sample_rate,
                      n_fft=n_fft, hop_length=hop_length, window=window)

    if output == '-' or (output is None and is_pipe(sys.stdout)):
        write_pipe(sd)
    elif output is None:
        raise click.UsageError("No output: provide an OUTPUT file or pipe via stdout")
    else:
        save(sd, output)
        click.echo(
            f"Wrote {sd.channels}ch spectral: {sd.frames} frames × {sd.bins} bins "
            f"({sd.duration:.3f}s) → {output}",
            err=True
        )
