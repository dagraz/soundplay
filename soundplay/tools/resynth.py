"""sp-resynth: Convert spectral (.spx) data back to an audio file."""

import sys
import click
from soundplay.core.audio import AudioData, save_output, is_pipe
from soundplay.core.spectral import load, read_pipe, compute_istft


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--format', 'fmt', default=None,
              help='Output format (wav, flac). Inferred from OUTPUT extension if omitted.')
def main(input, output, fmt):
    """Convert spectral (.spx) data back to an audio file.

    \b
    INPUT   .spx file or '-' for stdin pipe.
    OUTPUT  Audio file (wav, flac) or '-' for stdout pipe.

    \b
    Examples:
      sp-resynth song.spx reconstructed.wav
      sp-spectralize song.mp3 - | sp-resynth - out.wav
    """
    if input == '-' or (input is None and is_pipe(sys.stdin)):
        sd = read_pipe()
    elif input is None:
        raise click.UsageError("No input: provide an INPUT file or pipe via stdin")
    else:
        sd = load(input)

    samples = compute_istft(sd)
    audio = AudioData(samples, sd.sample_rate)

    save_output(audio, output, format=fmt)

    if output and output != '-':
        click.echo(
            f"Wrote {audio.channels}ch audio: {audio.frames} frames "
            f"({audio.duration:.3f}s) @ {audio.sample_rate}Hz → {output}",
            err=True
        )
