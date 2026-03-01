"""sp-gate: Zero spectral bins below a dB threshold."""

import sys
import numpy as np
import click
from pathlib import Path
from soundplay.core.spectral import SpectralData, load, read_pipe, save, write_pipe
from soundplay.core.audio import is_pipe


def _gate(sd: SpectralData, threshold_db: float) -> SpectralData:
    linear = 10.0 ** (threshold_db / 20.0)
    new_stft = sd.stft.copy()
    new_stft[np.abs(new_stft) < linear] = 0.0
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
@click.option('--threshold', default=-40.0, show_default=True,
              help='Gate threshold in dBFS. Bins below this level are zeroed.')
def main(input, output, threshold):
    """Zero spectral bins below a dB threshold.

    \b
    INPUT   .spx file or '-' for stdin pipe.
    OUTPUT  .spx file or '-' for stdout pipe.
            Default: INPUT with '_gate' suffix.

    \b
    Examples:
      sp-gate song.spx gated.spx
      sp-gate --threshold -30 noisy.spx clean.spx
      cat song.spx | sp-gate --threshold -50 - out.spx
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

    result = _gate(sd, threshold)

    if output is None:
        output = str(src_dir / f"{stem}_gate.spx")

    if output == '-':
        write_pipe(result)
    else:
        save(result, output)
        click.echo(f"Gated at {threshold} dB → {output}", err=True)
