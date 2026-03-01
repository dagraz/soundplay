"""sp-mix: Blend multiple audio files with per-file weights."""

import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import AudioData, load, save_output, is_pipe


@click.command()
@click.argument('inputs', nargs=-1, required=True)
@click.option('--output', '-o', default=None,
              help='Output file. Defaults to first input with _mix suffix.')
@click.option('--weights', '-w', default=None,
              help='Comma-separated weights per input (e.g. 0.7,0.3). '
                   'Defaults to equal weight (1/N) for each file.')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(inputs, output, weights, fmt):
    """Blend multiple audio files with per-file weights.

    Files are summed sample-by-sample. Shorter files are zero-padded
    to match the longest. Weights default to equal (1/N).

    \b
    Examples:
      sp-mix vocals.wav backing.wav -o combined.wav
      sp-mix -w 0.8,0.2 vocals.wav reverb.wav -o wet.wav
      sp-mix -w 1.0,0.5,0.5 drums.wav bass.wav keys.wav -o submix.wav
    """
    if len(inputs) < 2:
        raise click.UsageError("Need at least 2 input files")

    # ------------------------------------------------------------------ weights
    if weights is not None:
        w = [float(x) for x in weights.split(',')]
        if len(w) != len(inputs):
            raise click.UsageError(
                f"Got {len(w)} weights for {len(inputs)} inputs"
            )
    else:
        w = [1.0 / len(inputs)] * len(inputs)

    # ------------------------------------------------------------------ load
    parts = [load(f) for f in inputs]

    sr = parts[0].sample_rate
    for p in parts:
        if p.sample_rate != sr:
            raise click.UsageError(
                f"Sample rate mismatch: {p.sample_rate} vs {sr}"
            )

    # Match channels
    max_ch = max(p.channels for p in parts)
    aligned = []
    for p in parts:
        if p.channels < max_ch:
            aligned.append(p.as_stereo() if max_ch == 2 else p)
        else:
            aligned.append(p)

    # ------------------------------------------------------------------ mix
    max_frames = max(p.frames for p in aligned)
    acc = np.zeros((max_frames, max_ch), dtype=np.float32)

    for p, weight in zip(aligned, w):
        acc[:p.frames, :] += p.samples * weight

    result = AudioData(np.clip(acc, -1.0, 1.0).astype(np.float32), sr)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(inputs[0])
        output = str(p.with_stem(p.stem + '_mix'))

    # ------------------------------------------------------------------ save
    save_output(result, output, format=fmt)

    click.echo(
        f"Mixed {len(inputs)} files (weights {','.join(f'{x:.2g}' for x in w)}) → {output}",
        err=True,
    )
