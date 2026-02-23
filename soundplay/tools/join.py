"""sp-join: Combine multiple .spx files into one by spectral superposition."""

import sys
import numpy as np
import click
from pathlib import Path
from soundplay.core.spectral import load, read_pipe, save, write_pipe, SpectralData
from soundplay.core.audio import is_pipe


def _compatible(a: SpectralData, b: SpectralData, idx: int) -> None:
    """Raise if two SpectralData objects have incompatible STFT parameters."""
    mismatches = []
    if a.sample_rate != b.sample_rate:
        mismatches.append(f"sample_rate {a.sample_rate} vs {b.sample_rate}")
    if a.n_fft != b.n_fft:
        mismatches.append(f"n_fft {a.n_fft} vs {b.n_fft}")
    if a.hop_length != b.hop_length:
        mismatches.append(f"hop_length {a.hop_length} vs {b.hop_length}")
    if a.window != b.window:
        mismatches.append(f"window '{a.window}' vs '{b.window}'")
    if a.channels != b.channels:
        mismatches.append(f"channels {a.channels} vs {b.channels}")
    if mismatches:
        raise click.UsageError(
            f"Input {idx} is incompatible with input 0: {'; '.join(mismatches)}"
        )


def _join(parts: list[SpectralData], strict: bool) -> SpectralData:
    """
    Sum STFT arrays from all parts.
    If frame counts differ: error in strict mode, else pad shorter files with zeros.
    """
    max_frames = max(p.frames for p in parts)
    min_frames = min(p.frames for p in parts)

    if strict and max_frames != min_frames:
        raise click.UsageError(
            f"Frame count mismatch ({min_frames}–{max_frames} frames). "
            "Use --no-strict to pad shorter files with silence."
        )

    if max_frames != min_frames:
        click.echo(
            f"Warning: frame counts differ ({min_frames}–{max_frames}); "
            "padding shorter files with silence.",
            err=True,
        )

    ref = parts[0]
    ch, _, bins = ref.stft.shape
    acc = np.zeros((ch, max_frames, bins), dtype=np.complex64)

    for p in parts:
        acc[:, :p.frames, :] += p.stft

    max_orig = max(p.original_frames for p in parts)
    return SpectralData(
        stft=acc,
        sample_rate=ref.sample_rate,
        n_fft=ref.n_fft,
        hop_length=ref.hop_length,
        window=ref.window,
        original_frames=max_orig,
    )


@click.command()
@click.argument('inputs', nargs=-1, required=True)
@click.option('-o', '--output', required=True,
              help='Output .spx file or \'-\' for stdout pipe.')
@click.option('--strict', is_flag=True, default=True,
              help='Error if input files have different lengths (default).')
@click.option('--no-strict', 'strict', flag_value=False,
              help='Pad shorter files with silence instead of erroring.')
def main(inputs, output, strict):
    """Combine multiple .spx files into one by spectral superposition.

    The dual of sp-decompose: sums the STFT arrays of all inputs so that
    reconstructing the output gives the mix of all component sounds.

    All inputs must share the same sample rate, FFT size, hop length,
    window function, and channel count.

    \b
    Examples:
      sp-join part_440.0hz.spx part_554.0hz.spx part_659.0hz.spx -o rejoined.spx
      sp-join parts/*.spx -o rejoined.spx
      sp-join a.spx b.spx c.spx -o - | sp-resynth - mix.wav
    """
    parts = []
    for i, path in enumerate(inputs):
        sd = load(path)
        if i > 0:
            _compatible(parts[0], sd, i)
        parts.append(sd)
        click.echo(f"  Loaded {path}  ({sd.frames} frames, {sd.duration:.3f}s)", err=True)

    click.echo(f"Joining {len(parts)} file(s)...", err=True)
    result = _join(parts, strict=strict)

    if output == '-' or (output is None and is_pipe(sys.stdout)):
        write_pipe(result)
    else:
        save(result, output)

    click.echo(
        f"Wrote {result.channels}ch spectral: {result.frames} frames "
        f"({result.duration:.3f}s) → {output}",
        err=True,
    )
