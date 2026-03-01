"""sp-concat: Concatenate multiple audio or spectral files end-to-end."""

import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import AudioData, load, save_output, is_pipe
from soundplay.core import spectral as sp


def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    return 'spectral' if ext == '.spx' else 'audio'


def _concat_audio(parts: list[AudioData]) -> AudioData:
    # Resample all to the first file's sample rate
    sr = parts[0].sample_rate
    for p in parts:
        if p.sample_rate != sr:
            raise click.UsageError(
                f"Sample rate mismatch: {p.sample_rate} vs {sr}. "
                "All inputs must have the same sample rate."
            )

    # Match channel counts: upmix mono to stereo if needed
    max_ch = max(p.channels for p in parts)
    aligned = []
    for p in parts:
        if p.channels < max_ch:
            aligned.append(p.as_stereo() if max_ch == 2 else p)
        else:
            aligned.append(p)

    return AudioData(np.concatenate([p.samples for p in aligned], axis=0), sr)


def _concat_spectral(parts: list[sp.SpectralData]) -> sp.SpectralData:
    ref = parts[0]
    for i, p in enumerate(parts[1:], 1):
        if p.sample_rate != ref.sample_rate:
            raise click.UsageError(f"Sample rate mismatch in file {i+1}")
        if p.n_fft != ref.n_fft:
            raise click.UsageError(f"FFT size mismatch in file {i+1}")
        if p.hop_length != ref.hop_length:
            raise click.UsageError(f"Hop length mismatch in file {i+1}")
        if p.channels != ref.channels:
            raise click.UsageError(f"Channel count mismatch in file {i+1}")

    stft = np.concatenate([p.stft for p in parts], axis=1)  # concat along frames
    orig = sum(p.original_frames for p in parts)
    return sp.SpectralData(
        stft=stft,
        sample_rate=ref.sample_rate,
        n_fft=ref.n_fft,
        hop_length=ref.hop_length,
        window=ref.window,
        original_frames=orig,
    )


@click.command()
@click.argument('inputs', nargs=-1, required=True)
@click.option('--output', '-o', default=None,
              help='Output file. Defaults to first input with _concat suffix.')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(inputs, output, fmt):
    """Concatenate multiple audio or spectral files end-to-end.

    All inputs must be the same format (all audio or all spectral)
    and share the same sample rate.

    \b
    Examples:
      sp-concat intro.wav verse.wav chorus.wav -o song.wav
      sp-concat part1.spx part2.spx -o combined.spx
    """
    if len(inputs) < 2:
        raise click.UsageError("Need at least 2 input files")

    fmt_in = _detect_format(inputs[0])

    # ------------------------------------------------------------------ load
    if fmt_in == 'spectral':
        parts = [sp.load(f) for f in inputs]
    else:
        parts = [load(f) for f in inputs]

    # ------------------------------------------------------------------ concat
    if fmt_in == 'spectral':
        result = _concat_spectral(parts)
    else:
        result = _concat_audio(parts)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(inputs[0])
        output = str(p.with_stem(p.stem + '_concat'))

    fmt_out = _detect_format(output)

    # ------------------------------------------------------------------ save
    if fmt_out == 'spectral':
        sp.save(result, output)
    else:
        if fmt_in == 'spectral':
            samples = sp.compute_istft(result)
            result = AudioData(samples, result.sample_rate)
        save_output(result, output, format=fmt)

    total = result.duration if hasattr(result, 'duration') else (result.original_frames / result.sample_rate)
    click.echo(f"Concatenated {len(inputs)} files ({total:.3f}s) → {output}", err=True)
