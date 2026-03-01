"""sp-normalize: Normalize audio to a target peak or RMS level."""

import sys
import io
import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import AudioData, load, save_output, read_pipe, is_pipe
from soundplay.core import spectral as sp

_SPECTRAL_MAGIC = b'SPXF'


def _sniff_stream(stream) -> tuple[str, object]:
    header = stream.read(4)
    buffered = io.BytesIO(header + stream.read())
    fmt = 'spectral' if header == _SPECTRAL_MAGIC else 'audio'
    return fmt, buffered


def _detect_format(path: str | None) -> str | None:
    if path is None or path == '-':
        return None
    ext = Path(path).suffix.lower()
    return 'spectral' if ext == '.spx' else 'audio'


def _normalize_audio(audio: AudioData, target_db: float, mode: str) -> AudioData:
    if mode == 'peak':
        current = np.max(np.abs(audio.samples))
    else:  # rms
        current = np.sqrt(np.mean(audio.samples ** 2))

    if current < 1e-10:
        return audio  # silence, nothing to normalize

    target_linear = 10.0 ** (target_db / 20.0)
    factor = target_linear / current
    return AudioData(
        np.clip(audio.samples * factor, -1.0, 1.0).astype(np.float32),
        audio.sample_rate,
    )


def _normalize_spectral(sd: sp.SpectralData, target_db: float, mode: str) -> sp.SpectralData:
    # Resynth to measure level, compute factor, apply to STFT
    samples = sp.compute_istft(sd)
    if mode == 'peak':
        current = np.max(np.abs(samples))
    else:
        current = np.sqrt(np.mean(samples ** 2))

    if current < 1e-10:
        return sd

    target_linear = 10.0 ** (target_db / 20.0)
    factor = target_linear / current
    return sp.SpectralData(
        stft=(sd.stft * factor).astype(np.complex64),
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--target', '-t', default=0.0, show_default=True,
              help='Target level in dBFS (0 = full scale).')
@click.option('--mode', '-m', default='peak', show_default=True,
              type=click.Choice(['peak', 'rms']),
              help='Normalize by peak or RMS level.')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(input, output, target, mode, fmt):
    """Normalize audio to a target peak or RMS level.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_norm' suffix.

    \b
    Examples:
      sp-normalize song.wav normalized.wav
      sp-normalize --target -3 song.wav song_3db.wav
      sp-normalize --mode rms --target -14 song.wav radio.wav
      sp-normalize song.spx normalized.spx
    """
    # ------------------------------------------------------------------ format
    fmt_in = _detect_format(input)
    buffered = None
    if fmt_in is None:
        fmt_in, buffered = _sniff_stream(sys.stdin.buffer)

    fmt_out = _detect_format(output) or fmt_in

    # ------------------------------------------------------------------ load
    if fmt_in == 'spectral':
        sd = sp._read_stream(buffered) if buffered else sp.load(input)
    else:
        audio = read_pipe(buffered) if buffered else load(input)

    # ------------------------------------------------------------------ normalize
    if fmt_in == 'spectral':
        result = _normalize_spectral(sd, target, mode)
    else:
        result = _normalize_audio(audio, target, mode)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_norm'))

    # ------------------------------------------------------------------ save
    if fmt_out == 'spectral':
        if output == '-' or (output is None and is_pipe(sys.stdout)):
            sp.write_pipe(result)
        else:
            sp.save(result, output)
    else:
        if fmt_in == 'spectral':
            samples = sp.compute_istft(result)
            result = AudioData(samples, result.sample_rate)
        save_output(result, output, format=fmt)

    click.echo(f"Normalized ({mode} → {target:+.1f} dBFS) → {output}", err=True)
