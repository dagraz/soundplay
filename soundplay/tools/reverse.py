"""sp-reverse: Reverse audio or spectral data along the time axis."""

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


def _reverse_audio(audio: AudioData) -> AudioData:
    return AudioData(audio.samples[::-1].copy(), audio.sample_rate)


def _reverse_spectral(sd: sp.SpectralData) -> sp.SpectralData:
    # Reverse along the time (frames) axis: stft shape is (channels, frames, bins)
    return sp.SpectralData(
        stft=sd.stft[:, ::-1, :].copy(),
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(input, output, fmt):
    """Reverse audio or spectral data along the time axis.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_rev' suffix.

    \b
    Examples:
      sp-reverse song.wav reversed.wav
      sp-reverse song.spx reversed.spx
      cat song.wav | sp-reverse - out.wav
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
        total = sd.duration
    else:
        audio = read_pipe(buffered) if buffered else load(input)
        total = audio.duration

    # ------------------------------------------------------------------ reverse
    if fmt_in == 'spectral':
        result = _reverse_spectral(sd)
    else:
        result = _reverse_audio(audio)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_rev'))

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

    click.echo(f"Reversed ({total:.3f}s) → {output}", err=True)
