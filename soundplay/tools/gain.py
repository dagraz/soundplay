"""sp-gain: Scale volume by a factor or dB amount."""

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


def _parse_gain(gain_str: str) -> float:
    """Parse a gain string: plain number = linear factor, 'NdB' = decibels."""
    s = gain_str.strip()
    if s.lower().endswith('db'):
        db = float(s[:-2])
        return 10.0 ** (db / 20.0)
    return float(s)


def _gain_audio(audio: AudioData, factor: float) -> AudioData:
    return AudioData(
        np.clip(audio.samples * factor, -1.0, 1.0).astype(np.float32),
        audio.sample_rate,
    )


def _gain_spectral(sd: sp.SpectralData, factor: float) -> sp.SpectralData:
    return sp.SpectralData(
        stft=(sd.stft * factor).astype(np.complex64),
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('gain')
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(gain, input, output, fmt):
    """Scale volume by a linear factor or dB amount.

    GAIN is a linear multiplier (e.g. 0.5 = halve, 2.0 = double)
    or a dB value with suffix (e.g. -6dB, +3dB).

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_gain' suffix.

    \b
    Examples:
      sp-gain 0.5 song.wav quieter.wav
      sp-gain 2.0 song.wav louder.wav
      sp-gain -6dB song.wav song_quiet.wav
      sp-gain +3dB song.spx boosted.spx
      cat song.wav | sp-gain -6dB - out.wav
    """
    factor = _parse_gain(gain)

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

    # ------------------------------------------------------------------ gain
    if fmt_in == 'spectral':
        result = _gain_spectral(sd, factor)
    else:
        result = _gain_audio(audio, factor)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_gain'))

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

    click.echo(f"Gain ×{factor:.4g} ({20*np.log10(factor):+.1f} dB) → {output}", err=True)
