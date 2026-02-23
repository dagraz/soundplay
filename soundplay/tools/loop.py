"""sp-loop: Repeat an audio or spectral file N times."""

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


def _loop_audio(audio: AudioData, times: int) -> AudioData:
    return AudioData(np.tile(audio.samples, (times, 1)), audio.sample_rate)


def _loop_spectral(sd: sp.SpectralData, times: int) -> sp.SpectralData:
    return sp.SpectralData(
        stft=np.tile(sd.stft, (1, times, 1)),
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames * times,
    )


@click.command()
@click.argument('input',  default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--times', '-n', default=2, show_default=True,
              help='Number of times to repeat the file.')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(input, output, times, fmt):
    """Repeat an audio or spectral file N times.

    Works with .spx, .wav, .flac, and .mp3 files (auto-detected).
    Supports stdin/stdout pipes for both audio (SPAW) and spectral (SPXF) streams.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_loopN' suffix.

    \b
    Examples:
      sp-loop --times 4 beat.wav beat_x4.wav
      sp-loop -n 3 phrase.spx phrase_x3.spx
      cat beat.spx | sp-loop -n 8 - out.spx
    """
    if times < 1:
        raise click.BadParameter("Must be at least 1", param_hint='--times')

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

    # ------------------------------------------------------------------ loop
    if fmt_in == 'spectral':
        result = _loop_spectral(sd, times)
    else:
        result = _loop_audio(audio, times)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(f"{p.stem}_loop{times}"))

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

    new_dur = result.duration if hasattr(result, 'duration') else (result.original_frames / result.sample_rate)
    click.echo(
        f"Looped {times}× ({total:.3f}s → {new_dur:.3f}s) → {output}",
        err=True,
    )
