"""sp-fade: Apply fade-in and/or fade-out to audio or spectral data."""

import sys
import io
import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import AudioData, load, save_output, read_pipe, is_pipe
from soundplay.core import spectral as sp
from soundplay.core.timeutil import TIME, resolve as resolve_time

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


def _make_envelope(total_frames: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Build a gain envelope with linear fade-in and fade-out."""
    env = np.ones(total_frames, dtype=np.float32)
    if fade_in > 0:
        env[:fade_in] = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        env[-fade_out:] = np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return env


def _fade_audio(audio: AudioData, fade_in_s: float, fade_out_s: float) -> AudioData:
    fi = int(round(fade_in_s * audio.sample_rate))
    fo = int(round(fade_out_s * audio.sample_rate))
    env = _make_envelope(audio.frames, fi, fo)
    # Broadcast across channels: env is (frames,), samples is (frames, channels)
    return AudioData((audio.samples * env[:, np.newaxis]).astype(np.float32), audio.sample_rate)


def _fade_spectral(sd: sp.SpectralData, fade_in_s: float, fade_out_s: float) -> sp.SpectralData:
    fi = int(round(fade_in_s * sd.sample_rate / sd.hop_length))
    fo = int(round(fade_out_s * sd.sample_rate / sd.hop_length))
    env = _make_envelope(sd.frames, fi, fo)
    # STFT shape is (channels, frames, bins) — broadcast env over frames axis
    scaled = sd.stft * env[np.newaxis, :, np.newaxis]
    return sp.SpectralData(
        stft=scaled.astype(np.complex64),
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--fade-in', 'fade_in', default=None, type=TIME,
              help='Fade-in duration: seconds or percent (e.g. 2 or 10%).')
@click.option('--fade-out', 'fade_out', default=None, type=TIME,
              help='Fade-out duration: seconds or percent (e.g. 2 or 10%).')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(input, output, fade_in, fade_out, fmt):
    """Apply fade-in and/or fade-out.

    At least one of --fade-in or --fade-out must be specified.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_fade' suffix.

    \b
    Examples:
      sp-fade --fade-in 2 --fade-out 3 song.wav faded.wav
      sp-fade --fade-in 10% song.wav intro.wav
      sp-fade --fade-out 5 song.spx fadeout.spx
    """
    if fade_in is None and fade_out is None:
        raise click.UsageError("At least one of --fade-in or --fade-out is required")

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

    # ------------------------------------------------------------------ resolve times
    fi_s = resolve_time(fade_in, total) or 0.0
    fo_s = resolve_time(fade_out, total) or 0.0

    # ------------------------------------------------------------------ fade
    if fmt_in == 'spectral':
        result = _fade_spectral(sd, fi_s, fo_s)
    else:
        result = _fade_audio(audio, fi_s, fo_s)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_fade'))

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

    parts = []
    if fi_s > 0:
        parts.append(f"in {fi_s:.3f}s")
    if fo_s > 0:
        parts.append(f"out {fo_s:.3f}s")
    click.echo(f"Fade {', '.join(parts)} → {output}", err=True)
