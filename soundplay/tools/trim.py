"""sp-trim: Trim an audio or spectral file to a time range."""

import sys
import io
import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import load_input, save_output, AudioData, is_pipe
from soundplay.core import spectral as sp


_SPECTRAL_MAGIC = b'SPXF'
_AUDIO_MAGIC    = b'SPAW'


def _sniff_stream(stream) -> tuple[str, object]:
    """
    Peek at the first 4 bytes of a binary stream to determine format.
    Returns (format, buffered_stream) where format is 'spectral' or 'audio'
    and buffered_stream has the bytes put back.
    """
    header = stream.read(4)
    buffered = io.BytesIO(header + stream.read())
    fmt = 'spectral' if header == _SPECTRAL_MAGIC else 'audio'
    return fmt, buffered


def _detect_format(path: str | None) -> str | None:
    """Return 'spectral', 'audio', or None (needs sniffing) from file extension."""
    if path is None or path == '-':
        return None
    ext = Path(path).suffix.lower()
    return 'spectral' if ext == '.spx' else 'audio'


def _resolve_times(start: float, end: float | None, duration: float | None,
                   total: float) -> tuple[float, float]:
    """Resolve start/end/duration into a concrete (start_s, end_s) pair."""
    # Negative times = relative to end
    if start < 0:
        start = max(0.0, total + start)
    start = max(0.0, min(start, total))

    if duration is not None:
        end = start + duration

    if end is None:
        end = total
    elif end < 0:
        end = max(0.0, total + end)

    end = max(start, min(end, total))
    return start, end


def _trim_audio(audio: AudioData, start_s: float, end_s: float) -> AudioData:
    s = int(round(start_s * audio.sample_rate))
    e = int(round(end_s   * audio.sample_rate))
    return AudioData(audio.samples[s:e].copy(), audio.sample_rate)


def _trim_spectral(sd: sp.SpectralData, start_s: float, end_s: float) -> sp.SpectralData:
    sf = int(round(start_s * sd.sample_rate / sd.hop_length))
    ef = int(round(end_s   * sd.sample_rate / sd.hop_length))
    sf = max(0, min(sf, sd.frames))
    ef = max(sf, min(ef, sd.frames))
    new_stft = sd.stft[:, sf:ef, :].copy()
    new_orig  = int(round((end_s - start_s) * sd.sample_rate))
    return sp.SpectralData(
        stft=new_stft,
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=new_orig,
    )


@click.command()
@click.argument('input',  default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--start',    default=0.0,  show_default=True,
              help='Start time in seconds. Negative = relative to end.')
@click.option('--end',      default=None, type=float,
              help='End time in seconds. Negative = relative to end. '
                   'Mutually exclusive with --duration.')
@click.option('--duration', default=None, type=float,
              help='Duration in seconds from --start. Mutually exclusive with --end.')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac). '
                   'Inferred from OUTPUT extension when omitted.')
def main(input, output, start, end, duration, fmt):
    """Trim an audio or spectral file to a time range.

    Works with .spx, .wav, .flac, and .mp3 files (auto-detected).
    Supports stdin/stdout pipes for both audio (SPAW) and spectral (SPXF) streams.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_trim' suffix.

    \b
    Examples:
      sp-trim --start 5 --end 30 song.wav trimmed.wav
      sp-trim --start 1.5 --duration 10 song.spx clip.spx
      sp-trim --end -2 song.wav             # drop last 2 seconds (in-place name)
      sp-trim --start -10 song.mp3 outro.wav  # last 10 seconds
      cat song.spx | sp-trim --start 2 --end 8 - clipped.spx
    """
    if end is not None and duration is not None:
        raise click.UsageError("--end and --duration are mutually exclusive")

    # ------------------------------------------------------------------ format
    fmt_in = _detect_format(input)

    if fmt_in is None:
        # Pipe: sniff magic bytes
        raw_fmt, buffered = _sniff_stream(sys.stdin.buffer)
        fmt_in = raw_fmt
    else:
        buffered = None

    # Infer output format from output path if not specified
    fmt_out = _detect_format(output)
    if fmt_out is None:
        fmt_out = fmt_in  # same as input by default

    # ------------------------------------------------------------------ load
    if fmt_in == 'spectral':
        if buffered is not None:
            sd = sp._read_stream(buffered)
        else:
            sd = sp.load(input)
        total = sd.duration
    else:
        if buffered is not None:
            audio = sp  # won't use sp here
            from soundplay.core.audio import read_pipe
            audio = read_pipe(buffered)
        else:
            from soundplay.core.audio import load
            audio = load(input)
        total = audio.duration

    # ------------------------------------------------------------------ trim
    start_s, end_s = _resolve_times(start, end, duration, total)
    trimmed_duration = end_s - start_s

    if fmt_in == 'spectral':
        result = _trim_spectral(sd, start_s, end_s)
    else:
        result = _trim_audio(audio, start_s, end_s)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_trim'))

    # ------------------------------------------------------------------ save
    if fmt_out == 'spectral':
        if output == '-' or (output is None and is_pipe(sys.stdout)):
            sp.write_pipe(result)
        else:
            sp.save(result, output)
    else:
        if fmt_in == 'spectral':
            # Requested audio output from spectral input — resynth first
            samples = sp.compute_istft(result)
            result = AudioData(samples, result.sample_rate)
        save_output(result, output, format=fmt)

    click.echo(
        f"Trimmed {start_s:.3f}s–{end_s:.3f}s ({trimmed_duration:.3f}s) → {output}",
        err=True,
    )
