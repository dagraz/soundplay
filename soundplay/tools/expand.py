"""sp-expand: Pad an audio or spectral file with silence."""

import sys
import io
import numpy as np
import click
from pathlib import Path
from soundplay.core.audio import AudioData, load, save_output, read_pipe, write_pipe, is_pipe
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


def _pad_audio(audio: AudioData, pad_start_s: float, pad_end_s: float) -> AudioData:
    sr = audio.sample_rate
    ch = audio.channels
    start_frames = int(round(pad_start_s * sr))
    end_frames   = int(round(pad_end_s   * sr))
    silence_start = np.zeros((start_frames, ch), dtype=np.float32)
    silence_end   = np.zeros((end_frames,   ch), dtype=np.float32)
    padded = np.concatenate([silence_start, audio.samples, silence_end], axis=0)
    return AudioData(padded, sr)


def _pad_spectral(sd: sp.SpectralData, pad_start_s: float, pad_end_s: float) -> sp.SpectralData:
    frames_start = int(round(pad_start_s * sd.sample_rate / sd.hop_length))
    frames_end   = int(round(pad_end_s   * sd.sample_rate / sd.hop_length))
    ch, _, bins = sd.stft.shape
    silence_start = np.zeros((ch, frames_start, bins), dtype=np.complex64)
    silence_end   = np.zeros((ch, frames_end,   bins), dtype=np.complex64)
    padded_stft = np.concatenate([silence_start, sd.stft, silence_end], axis=1)
    orig_start = int(round(pad_start_s * sd.sample_rate))
    orig_end   = int(round(pad_end_s   * sd.sample_rate))
    return sp.SpectralData(
        stft=padded_stft,
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames + orig_start + orig_end,
    )


@click.command()
@click.argument('input',  default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--pad-start', default='0', type=TIME, show_default=True,
              help='Silence to add before the audio: seconds or percent of duration (e.g. 2 or 10%).')
@click.option('--pad-end',   default='0', type=TIME, show_default=True,
              help='Silence to add after the audio: seconds or percent of duration.')
@click.option('--pad',       default=None, type=TIME,
              help='Add equal silence to both start and end. Overridden by --pad-start/--pad-end.')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(input, output, pad_start, pad_end, pad, fmt):
    """Pad an audio or spectral file with silence.

    Works with .spx, .wav, .flac, and .mp3 files (auto-detected).
    Supports stdin/stdout pipes for both audio (SPAW) and spectral (SPXF) streams.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_pad' suffix.

    \b
    Examples:
      sp-expand --pad-end 3 song.wav padded.wav
      sp-expand --pad-start 10% --pad-end 10% song.spx padded.spx
      sp-expand --pad 2 song.wav                  # 2s silence on both sides
      sp-expand --pad 5% song.spx                 # 5% silence on both sides
      cat song.spx | sp-expand --pad-end 4 - out.spx
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

    # ------------------------------------------------------------------ resolve pad amounts
    if pad is not None:
        pad_s = resolve_time(pad, total)
        start_s = pad_s
        end_s   = pad_s
    else:
        start_s = resolve_time(pad_start, total)
        end_s   = resolve_time(pad_end,   total)

    if start_s < 0 or end_s < 0:
        raise click.UsageError("Pad amounts must be non-negative")

    # ------------------------------------------------------------------ pad
    if fmt_in == 'spectral':
        result = _pad_spectral(sd, start_s, end_s)
    else:
        result = _pad_audio(audio, start_s, end_s)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_pad'))

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
        f"Padded +{start_s:.3f}s start, +{end_s:.3f}s end "
        f"({total:.3f}s → {new_dur:.3f}s) → {output}",
        err=True,
    )
