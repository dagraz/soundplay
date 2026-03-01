"""sp-filter: Apply lowpass, highpass, bandpass, or notch filter."""

import sys
import io
import numpy as np
import click
from pathlib import Path
from scipy.signal import butter, sosfilt
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


def _apply_filter(samples: np.ndarray, sr: int, ftype: str,
                  freq: float, freq_hi: float | None, order: int) -> np.ndarray:
    nyq = sr / 2.0
    if ftype in ('bandpass', 'bandstop'):
        if freq_hi is None:
            raise click.UsageError(f"--freq-hi is required for {ftype} filter")
        sos = butter(order, [freq / nyq, freq_hi / nyq], btype=ftype, output='sos')
    else:
        btype = 'low' if ftype == 'lowpass' else 'high'
        sos = butter(order, freq / nyq, btype=btype, output='sos')

    # Filter each channel independently
    out = np.empty_like(samples)
    for ch in range(samples.shape[1]):
        out[:, ch] = sosfilt(sos, samples[:, ch])
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _filter_spectral(sd: sp.SpectralData, ftype: str,
                     freq: float, freq_hi: float | None) -> sp.SpectralData:
    """Apply filter in the frequency domain by zeroing out bins."""
    freqs = np.fft.rfftfreq(sd.n_fft, 1.0 / sd.sample_rate)

    if ftype == 'lowpass':
        mask = freqs <= freq
    elif ftype == 'highpass':
        mask = freqs >= freq
    elif ftype == 'bandpass':
        if freq_hi is None:
            raise click.UsageError("--freq-hi is required for bandpass filter")
        mask = (freqs >= freq) & (freqs <= freq_hi)
    elif ftype == 'bandstop':
        if freq_hi is None:
            raise click.UsageError("--freq-hi is required for bandstop (notch) filter")
        mask = (freqs < freq) | (freqs > freq_hi)

    stft = sd.stft.copy()
    stft[:, :, ~mask] = 0.0
    return sp.SpectralData(
        stft=stft.astype(np.complex64),
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('input', default=None, required=False)
@click.argument('output', default=None, required=False)
@click.option('--type', 'ftype', required=True,
              type=click.Choice(['lowpass', 'highpass', 'bandpass', 'bandstop']),
              help='Filter type.')
@click.option('--freq', '-f', required=True, type=float,
              help='Cutoff frequency in Hz (or low edge for band filters).')
@click.option('--freq-hi', default=None, type=float,
              help='High edge frequency in Hz (required for bandpass/bandstop).')
@click.option('--order', '-n', default=4, show_default=True,
              help='Filter order (audio mode only; higher = steeper rolloff).')
@click.option('--format', 'fmt', default=None,
              help='Force output format for audio (wav, flac).')
def main(input, output, ftype, freq, freq_hi, order, fmt):
    """Apply a frequency filter to audio or spectral data.

    For audio input, uses a Butterworth IIR filter.
    For spectral input, applies a brick-wall filter in the frequency domain.

    \b
    INPUT   Source file or '-' for stdin pipe.
    OUTPUT  Destination file or '-' for stdout pipe.
            Defaults to INPUT with '_filt' suffix.

    \b
    Examples:
      sp-filter --type lowpass --freq 2000 song.wav muffled.wav
      sp-filter --type highpass --freq 200 vocals.wav no_rumble.wav
      sp-filter --type bandpass --freq 300 --freq-hi 3000 speech.wav telephone.wav
      sp-filter --type bandstop --freq 50 --freq-hi 60 recording.wav dehum.wav
      sp-filter --type lowpass --freq 1000 song.spx filtered.spx
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

    # ------------------------------------------------------------------ filter
    if fmt_in == 'spectral':
        result = _filter_spectral(sd, ftype, freq, freq_hi)
    else:
        filtered = _apply_filter(audio.samples, audio.sample_rate, ftype, freq, freq_hi, order)
        result = AudioData(filtered, audio.sample_rate)

    # ------------------------------------------------------------------ output path
    if output is None:
        p = Path(input)
        output = str(p.with_stem(p.stem + '_filt'))

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

    desc = f"{ftype} {freq:.0f}Hz"
    if freq_hi is not None:
        desc += f"–{freq_hi:.0f}Hz"
    click.echo(f"Filtered ({desc}) → {output}", err=True)
