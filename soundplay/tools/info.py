"""sp-info: Display information about an audio or spectral file."""

import sys
import io
import click
from pathlib import Path

_MAGIC_SPECTRAL = b'SPXF'
_MAGIC_SPAW     = b'SPAW'
_MAGIC_WAV      = b'RIFF'
_MAGIC_FLAC     = b'fLaC'
_MAGIC_OGG      = b'OggS'


def _detect_format(path):
    """Return 'spectral' or 'audio' from file extension, or None if unknown."""
    if path is None or path == '-':
        return None
    return 'spectral' if Path(path).suffix.lower() == '.spx' else 'audio'


def _load_audio_from_buffer(buffered):
    """Read AudioData from a BytesIO buffer containing any supported audio format."""
    import soundfile as sf
    import numpy as np
    from soundplay.core.audio import AudioData, read_pipe
    # Peek at magic to decide whether it's our internal SPAW stream or a raw file
    magic = buffered.read(4)
    buffered.seek(0)
    if magic == _MAGIC_SPAW:
        return read_pipe(buffered)
    # Raw audio file (WAV, FLAC, OGG…) — soundfile reads from file-like objects
    data, sr = sf.read(buffered, dtype='float32', always_2d=True)
    return AudioData(data, sr)


@click.command()
@click.argument('input', default=None, required=False)
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def main(input, as_json):
    """Display information about an audio or spectral (.spx) file.

    INPUT may be a file path, '-' for stdin, or omitted when stdin is a pipe.
    Accepts raw WAV/FLAC files piped directly as well as SPAW/SPXF streams.
    """
    fmt = _detect_format(input)

    if fmt is None:
        # Sniff magic bytes from stdin to determine format
        header = sys.stdin.buffer.read(4)
        buffered = io.BytesIO(header + sys.stdin.buffer.read())
        if header == _MAGIC_SPECTRAL:
            fmt = 'spectral'
        else:
            fmt = 'audio'
    else:
        buffered = None

    if fmt == 'spectral':
        from soundplay.core.spectral import load, _read_stream
        sd = _read_stream(buffered) if buffered else load(input)
        if as_json:
            import json
            print(json.dumps({
                'sample_rate': sd.sample_rate,
                'channels': sd.channels,
                'duration': sd.duration,
                'frames': sd.frames,
                'bins': sd.bins,
                'n_fft': sd.n_fft,
                'hop_length': sd.hop_length,
                'window': sd.window,
                'original_frames': sd.original_frames,
            }))
        else:
            print(f"Format      : spectral (.spx)")
            print(f"Sample rate : {sd.sample_rate} Hz")
            print(f"Channels    : {sd.channels}")
            print(f"Duration    : {sd.duration:.3f} s")
            print(f"STFT frames : {sd.frames}")
            print(f"Freq bins   : {sd.bins}")
            print(f"FFT size    : {sd.n_fft}")
            print(f"Hop length  : {sd.hop_length}")
            print(f"Window      : {sd.window}")
    else:
        from soundplay.core.audio import load_input
        audio = _load_audio_from_buffer(buffered) if buffered else load_input(input)
        if as_json:
            import json
            print(json.dumps({
                'sample_rate': audio.sample_rate,
                'channels': audio.channels,
                'frames': audio.frames,
                'duration': audio.duration,
            }))
        else:
            print(f"Format      : audio")
            print(f"Sample rate : {audio.sample_rate} Hz")
            print(f"Channels    : {audio.channels}")
            print(f"Frames      : {audio.frames}")
            print(f"Duration    : {audio.duration:.3f} s")
