"""sp-info: Display information about an audio or spectral file."""

import sys
import io
import click
from pathlib import Path

_SPECTRAL_MAGIC = b'SPXF'


def _detect_format(path):
    if path is None or path == '-':
        return None
    return 'spectral' if Path(path).suffix.lower() == '.spx' else 'audio'


@click.command()
@click.argument('input', default=None, required=False)
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def main(input, as_json):
    """Display information about an audio or spectral (.spx) file.

    INPUT may be a file path or '-' to read from stdin pipe.
    """
    fmt = _detect_format(input)

    if fmt is None:
        header = sys.stdin.buffer.read(4)
        buffered = io.BytesIO(header + sys.stdin.buffer.read())
        fmt = 'spectral' if header == _SPECTRAL_MAGIC else 'audio'
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
        from soundplay.core.audio import load_input, read_pipe
        audio = read_pipe(buffered) if buffered else load_input(input)
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
