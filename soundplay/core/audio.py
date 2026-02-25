"""
Core audio I/O: load, save, and pipe audio data.

All tools work with AudioData, a simple container of a numpy float32 array
(samples x channels) plus sample rate. Samples are always in [-1.0, 1.0].

Pipe protocol (raw PCM via stdin/stdout):
  - Header: 16 bytes
      bytes 0-3:  magic b'SPAW'
      bytes 4-7:  sample_rate (uint32 little-endian)
      bytes 8-11: num_channels (uint32 little-endian)
      bytes 12-15: num_frames (uint32 little-endian)  0 = streaming/unknown
  - Body: float32 little-endian samples, interleaved by channel
"""

import sys
import struct
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path

PIPE_MAGIC = b'SPAW'
PIPE_HEADER_FMT = '<4sIII'
PIPE_HEADER_SIZE = struct.calcsize(PIPE_HEADER_FMT)  # 16 bytes


@dataclass
class AudioData:
    samples: np.ndarray   # shape: (frames, channels), dtype float32
    sample_rate: int

    @property
    def channels(self) -> int:
        return self.samples.shape[1] if self.samples.ndim == 2 else 1

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def as_mono(self) -> 'AudioData':
        if self.channels == 1:
            return self
        return AudioData(self.samples.mean(axis=1, keepdims=True).astype(np.float32), self.sample_rate)

    def as_stereo(self) -> 'AudioData':
        if self.channels == 2:
            return self
        if self.channels == 1:
            return AudioData(np.repeat(self.samples, 2, axis=1), self.sample_rate)
        raise ValueError(f"Cannot convert {self.channels}-channel audio to stereo directly")


_PYDUB_FORMATS = {'.mp3', '.m4a', '.aac', '.mp4'}


def load(path: str | Path) -> AudioData:
    """Load an audio file. Supports WAV, FLAC, OGG via soundfile; MP3/M4A/AAC via pydub/ffmpeg."""
    path = Path(path)
    if path.suffix.lower() in _PYDUB_FORMATS:
        return _load_via_pydub(path)
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return AudioData(data, sr)


def _load_via_pydub(path: Path) -> AudioData:
    from pydub import AudioSegment
    seg = AudioSegment.from_file(str(path))
    sr = seg.frame_rate
    channels = seg.channels
    raw = np.array(seg.get_array_of_samples(), dtype=np.int16)
    samples = raw.reshape(-1, channels).astype(np.float32) / 32768.0
    return AudioData(samples, sr)


def save(audio: AudioData, path: str | Path, format: str | None = None) -> None:
    """Save audio to a file. Format inferred from extension if not given."""
    path = Path(path)
    fmt = format or path.suffix.lstrip('.').upper()
    if fmt == 'WAV':
        sf_format = 'WAV'
        subtype = 'PCM_16'
    elif fmt == 'FLAC':
        sf_format = 'FLAC'
        subtype = 'PCM_24'
    elif fmt in ('OGG', 'VORBIS'):
        sf_format = 'OGG'
        subtype = 'VORBIS'
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    sf.write(str(path), audio.samples, audio.sample_rate, format=sf_format, subtype=subtype)


def read_pipe(stream=None) -> AudioData:
    """Read AudioData from a pipe stream (default: stdin binary)."""
    if stream is None:
        stream = sys.stdin.buffer
    header = stream.read(PIPE_HEADER_SIZE)
    if len(header) < PIPE_HEADER_SIZE:
        raise ValueError("Incomplete pipe header")
    magic, sr, channels, frames = struct.unpack(PIPE_HEADER_FMT, header)
    if magic != PIPE_MAGIC:
        raise ValueError(f"Invalid pipe magic: {magic!r} (expected {PIPE_MAGIC!r})")
    if frames == 0:
        raw = stream.read()
    else:
        raw = stream.read(frames * channels * 4)
    samples = np.frombuffer(raw, dtype='<f4').reshape(-1, channels)
    return AudioData(samples.astype(np.float32), sr)


def write_pipe(audio: AudioData, stream=None) -> None:
    """Write AudioData to a pipe stream (default: stdout binary)."""
    if stream is None:
        stream = sys.stdout.buffer
    header = struct.pack(PIPE_HEADER_FMT, PIPE_MAGIC, audio.sample_rate, audio.channels, audio.frames)
    stream.write(header)
    stream.write(audio.samples.astype('<f4').tobytes())
    stream.flush()


def is_pipe(stream) -> bool:
    """Return True if the stream is a pipe/non-interactive."""
    return not stream.isatty()


def load_input(path: str | None) -> AudioData:
    """
    Load audio from a file path, or from stdin if path is '-' or stdin is a pipe.
    """
    if path == '-' or (path is None and is_pipe(sys.stdin)):
        return read_pipe()
    if path is None:
        raise ValueError("No input: provide a file path or pipe audio via stdin")
    return load(path)


def save_output(audio: AudioData, path: str | None, format: str | None = None) -> None:
    """
    Save audio to a file path, or to stdout if path is '-' or stdout is a pipe.
    """
    if path == '-' or (path is None and is_pipe(sys.stdout)):
        write_pipe(audio)
        return
    if path is None:
        raise ValueError("No output: provide a file path or pipe audio via stdout")
    save(audio, path, format=format)
