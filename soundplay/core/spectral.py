"""
Spectral data container and .spx file format.

.spx format:
    bytes 0-3:   magic b'SPXF'
    bytes 4-7:   header length (uint32 little-endian)
    bytes 8+:    UTF-8 JSON header
    remaining:   float32 little-endian binary data

JSON header fields:
    version         int     format version (1)
    sample_rate     int     original audio sample rate
    n_fft           int     FFT window size
    hop_length      int     samples between frames
    window          str     window function name
    original_frames int     frame count of source audio
    channels        int     number of audio channels
    frames          int     number of STFT time frames
    bins            int     number of frequency bins (n_fft // 2 + 1)

Data layout:
    float32 array of shape (channels, frames, bins, 2)
    last axis is [real, imag]
    stored in C order (row-major)
"""

import json
import struct
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import get_window

MAGIC = b'SPXF'
HEADER_LEN_FMT = '<4sI'
HEADER_LEN_SIZE = struct.calcsize(HEADER_LEN_FMT)  # 8 bytes
FORMAT_VERSION = 1


@dataclass
class SpectralData:
    # complex64 array, shape: (channels, frames, bins)
    stft: np.ndarray
    sample_rate: int
    n_fft: int
    hop_length: int
    window: str
    original_frames: int

    @property
    def channels(self) -> int:
        return self.stft.shape[0]

    @property
    def frames(self) -> int:
        return self.stft.shape[1]

    @property
    def bins(self) -> int:
        return self.stft.shape[2]

    @property
    def duration(self) -> float:
        return self.original_frames / self.sample_rate


def _header_dict(sd: SpectralData) -> dict:
    return {
        'version': FORMAT_VERSION,
        'sample_rate': sd.sample_rate,
        'n_fft': sd.n_fft,
        'hop_length': sd.hop_length,
        'window': sd.window,
        'original_frames': sd.original_frames,
        'channels': sd.channels,
        'frames': sd.frames,
        'bins': sd.bins,
    }


def _stft_to_bytes(sd: SpectralData) -> bytes:
    # Store as (channels, frames, bins, 2) float32
    ri = np.stack([sd.stft.real, sd.stft.imag], axis=-1).astype('<f4')
    return ri.tobytes()


def _bytes_to_stft(data: bytes, channels: int, frames: int, bins: int) -> np.ndarray:
    ri = np.frombuffer(data, dtype='<f4').reshape(channels, frames, bins, 2)
    return (ri[..., 0] + 1j * ri[..., 1]).astype(np.complex64)


def save(sd: SpectralData, path: str | Path) -> None:
    header_json = json.dumps(_header_dict(sd)).encode('utf-8')
    body = _stft_to_bytes(sd)
    with open(path, 'wb') as f:
        f.write(struct.pack(HEADER_LEN_FMT, MAGIC, len(header_json)))
        f.write(header_json)
        f.write(body)


def load(path: str | Path) -> SpectralData:
    with open(path, 'rb') as f:
        return _read_stream(f)


def write_pipe(sd: SpectralData, stream=None) -> None:
    if stream is None:
        stream = sys.stdout.buffer
    header_json = json.dumps(_header_dict(sd)).encode('utf-8')
    body = _stft_to_bytes(sd)
    stream.write(struct.pack(HEADER_LEN_FMT, MAGIC, len(header_json)))
    stream.write(header_json)
    stream.write(body)
    stream.flush()


def read_pipe(stream=None) -> SpectralData:
    if stream is None:
        stream = sys.stdin.buffer
    return _read_stream(stream)


def _read_stream(stream) -> SpectralData:
    prefix = stream.read(HEADER_LEN_SIZE)
    if len(prefix) < HEADER_LEN_SIZE:
        raise ValueError("Truncated .spx stream: missing header prefix")
    magic, hlen = struct.unpack(HEADER_LEN_FMT, prefix)
    if magic != MAGIC:
        raise ValueError(f"Not a .spx file (magic={magic!r})")
    header = json.loads(stream.read(hlen).decode('utf-8'))
    if header['version'] != FORMAT_VERSION:
        raise ValueError(f"Unsupported .spx version: {header['version']}")
    body = stream.read()
    stft = _bytes_to_stft(body, header['channels'], header['frames'], header['bins'])
    return SpectralData(
        stft=stft,
        sample_rate=header['sample_rate'],
        n_fft=header['n_fft'],
        hop_length=header['hop_length'],
        window=header['window'],
        original_frames=header['original_frames'],
    )


# ---------------------------------------------------------------------------
# STFT / iSTFT using scipy (no librosa dependency)
# ---------------------------------------------------------------------------

def compute_stft(samples: np.ndarray, sample_rate: int,
                 n_fft: int = 2048, hop_length: int = 512,
                 window: str = 'hann') -> SpectralData:
    """
    Compute STFT of audio samples.

    samples: float32 array shape (frames, channels)
    Returns SpectralData with stft shape (channels, time_frames, bins)
    """
    from scipy.signal import stft as scipy_stft

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    channels = samples.shape[1]
    original_frames = samples.shape[0]
    win = get_window(window, n_fft)

    stft_channels = []
    for ch in range(channels):
        _, _, Zxx = scipy_stft(
            samples[:, ch],
            fs=sample_rate,
            window=win,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary='zeros',
            padded=True,
        )
        stft_channels.append(Zxx.T)  # transpose to (time_frames, bins)

    stft = np.stack(stft_channels, axis=0).astype(np.complex64)  # (ch, frames, bins)
    return SpectralData(
        stft=stft,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        original_frames=original_frames,
    )


def compute_istft(sd: SpectralData) -> np.ndarray:
    """
    Inverse STFT. Returns float32 array shape (frames, channels).
    """
    from scipy.signal import istft as scipy_istft

    win = get_window(sd.window, sd.n_fft)
    channels = []
    for ch in range(sd.channels):
        _, x = scipy_istft(
            sd.stft[ch].T,  # scipy expects (bins, time_frames)
            fs=sd.sample_rate,
            window=win,
            nperseg=sd.n_fft,
            noverlap=sd.n_fft - sd.hop_length,
            boundary=True,
        )
        channels.append(x)

    out = np.stack(channels, axis=1).astype(np.float32)  # (frames, channels)
    # Trim to original length
    out = out[:sd.original_frames]
    return np.clip(out, -1.0, 1.0)
