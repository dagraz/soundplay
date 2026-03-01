"""Non-blocking audio playback via sounddevice."""

from __future__ import annotations

from soundplay.core.audio import AudioData

_player = None


class Player:
    def __init__(self):
        import sounddevice as sd
        self._sd = sd
        self._stream = None

    def play(self, audio: AudioData, start: float = 0.0) -> None:
        self.stop()
        start_frame = int(round(start * audio.sample_rate))
        samples = audio.samples[start_frame:]
        self._sd.play(samples, samplerate=audio.sample_rate)

    def stop(self) -> None:
        self._sd.stop()

    def pause(self) -> None:
        # sounddevice doesn't have native pause; stop is the fallback
        self._sd.stop()

    def resume(self) -> None:
        pass  # not supported without stream-level control


class _StubPlayer:
    """Fallback when sounddevice is not installed."""

    def play(self, audio: AudioData, start: float = 0.0) -> None:
        print("sounddevice not installed. Install with: pip install sounddevice")

    def stop(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


def get_player() -> Player | _StubPlayer:
    global _player
    if _player is None:
        try:
            _player = Player()
        except (ImportError, OSError):
            _player = _StubPlayer()
    return _player
