"""sp-repl: Interactive Python REPL for soundplay."""

from __future__ import annotations

import numpy as np
from soundplay.studio.sound import Sound, load, concat, mix


def guide(*args):
    """Print help. Call guide() for overview, guide('topic') for details.

    Topics: load, transforms, spectral, analysis, io, operators, examples
    """
    topics = {
        'load': _HELP_LOAD,
        'transforms': _HELP_TRANSFORMS,
        'spectral': _HELP_SPECTRAL,
        'analysis': _HELP_ANALYSIS,
        'io': _HELP_IO,
        'operators': _HELP_OPERATORS,
        'examples': _HELP_EXAMPLES,
    }

    if not args:
        print(_HELP_OVERVIEW)
        return

    topic = str(args[0]).lower()
    if topic in topics:
        print(topics[topic])
    else:
        print(f"Unknown topic: {topic!r}")
        print(f"Available topics: {', '.join(topics)}")


_HELP_OVERVIEW = """
\033[1msoundplay studio\033[0m — interactive audio workbench
══════════════════════════════════════════════

\033[1mQuick start:\033[0m
  s = load("voice.wav")        # load any audio or .spx file
  s                             # Sound("voice.wav", 2ch, 44100Hz, 12.340s)
  s2 = s.trim(1, 5).transpose(12).gate(-30).fade(fade_in=0.5)
  s2.play()                     # non-blocking playback
  s2.plot()                     # spectrogram
  combined = s * 0.5 + s2 * 0.3

\033[1mAvailable methods:\033[0m
  \033[36mtransforms:\033[0m  gain  normalize  fade  reverse  trim  loop  pad  filter
  \033[36mspectral:\033[0m    transpose  gate  denoise  stretch  morph
  \033[36manalysis:\033[0m    decompose  pitch_track  rms
  \033[36mi/o:\033[0m         load  save  play  stop  pause  plot  waveform
  \033[36moperators:\033[0m   * (gain)  + (mix)

\033[1mDetailed help:\033[0m
  guide('transforms')    guide('spectral')     guide('analysis')
  guide('io')            guide('operators')    guide('examples')
  guide('load')
"""

_HELP_LOAD = """
\033[1mLoading sounds\033[0m
══════════════

  s = load("file.wav")     # WAV, FLAC, OGG, MP3, M4A, AAC
  s = load("file.spx")     # spectral .spx format (lazy — no iSTFT until needed)

  Sound(audio=audio_data)              # wrap an AudioData directly
  Sound(spectral=spectral_data)        # wrap a SpectralData directly

Loaded Sound objects convert between audio and spectral representations
lazily — accessing .spectral on an audio-loaded Sound computes the STFT
only on first access, and vice versa.
"""

_HELP_TRANSFORMS = """
\033[1mDual-domain transforms\033[0m (work on audio or spectral, whichever is cached)
══════════════════════

  s.gain(factor)                      # linear gain (0.5 = halve, 2.0 = double)
  s.normalize(target_db=0, mode='peak')  # 'peak' or 'rms'
  s.fade(fade_in=0.5, fade_out=1.0)  # seconds
  s.reverse()                         # time reversal
  s.trim(start=1.0, end=5.0)         # seconds
  s.loop(times=3)                     # repeat N times
  s.pad(start=0.5, end=0.5)          # add silence (seconds)
  s.filter('lowpass', 2000)           # lowpass/highpass/bandpass/bandstop
  s.filter('bandpass', 300, freq_hi=3000)

All transforms return a new Sound — the original is unchanged.
Chain freely: s.trim(1, 5).gain(0.8).fade(fade_out=0.5)
"""

_HELP_SPECTRAL = """
\033[1mSpectral-domain transforms\033[0m (auto-convert to spectral if needed)
══════════════════════════

  s.transpose(semitones)              # pitch shift (12 = octave up)
  s.gate(threshold_db=-40)            # zero bins below threshold
  s.denoise(noise_start=0, noise_end=0.5, oversubtract=1.0)
  s.stretch(factor)                   # time stretch (2.0 = half speed)
  s.morph(other, blend_start=0, blend_end=1)  # spectral crossfade
"""

_HELP_ANALYSIS = """
\033[1mAnalysis\033[0m
════════

  parts = s.decompose(max_notes=12, min_freq=50, prominence=15)
      Returns a list of Sound objects, one per detected harmonic component.
      Each part contains the fundamental + harmonics of one note.

  rows = s.pitch_track(fmin=50, fmax=2000)
      Returns [(time_s, freq_hz, midi_note, note_name), ...]

  rows = s.rms(window=0.1, hop=None)
      Returns [(time_s, rms_db, peak_db), ...]
"""

_HELP_IO = """
\033[1mI/O, playback, and visualization\033[0m
════════════════════════════════

  s = load("file.wav")        # load from file
  s.save("output.wav")        # save as WAV/FLAC/OGG (by extension)
  s.save("output.spx")        # save as spectral .spx

  s.play()                     # non-blocking playback (needs sounddevice)
  s.play(start=2.5)           # start from 2.5 seconds
  s.stop()                    # stop playback
  s.pause()                   # pause playback

  s.plot()                     # spectrogram (needs matplotlib)
  s.plot(notes=True, note_labels=True)  # with note guidelines
  s.waveform()                 # amplitude vs time plot
"""

_HELP_OPERATORS = """
\033[1mOperators\033[0m
═════════

  s * 0.5          # gain by factor (same as s.gain(0.5))
  0.5 * s          # same thing
  s1 + s2          # mix two sounds (equal weight)

Combine for weighted mix:
  combined = s1 * 0.7 + s2 * 0.3

For more control over mixing:
  mix(s1, s2, s3, weights=[0.5, 0.3, 0.2])
  concat(s1, s2, s3)          # end-to-end concatenation
"""

_HELP_EXAMPLES = """
\033[1mExamples\033[0m
════════

\033[36m# Load, trim, transpose, save\033[0m
  s = load("voice.wav")
  clip = s.trim(1, 5).transpose(12).fade(fade_in=0.5)
  clip.save("clip_up_octave.wav")

\033[36m# Decompose a chord and listen to each note\033[0m
  chord = load("chord.wav")
  parts = chord.decompose(max_notes=4)
  for p in parts: print(p)
  parts[0].play()

\033[36m# Spectral morph between two sounds\033[0m
  a = load("piano.wav")
  b = load("strings.wav")
  morphed = a.morph(b, blend_start=0.0, blend_end=1.0)
  morphed.save("piano_to_strings.wav")

\033[36m# Denoise a recording\033[0m
  noisy = load("recording.wav")
  clean = noisy.denoise(noise_start=0, noise_end=0.5, oversubtract=1.2)
  clean.save("clean.wav")

\033[36m# Quick weighted mix\033[0m
  drums = load("drums.wav")
  bass = load("bass.wav")
  submix = drums * 0.8 + bass * 0.5
  submix.normalize().save("submix.wav")
"""


class _SpPrompt:
    """Custom IPython prompt: sp> and ..> for continuation."""

    in_prompt = '\033[32msp>\033[0m '
    in2_prompt = '\033[32m..>\033[0m '
    out_prompt = ''
    rewrite_prompt = ''


def _configure_prompt(ipython):
    """Set up the sp> prompt on an IPython instance."""
    ipython.prompts_class = type(
        'SpPrompts', (),
        {
            '__init__': lambda self, shell: None,
            'in_prompt_tokens': lambda self: [
                ('class:prompt', 'sp> '),
            ],
            'continuation_prompt_tokens': lambda self, width=None: [
                ('class:prompt', '..> '),
            ],
            'out_prompt_tokens': lambda self: [],
            'rewrite_prompt_tokens': lambda self: [],
        },
    )
    # Apply immediately by refreshing prompts
    ipython.prompts = ipython.prompts_class(ipython)


def main():
    try:
        from IPython.terminal.embed import InteractiveShellEmbed
        from IPython.terminal.prompts import Prompts, Token
    except ImportError:
        print("IPython not installed. Install with: pip install 'soundplay[studio]'")
        return

    class SpPrompts(Prompts):
        def in_prompt_tokens(self):
            return [(Token.Prompt, 'sp> ')]

        def continuation_prompt_tokens(self, width=None):
            return [(Token.Prompt, '..> ')]

        def out_prompt_tokens(self):
            return []

    user_ns = {
        'Sound': Sound,
        'load': load,
        'concat': concat,
        'mix': mix,
        'np': np,
        'guide': guide,
    }

    shell = InteractiveShellEmbed(user_ns=user_ns, banner1='')
    shell.prompts = SpPrompts(shell)

    print("soundplay v0.1 — type guide() for help")
    shell()


if __name__ == '__main__':
    main()
