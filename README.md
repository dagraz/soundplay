# soundplay

A suite of Unix-style command-line tools for audio file manipulation.
Each tool does one thing and can be composed with others through shell
pipes to perform complex audio processing without a GUI.

## Philosophy

soundplay follows the Unix philosophy: small tools, text (and binary)
streams, composable via pipes. Audio flows between tools either as files
or as raw streams on stdin/stdout. Two stream formats are used:

- **SPAW** — raw PCM audio (16-byte header + float32 samples)
- **SPXF** — spectral data in the `.spx` format (see below)

Tools auto-detect the format from the file extension or stream magic bytes.

## Installation

Requires Python 3.10+ and `ffmpeg` (for MP3 support).

```bash
sudo apt install ffmpeg          # or brew install ffmpeg on macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Tools

### Audio I/O and inspection

#### `sp-info`
Display metadata about an audio file.

```bash
sp-info song.wav
sp-info --json song.flac
cat song.wav | sp-info          # pipe
```

#### `sp-convert`
Convert between audio formats (wav, flac, ogg). Reads wav, flac, ogg, mp3, m4a/aac.

```bash
sp-convert song.mp3 song.wav
sp-convert recording.wav recording.flac
sp-convert song.wav --format ogg
```

---

### Volume and dynamics

#### `sp-gain`
Scale volume by a linear factor or dB amount.

```bash
sp-gain 0.5 song.wav quieter.wav
sp-gain 2.0 song.wav louder.wav
sp-gain -- -6dB song.wav song_quiet.wav
sp-gain +3dB song.spx boosted.spx
```

#### `sp-normalize`
Normalize to a target peak or RMS level.

```bash
sp-normalize song.wav normalized.wav
sp-normalize --target -3 song.wav song_3db.wav
sp-normalize --mode rms --target -14 song.wav radio.wav
```

#### `sp-fade`
Apply fade-in and/or fade-out.

```bash
sp-fade --fade-in 2 --fade-out 3 song.wav faded.wav
sp-fade --fade-in 10% song.wav intro.wav
sp-fade --fade-out 5 song.spx fadeout.spx
```

---

### Combining and arranging

#### `sp-reverse`
Reverse audio or spectral data along the time axis.

```bash
sp-reverse song.wav reversed.wav
sp-reverse song.spx reversed.spx
```

#### `sp-concat`
Concatenate multiple audio or spectral files end-to-end.

```bash
sp-concat intro.wav verse.wav chorus.wav -o song.wav
sp-concat part1.spx part2.spx -o combined.spx
```

#### `sp-mix`
Blend multiple audio files with per-file weights.

```bash
sp-mix vocals.wav backing.wav -o combined.wav
sp-mix -w 0.8,0.2 vocals.wav reverb.wav -o wet.wav
```

---

### Filtering

#### `sp-filter`
Apply lowpass, highpass, bandpass, or bandstop filter.

```bash
sp-filter --type lowpass --freq 2000 song.wav muffled.wav
sp-filter --type highpass --freq 200 vocals.wav no_rumble.wav
sp-filter --type bandpass --freq 300 --freq-hi 3000 speech.wav telephone.wav
```

---

### Spectral conversion

soundplay's core feature is a lossless spectral file format (`.spx`) based
on the Short-Time Fourier Transform. Working in the spectral domain enables
manipulations that are difficult or artifact-prone on raw audio — pitch
shifting, noise removal, morphing, and component separation.

#### `sp-spectralize`
Convert an audio file to `.spx` spectral format.

```bash
sp-spectralize song.mp3 song.spx
sp-spectralize --n-fft 4096 --hop-length 256 song.wav song.spx
sp-spectralize song.wav - | sp-resynth - out.wav   # round-trip via pipe
```

#### `sp-resynth`
Convert a `.spx` file back to audio via inverse STFT.

```bash
sp-resynth song.spx reconstructed.wav
cat song.spx | sp-resynth - out.wav
```

---

### Visualisation

#### `sp-plot`
Render a `.spx` file as a PNG spectrogram, with optional musical note
frequency guidelines.

```bash
sp-plot song.spx                                      # basic spectrogram
sp-plot --notes song.spx                              # show C-note guidelines
sp-plot --notes --note-labels --all-notes song.spx    # label every note
sp-plot --fmin 80 --fmax 4000 --db-range 60 song.spx song.png
sp-plot --colormap magma --dpi 300 song.spx song.png
```

---

### Time editing

All time values accept raw seconds (`5`, `1.5`) or a percentage of the
file's duration (`10%`, `25%`). Negative values are relative to the end.

#### `sp-trim`
Trim to a time range. Works on `.spx`, `.wav`, `.flac`, `.mp3`.

```bash
sp-trim --start 5 --end 30 song.wav clip.wav
sp-trim --start 10% --end 90% song.wav clip.wav
sp-trim --start 1.5 --duration 10 song.spx clip.spx
sp-trim --end -2 song.wav                    # drop last 2 seconds
sp-trim --start -10 song.mp3 outro.wav       # last 10 seconds
```

#### `sp-expand`
Pad with silence. Works on `.spx`, `.wav`, `.flac`, `.mp3`.

```bash
sp-expand --pad-start 2 --pad-end 3 song.wav padded.wav
sp-expand --pad-start 10% --pad-end 10% song.spx padded.spx
sp-expand --pad 5% song.wav                  # equal padding both sides
```

#### `sp-loop`
Repeat a file N times.

```bash
sp-loop --times 4 beat.wav beat_x4.wav
sp-loop -n 3 phrase.spx phrase_x3.spx
cat beat.spx | sp-loop -n 8 - out.spx
```

---

### Spectral manipulation

#### `sp-transpose`
Pitch-shift a `.spx` file by shifting frequency bins. Accepts semitones or cents.

```bash
sp-transpose 12 song.spx octave_up.spx
sp-transpose -7 song.spx lower.spx
sp-transpose 100c song.spx song_100c.spx    # 100 cents = 1 semitone
sp-transpose 12 song.spx - | sp-resynth - out.wav
```

#### `sp-gate`
Zero spectral bins below a dB threshold. Useful for noise floor removal.

```bash
sp-gate noisy.spx clean.spx
sp-gate --threshold -30 noisy.spx clean.spx
cat noisy.spx | sp-gate --threshold -50 - out.spx
```

#### `sp-denoise`
Spectral subtraction noise reduction. Estimates a noise profile from a quiet
region and subtracts it from the whole file's magnitude spectrum.

```bash
sp-denoise recording.spx clean.spx
sp-denoise --noise-start 0 --noise-end 1.0 recording.spx clean.spx
sp-denoise --oversubtract 1.5 noisy.spx cleaner.spx
```

#### `sp-morph`
Interpolate between two `.spx` files for timbral transitions. Resamples the
second file to the first file's frame count, then blends with a linear alpha
envelope.

```bash
sp-morph a.spx b.spx --output morphed.spx
sp-morph a.spx b.spx --blend-start 0.0 --blend-end 1.0 --output crossfade.spx
sp-morph a.spx b.spx --output - | sp-resynth - out.wav
```

#### `sp-stretch`
Time-stretch a `.spx` file without pitch change. Resamples the STFT frame
axis by the given factor.

```bash
sp-stretch 2.0 song.spx slow.spx      # half speed
sp-stretch 0.5 song.spx fast.spx      # double speed
cat song.spx | sp-stretch 1.5 - out.spx
```

---

### Analysis

#### `sp-pitch-track`
Detect dominant pitch frame-by-frame from a `.spx` file. Outputs a CSV/TSV
table with columns: `time_s`, `freq_hz`, `midi_note`, `note_name`.

```bash
sp-pitch-track song.spx
sp-pitch-track song.spx --output pitches.csv
sp-pitch-track --fmin 80 --fmax 1000 --format tsv song.spx
cat song.spx | sp-pitch-track
```

#### `sp-rms`
Measure RMS and peak levels over sliding time windows. Accepts any audio file
or pipe. Outputs a CSV/TSV table with columns: `time_s`, `rms_db`, `peak_db`.
Silence is reported as -96.0 dBFS.

```bash
sp-rms song.wav
sp-rms --window 0.5 --hop 0.1 song.wav --output levels.csv
cat song.wav | sp-rms --format tsv
```

---

### Spectral decomposition

#### `sp-decompose`
Detect the fundamental frequencies present in a `.spx` file and write one
`.spx` per note, plus a remainder file. Joining all outputs with `sp-join`
perfectly reconstructs the original.

```bash
sp-decompose chord.spx                          # auto-detect components
sp-decompose --output-dir parts/ chord.spx
sp-decompose --max-notes 4 --prominence 20 chord.spx
sp-decompose --no-harmonics chord.spx           # fundamentals only
sp-decompose --no-remainder chord.spx           # skip remainder file
```

Output files are named `{stem}_{frequency}hz.spx` and `{stem}_remainder.spx`.

#### `sp-join`
Combine multiple `.spx` files by spectral superposition (the dual of
`sp-decompose`).

```bash
sp-join parts/chord_440.0hz.spx parts/chord_554.0hz.spx -o rejoined.spx
sp-join parts/*.spx -o rejoined.spx
sp-join a.spx b.spx -o - | sp-resynth - mix.wav
```

All inputs must share the same sample rate, FFT size, hop length, window,
and channel count.

---

### Pipe protocol

Tools detect whether they are reading from a file or a pipe and switch
behaviour automatically. Pass `-` as a path to force pipe mode.

```bash
# Spectralize → decompose → resynth the 440 Hz component
sp-spectralize song.mp3 song.spx
sp-decompose --output-dir parts/ song.spx
sp-resynth parts/song_440.0hz.spx note_a.wav

# Chained via pipes
sp-spectralize song.mp3 - | sp-trim --start 1 --end 5 - - | sp-resynth - clip.wav
```

---

## The `.spx` format

`.spx` is soundplay's binary spectral file format.

```
Bytes 0–3    Magic: SPXF
Bytes 4–7    Header length (uint32 little-endian)
Bytes 8+     UTF-8 JSON header
Remaining    float32 little-endian STFT data
```

The JSON header contains: `version`, `sample_rate`, `n_fft`, `hop_length`,
`window`, `original_frames`, `channels`, `frames`, `bins`.

The data block is a C-order float32 array of shape
`(channels, frames, bins, 2)` where the last axis is `[real, imag]`.

Since the full complex STFT is stored, inverse STFT reconstruction is
lossless to float32 precision. The same format is used for both files
and pipes.

---

## Development

```bash
source .venv/bin/activate
pip install -e .
```

Each tool lives in `soundplay/tools/<name>.py` and is registered as a
`project.scripts` entry point in `pyproject.toml`. Shared audio I/O is in
`soundplay/core/audio.py` and spectral I/O in `soundplay/core/spectral.py`.

### Testing

```bash
pip install -e '.[test]'
pytest tests/ -v
```

85 tests cover the three core modules (`audio`, `spectral`, `timeutil`) and
10 CLI tools. All tests use synthetically generated audio — no real audio
files are needed.
