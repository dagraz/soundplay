# soundplay TODO

## Implemented

- [x] `sp-gain`     — scale volume by a factor or dB amount
- [x] `sp-normalize` — normalize to target peak or RMS level
- [x] `sp-fade`     — apply fade-in and/or fade-out
- [x] `sp-reverse`  — reverse audio or spectral data along the time axis
- [x] `sp-concat`   — concatenate multiple audio/spectral files end-to-end
- [x] `sp-mix`      — blend multiple audio files with per-file weights
- [x] `sp-filter`   — lowpass / highpass / bandpass / notch filter
- [x] `sp-convert`  — convert between audio formats (wav ↔ flac ↔ ogg)
- [x] Test suite    — 85 tests covering core modules and all CLI tools

## Spectral-domain tools

These operate on `.spx` files and exploit the spectral representation
for operations that are difficult or artifact-prone on raw audio.

- [ ] `sp-transpose`  — pitch-shift by shifting frequency bins; accepts semitones or cents
- [ ] `sp-gate`       — zero out bins below a dB threshold; useful for noise floor removal
- [ ] `sp-denoise`    — spectral subtraction: capture a noise profile from a quiet region,
                        subtract it across the whole file
- [ ] `sp-morph`      — interpolate between two `.spx` files over time for timbral transitions
- [ ] `sp-stretch`    — time-stretch without pitch change by duplicating/interpolating STFT frames

## Analysis tools

- [ ] `sp-pitch-track` — detect dominant pitch frame-by-frame; output as time-series (CSV/TSV)
- [ ] `sp-rms`         — measure RMS and peak levels over sliding time windows
