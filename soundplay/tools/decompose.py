"""sp-decompose: Decompose a chord .spx file into per-note spectral components."""

import sys
import numpy as np
import click
from pathlib import Path
from soundplay.core.spectral import load, read_pipe, save, SpectralData
from soundplay.core.audio import is_pipe


def _mean_magnitude(sd: SpectralData) -> np.ndarray:
    """Average magnitude spectrum across time frames (and channels). Shape: (bins,)"""
    # stft shape: (channels, frames, bins)
    return np.abs(sd.stft).mean(axis=(0, 1))  # → (bins,)


def _find_fundamentals(mean_mag: np.ndarray, freq_bins: np.ndarray,
                       min_freq: float, max_notes: int,
                       prominence_db: float) -> list[tuple[float, int]]:
    """
    Detect prominent peaks in the mean magnitude spectrum.
    Returns list of (frequency_hz, bin_index) sorted by frequency.
    """
    from scipy.signal import find_peaks

    db = 20.0 * np.log10(np.maximum(mean_mag, 1e-10))

    # Suppress bins below min_freq
    db_search = db.copy()
    db_search[freq_bins < min_freq] = -np.inf

    peaks, props = find_peaks(db_search, prominence=prominence_db)
    if len(peaks) == 0:
        return []

    # Keep top max_notes by prominence
    order = np.argsort(-props['prominences'])
    peaks = peaks[order[:max_notes]]

    # Sieve: remove any peak that is close to an integer harmonic of a
    # lower-frequency peak. Tolerance of 5% (~84 cents) handles slight
    # intonation and bin-quantisation offsets.
    candidates = sorted((freq_bins[p], int(p)) for p in peaks)
    fundamentals = []
    for hz, idx in candidates:
        is_harmonic = any(
            abs(hz / f - round(hz / f)) / round(hz / f) < 0.05
            for f, _ in fundamentals
            if round(hz / f) >= 2        # only flag as harmonic of n>=2
        )
        if not is_harmonic:
            fundamentals.append((hz, idx))

    return fundamentals


def _harmonic_bins(fundamental_hz: float, freq_bins: np.ndarray,
                   bin_window: int, max_harmonics: int) -> np.ndarray:
    """
    Boolean mask: True for bins within bin_window of each harmonic of fundamental_hz.
    """
    mask = np.zeros(len(freq_bins), dtype=bool)
    for n in range(1, max_harmonics + 1):
        target = n * fundamental_hz
        if target > freq_bins[-1]:
            break
        nearest = int(np.argmin(np.abs(freq_bins - target)))
        lo = max(0, nearest - bin_window)
        hi = min(len(freq_bins) - 1, nearest + bin_window)
        mask[lo:hi + 1] = True
    return mask


def _fundamental_only_bins(fundamental_hz: float, freq_bins: np.ndarray,
                            bin_window: int) -> np.ndarray:
    """Boolean mask: True only for bins within bin_window of the fundamental."""
    mask = np.zeros(len(freq_bins), dtype=bool)
    nearest = int(np.argmin(np.abs(freq_bins - fundamental_hz)))
    lo = max(0, nearest - bin_window)
    hi = min(len(freq_bins) - 1, nearest + bin_window)
    mask[lo:hi + 1] = True
    return mask


def _apply_mask(sd: SpectralData, bin_mask: np.ndarray) -> SpectralData:
    """Zero out all bins not in bin_mask, return new SpectralData."""
    masked = sd.stft.copy()
    masked[:, :, ~bin_mask] = 0.0
    return SpectralData(
        stft=masked,
        sample_rate=sd.sample_rate,
        n_fft=sd.n_fft,
        hop_length=sd.hop_length,
        window=sd.window,
        original_frames=sd.original_frames,
    )


@click.command()
@click.argument('input', default=None, required=False)
@click.option('--output-dir', default=None,
              help='Directory for output files. Defaults to same directory as INPUT.')
@click.option('--no-harmonics', is_flag=True,
              help='Extract only the fundamental bin, stripping harmonics.')
@click.option('--max-notes', default=12, show_default=True,
              help='Maximum number of notes to detect.')
@click.option('--min-freq', default=50.0, show_default=True,
              help='Minimum frequency to consider for fundamentals (Hz).')
@click.option('--prominence', default=15.0, show_default=True,
              help='Minimum peak prominence in dB. Higher = only strong notes detected.')
@click.option('--bin-window', default=2, show_default=True,
              help='Number of bins on each side to include per harmonic/fundamental.')
@click.option('--max-harmonics', default=16, show_default=True,
              help='Maximum number of harmonics to include per fundamental.')
@click.option('--no-remainder', is_flag=True,
              help='Skip writing the remainder file.')
def main(input, output_dir, no_harmonics, max_notes, min_freq,
         prominence, bin_window, max_harmonics, no_remainder):
    """Decompose a chord .spx file into per-note spectral components.

    Detects the dominant fundamental frequencies in the spectrum and
    outputs one .spx file per detected note, named by frequency.

    \b
    INPUT   .spx file or '-' for stdin pipe.

    By default also writes a remainder file ({stem}_remainder.spx) containing
    all spectral energy not captured by any component mask, so that joining
    all components plus the remainder perfectly reconstructs the original.

    \b
    Examples:
      sp-decompose chord.spx
      sp-decompose --max-notes 4 --prominence 20 chord.spx
      sp-decompose --no-harmonics chord.spx
      sp-decompose --no-remainder chord.spx
      sp-decompose --output-dir parts/ chord.spx
      cat chord.spx | sp-decompose -
    """
    if input == '-' or (input is None and is_pipe(sys.stdin)):
        sd = read_pipe()
        stem = 'stdin'
        src_dir = Path('.')
    elif input is None:
        raise click.UsageError("No input: provide an INPUT file or pipe via stdin")
    else:
        sd = load(input)
        stem = Path(input).stem
        src_dir = Path(input).parent

    out_dir = Path(output_dir) if output_dir else src_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frequency axis
    nyquist = sd.sample_rate / 2.0
    freq_bins = np.linspace(0, nyquist, sd.bins)

    # Detect fundamentals
    mean_mag = _mean_magnitude(sd)
    fundamentals = _find_fundamentals(mean_mag, freq_bins,
                                      min_freq=min_freq,
                                      max_notes=max_notes,
                                      prominence_db=prominence)

    if not fundamentals:
        click.echo("No prominent peaks detected. Try lowering --prominence.", err=True)
        sys.exit(1)

    click.echo(f"Detected {len(fundamentals)} component(s):", err=True)

    results = []
    claimed = np.zeros(sd.bins, dtype=bool)  # bins assigned to any component so far

    for hz, bin_idx in fundamentals:
        if no_harmonics:
            mask = _fundamental_only_bins(hz, freq_bins, bin_window)
            label = 'fundamental only'
        else:
            mask = _harmonic_bins(hz, freq_bins, bin_window, max_harmonics)
            label = f'+ harmonics (window ±{bin_window} bins)'

        # Strip bins already owned by an earlier component so masks never overlap.
        mask = mask & ~claimed
        claimed |= mask

        component = _apply_mask(sd, mask)
        out_name = f"{stem}_{hz:.1f}hz.spx"
        out_path = out_dir / out_name
        save(component, out_path)

        energy = float(np.abs(component.stft).mean())
        click.echo(f"  {hz:8.2f} Hz  →  {out_path}  [{label}, mean energy {energy:.4f}]",
                   err=True)
        results.append(out_path)

    if not no_remainder:
        remainder = _apply_mask(sd, ~claimed)
        rem_path = out_dir / f"{stem}_remainder.spx"
        save(remainder, rem_path)
        rem_energy = float(np.abs(remainder.stft).mean())
        uncovered_pct = (~claimed).sum() / sd.bins * 100
        click.echo(
            f"  remainder  →  {rem_path}  "
            f"[{uncovered_pct:.1f}% of bins, mean energy {rem_energy:.4f}]",
            err=True
        )
        results.append(rem_path)

    click.echo(f"\nWrote {len(results)} file(s) to {out_dir}/", err=True)
