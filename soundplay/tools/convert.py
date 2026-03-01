"""sp-convert: Convert between audio formats (wav, flac, ogg, mp3)."""

import click
from pathlib import Path
from soundplay.core.audio import load, save, AudioData


_SUPPORTED_OUT = {'wav', 'flac', 'ogg'}
_SUPPORTED_IN  = {'wav', 'flac', 'ogg', 'mp3', 'm4a', 'aac'}


@click.command()
@click.argument('input')
@click.argument('output', default=None, required=False)
@click.option('--format', 'fmt', default=None,
              type=click.Choice(['wav', 'flac', 'ogg'], case_sensitive=False),
              help='Output format. Inferred from OUTPUT extension if omitted.')
def main(input, output, fmt):
    """Convert between audio formats.

    Reads WAV, FLAC, OGG, MP3, M4A/AAC.
    Writes WAV (PCM 16-bit), FLAC (24-bit), or OGG (Vorbis).

    \b
    INPUT   Source audio file.
    OUTPUT  Destination file. Defaults to INPUT with new extension.

    \b
    Examples:
      sp-convert song.mp3 song.wav
      sp-convert recording.wav recording.flac
      sp-convert song.m4a --format ogg song.ogg
      sp-convert song.mp3 --format flac
    """
    in_ext = Path(input).suffix.lstrip('.').lower()
    if in_ext not in _SUPPORTED_IN:
        raise click.UsageError(f"Unsupported input format: .{in_ext}")

    # Determine output format
    if output is not None:
        out_ext = Path(output).suffix.lstrip('.').lower()
        fmt = fmt or out_ext
    elif fmt is not None:
        output = str(Path(input).with_suffix(f'.{fmt}'))
    else:
        raise click.UsageError("Provide OUTPUT path or --format")

    if fmt not in _SUPPORTED_OUT:
        raise click.UsageError(
            f"Unsupported output format: {fmt}. Use one of: {', '.join(sorted(_SUPPORTED_OUT))}"
        )

    if Path(input).resolve() == Path(output).resolve():
        raise click.UsageError("Input and output must be different files")

    # ------------------------------------------------------------------ convert
    audio = load(input)
    save(audio, output, format=fmt.upper())

    click.echo(f"Converted {in_ext} → {fmt} ({audio.duration:.3f}s) → {output}", err=True)
