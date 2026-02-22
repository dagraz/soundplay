"""sp-info: Display information about an audio file."""

import click
from soundplay.core.audio import load_input


@click.command()
@click.argument('input', default=None, required=False)
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def main(input, as_json):
    """Display information about an audio file.

    INPUT may be a file path or '-' to read from stdin pipe.
    """
    audio = load_input(input)

    if as_json:
        import json
        print(json.dumps({
            'sample_rate': audio.sample_rate,
            'channels': audio.channels,
            'frames': audio.frames,
            'duration': audio.duration,
        }))
    else:
        print(f"Sample rate : {audio.sample_rate} Hz")
        print(f"Channels    : {audio.channels}")
        print(f"Frames      : {audio.frames}")
        print(f"Duration    : {audio.duration:.3f} s")
