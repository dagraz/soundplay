"""Shared time-value parsing: accepts raw seconds or a percentage string (e.g. '10%')."""

import click


class TimeParam(click.ParamType):
    """Click parameter type that accepts either a float (seconds) or 'X%' (percent of total)."""
    name = 'TIME'

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, float):
            return str(value)
        s = str(value).strip()
        if s.endswith('%'):
            try:
                float(s[:-1])
                return s
            except ValueError:
                self.fail(f"{s!r} is not a valid percentage", param, ctx)
        try:
            float(s)
            return s
        except ValueError:
            self.fail(f"{s!r} is not a valid time value — use seconds (1.5) or percent (10%)",
                      param, ctx)


TIME = TimeParam()


def resolve(value: str | None, total: float) -> float | None:
    """Convert a stored TimeParam string to an absolute number of seconds."""
    if value is None:
        return None
    s = str(value).strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0 * total
    return float(s)
