"""Tests for soundplay.core.timeutil."""

import pytest
from click import Context, Command

from soundplay.core.timeutil import TimeParam, resolve, TIME


class TestTimeParamConvert:
    def _convert(self, value):
        return TIME.convert(value, None, None)

    def test_seconds_string(self):
        assert self._convert("1.5") == "1.5"

    def test_seconds_float(self):
        assert self._convert(2.0) == "2.0"

    def test_percentage(self):
        assert self._convert("10%") == "10%"

    def test_none_passthrough(self):
        assert self._convert(None) is None

    def test_invalid_value_fails(self):
        with pytest.raises(Exception):
            self._convert("abc")

    def test_invalid_percent_fails(self):
        with pytest.raises(Exception):
            self._convert("abc%")


class TestResolve:
    def test_seconds(self):
        assert resolve("1.5", 10.0) == 1.5

    def test_percentage(self):
        assert resolve("50%", 10.0) == pytest.approx(5.0)

    def test_negative_seconds(self):
        assert resolve("-2.0", 10.0) == -2.0

    def test_none(self):
        assert resolve(None, 10.0) is None
