"""The exception a check raises when a solution's answer is wrong."""

from __future__ import annotations


class CheckFailure(Exception):
    """A case failed: the solution ran but its answer is wrong.

    ``message`` is one short line. ``detail`` is the longer explanation (the call, and
    expected vs. got excerpts). Both are shown for every case, public or hidden.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
