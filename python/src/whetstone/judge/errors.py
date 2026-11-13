"""The exception a check raises when a solution's answer is wrong."""

from __future__ import annotations


class CheckFailure(Exception):
    """A case failed: the solution ran but its answer is wrong.

    ``message`` is one short line and is always shown, even for hidden cases, so it
    must never contain expected values. ``detail`` is the longer explanation (expected
    vs. got excerpts); the harness drops it for hidden cases.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
