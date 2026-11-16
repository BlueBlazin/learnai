"""Instrumentation counters for the cost judge."""

from __future__ import annotations


class Counters:
    """Named integer counters that a kit object increments as it does work.

        lm.counters.tokens_processed      # read one
        lm.counters.snapshot()            # {"forward_calls": 3, "tokens_processed": 12}
        lm.counters.reset()               # back to zero

    Tests assert on them with ``check.cost(lm.counters, "tokens_processed", at_most=12)``.
    """

    def __init__(self, *names: str) -> None:
        self._names = names
        self.reset()

    def reset(self) -> None:
        for name in self._names:
            setattr(self, name, 0)

    def add(self, name: str, amount: int = 1) -> None:
        setattr(self, name, getattr(self, name) + amount)

    def snapshot(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self._names}

    def __repr__(self) -> str:
        inner = ", ".join(f"{name}={value}" for name, value in self.snapshot().items())
        return f"Counters({inner})"
