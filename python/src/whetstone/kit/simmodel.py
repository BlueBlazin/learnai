"""SimModel: a stand-in for a model server, living in simulated time."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from whetstone.kit.counters import Counters


class SimModel:
    """A batch-processing "model" whose latency is virtual.

    ``await model(batch)`` takes ``base_s + per_item_s * len(batch)`` virtual seconds
    and returns one result per item (``fn(item)``; by default the item itself). Like a
    single GPU, it processes one batch at a time: a call made while another is running
    raises ``RuntimeError``, and so does a batch that is empty or over
    ``max_batch_size``.

    Every call is recorded in the simulation's trace as a ``batch`` event when it
    starts (fields ``size`` and ``items``) and a ``batch_done`` event when it ends.
    Counters: ``batches`` and ``items``.
    """

    def __init__(
        self,
        sim: Any,
        *,
        base_s: float = 0.010,
        per_item_s: float = 0.001,
        max_batch_size: int | None = None,
        fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.sim = sim
        self.base_s = base_s
        self.per_item_s = per_item_s
        self.max_batch_size = max_batch_size
        self.fn = fn or (lambda item: item)
        self.counters = Counters("batches", "items")
        self._busy = False

    async def __call__(self, batch: Sequence[Any]) -> list[Any]:
        items = list(batch)
        if not items:
            raise RuntimeError("SimModel was called with an empty batch")
        if self.max_batch_size is not None and len(items) > self.max_batch_size:
            raise RuntimeError(
                f"batch of {len(items)} exceeds the model's max_batch_size {self.max_batch_size}"
            )
        if self._busy:
            raise RuntimeError("SimModel is busy: it processes one batch at a time")
        self._busy = True
        try:
            self.counters.add("batches")
            self.counters.add("items", len(items))
            self.sim.trace.record("batch", size=len(items), items=items)
            await self.sim.sleep(self.base_s + self.per_item_s * len(items))
            self.sim.trace.record("batch_done", size=len(items), items=items)
            return [self.fn(item) for item in items]
        finally:
            self._busy = False
