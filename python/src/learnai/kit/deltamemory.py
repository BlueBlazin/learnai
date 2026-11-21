"""DeltaMemory: the memory of a gated delta layer, reachable only through counted calls.

``gated-delta-chunked`` asks for the delta rule a chunk at a time. A token-by-token
loop returns the same numbers, so the judge counts instead: the memory is handed over
as an object that can be read and updated only in batched calls, each counted once
whatever its number of rows. A loop over tokens needs one read per token however it
is written or compiled; the chunked form needs one per chunk.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from learnai.kit.counters import Counters


class DeltaMemory:
    """The memory ``S`` of ``H`` heads, ``(H, Dv, Dk)``, behind two counted calls.

        memory = DeltaMemory(ctx.fw, state, max_rows=8)      # state (H, Dv, Dk)
        out = memory.read(rows)                              # rows (H, R, Dk) -> (H, R, Dv)
        memory = memory.update(decay, values, keys)          # a new memory, same counters
        check.cost(memory.counters, "reads", at_most=3, unit="calls")
        memory.peek()                                        # numpy copy, for tests

    Args:
        fw: the framework adapter (``ctx.fw``).
        state: the initial memory, array or tensor ``(H, Dv, Dk)``.
        max_rows: the largest number of rows ``R`` one call takes. Keep it small
            against ``Dk``, so that the memory cannot be read out whole and processed
            elsewhere within the call limits.

    Attributes:
        shape: ``(H, Dv, Dk)``.
        max_rows: as given.
        counters: ``reads`` and ``writes``, one per call. A memory made by ``update``
            shares the counters of the one it came from.

    Hand it to a solution guarded (``class Memory(Guarded, DeltaMemory)``, with
    ``peek`` added to ``refused``): the counters and the array behind them are then
    out of the solution's reach.
    """

    def __init__(self, fw: Any, state: Any, max_rows: int) -> None:
        self.fw = fw
        self._state = np.array(fw.numpy(state) if fw.is_tensor(state) else state, dtype=np.float32)
        if self._state.ndim != 3:
            raise ValueError(f"state must have shape (H, Dv, Dk), got {self._state.shape}")
        self.max_rows = int(max_rows)
        self.counters = Counters("reads", "writes")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._state.shape

    def with_framework(self, fw: Any) -> DeltaMemory:
        """The same memory speaking another framework, with fresh counters."""
        twin = object.__new__(DeltaMemory)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters("reads", "writes")
        return twin

    def read(self, rows: Any) -> Any:
        """``rows @ S^T``: rows ``(H, R, Dk)`` to ``(H, R, Dv)``. One read, whatever ``R``."""
        rows = self._rows(rows, self._state.shape[2], "rows")
        self.counters.add("reads")
        return self.fw.tensor(rows @ self._state.transpose(0, 2, 1))

    def update(self, decay: Any, values: Any, keys: Any) -> DeltaMemory:
        """A new memory holding ``decay * S + values^T keys``. One write, whatever ``R``.

        ``decay`` is ``(H,)``, ``values`` ``(H, R, Dv)``, ``keys`` ``(H, R, Dk)``. This
        memory is unchanged; the new one shares its counters.
        """
        heads, d_v, d_k = self._state.shape
        values, keys = self._rows(values, d_v, "values"), self._rows(keys, d_k, "keys")
        decay = np.asarray(self.fw.numpy(decay) if self.fw.is_tensor(decay) else decay, np.float32)
        if decay.shape != (heads,) or values.shape[1] != keys.shape[1]:
            raise ValueError(
                f"decay must have shape ({heads},) and values and keys the same number of "
                f"rows, got {decay.shape}, {values.shape}, {keys.shape}"
            )
        self.counters.add("writes")
        new = object.__new__(type(self))  # a guarded memory stays guarded
        new.__dict__.update(self.__dict__)
        new._state = self._state * decay[:, None, None] + values.transpose(0, 2, 1) @ keys
        return new

    def peek(self) -> np.ndarray:
        """The memory as a numpy copy, for tests. Not counted."""
        return self._state.copy()

    def _rows(self, value: Any, width: int, what: str) -> np.ndarray:
        rows = np.asarray(self.fw.numpy(value) if self.fw.is_tensor(value) else value, np.float32)
        heads = self._state.shape[0]
        if rows.ndim != 3 or rows.shape[0] != heads or rows.shape[2] != width:
            raise ValueError(f"{what} must have shape ({heads}, R, {width}), got {rows.shape}")
        if not 1 <= rows.shape[1] <= self.max_rows:
            raise ValueError(
                f"{what}: one call takes 1 to max_rows = {self.max_rows} rows, got {rows.shape[1]}"
            )
        return rows

    def __repr__(self) -> str:
        return f"DeltaMemory(shape={self.shape}, max_rows={self.max_rows}, {self.counters!r})"
