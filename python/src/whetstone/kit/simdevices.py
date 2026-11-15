"""SimDevices: several devices with memory budgets and a metered network between them.

A model that does not fit on one GPU is spread over several, and from then on two
things decide how fast it runs: what is stored where, and how many bytes cross between
devices. Neither can be seen on one machine, where a tensor is simply there. With
``SimDevices`` both are explicit:

    devs = SimDevices(ctx.fw, memory_bytes=[1 << 16, 1 << 16])
    devs.put(0, "w_up", shard0)                  # store on device 0; counted against its budget
    w = devs.get(0, "w_up")                      # the tensor, in the solution's framework
    there = devs.send(0, 1, x)                   # one message of x.nbytes from 0 to 1
    y = devs.all_reduce([part0, part1])          # one sum across all devices

Nothing is moved, of course: every tensor lives in this process. The object is a
ledger. It records what the solution says it stores and sends, enforces the memory
budgets, and gives the collective operations a cost, so that a test can demand
"exactly one all-reduce of the output's size" or "``comm_bytes`` equal to the routed
rows" with exact numbers.

Memory
------
``put(device, name, array)`` stores ``array`` under ``name`` and charges its size in
bytes (elements times item size) to the device. A device may never hold more than its
budget: the ``put`` that would exceed it raises ``OutOfDeviceMemory`` and stores
nothing. Putting an existing name replaces it (the old size is released first).
``free(device, name)`` releases it. ``get`` returns the stored tensor without cost.

Communication
-------------
``send(src, dst, array)`` is one message. It returns ``array`` itself (for the
receiving side to use) and adds its size to ``comm_bytes``. ``src == dst`` is not a
message and raises ``ValueError``.

Two collectives over *all* devices, each taking one tensor per device, in device
order:

    all_reduce(parts)             the elementwise sum, available on every device.
                                  Costs one all-reduce and ``n_devices * nbytes`` of
                                  one part (every device contributes its part once).
    all_gather(parts, axis=-1)    the parts concatenated along ``axis``, on every
                                  device. Costs one all-gather and ``(n_devices - 1)``
                                  times the size of the result (every device receives
                                  what the others hold).

So two shards of an ``(m, d)`` output cost ``2 * m * d * 4`` bytes to all-reduce.

Counters and logs
-----------------
    counters.sends, counters.comm_bytes, counters.all_reduces, counters.all_gathers,
    counters.bytes_placed (total bytes ever put)
    devs.messages         [(src, dst, nbytes), ...] for every send, in order
    devs.sent(src, dst)   number of sends from src to dst
    devs.used             bytes currently stored per device
    devs.peak             the most each device ever held
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from whetstone.kit.counters import Counters

_COUNTERS = ("sends", "comm_bytes", "all_reduces", "all_gathers", "bytes_placed")


class OutOfDeviceMemory(RuntimeError):
    """A ``put`` would take a device over its memory budget."""


class SimDevices:
    """A ledger of what is stored on which simulated device and what crosses between them.

    Args:
        fw: the framework adapter (``ctx.fw``).
        memory_bytes: one budget per device, in bytes; ``None`` in the list (or
            ``n_devices=`` instead of the list) for a device without a limit.
        n_devices: number of devices when no budgets are given.

    See the module docstring for the operations and their costs.
    """

    def __init__(
        self,
        fw: Any,
        memory_bytes: Sequence[int | None] | None = None,
        *,
        n_devices: int | None = None,
    ) -> None:
        if memory_bytes is None:
            if n_devices is None:
                raise ValueError("give memory_bytes=[...] or n_devices=")
            memory_bytes = [None] * n_devices
        elif n_devices is not None and n_devices != len(memory_bytes):
            raise ValueError("n_devices does not match the length of memory_bytes")
        if len(memory_bytes) < 1:
            raise ValueError("at least one device is needed")
        for budget in memory_bytes:
            if budget is not None and (isinstance(budget, bool) or budget < 0):
                raise ValueError(f"a memory budget must be None or a size >= 0, got {budget!r}")
        self.fw = fw
        self.memory_bytes: tuple[int | None, ...] = tuple(
            None if b is None else int(b) for b in memory_bytes
        )
        self.counters = Counters(*_COUNTERS)
        self.messages: list[tuple[int, int, int]] = []
        self.peak = [0] * len(self.memory_bytes)
        self._store: list[dict[str, np.ndarray]] = [{} for _ in self.memory_bytes]
        self._tensors: dict[tuple[int, str], Any] = {}

    @property
    def n_devices(self) -> int:
        return len(self.memory_bytes)

    def with_framework(self, fw: Any) -> SimDevices:
        """The same devices *with what they store*, speaking another framework, with
        fresh counters and logs. (A test places the shards once; the solution and the
        oracle each get their own ledger.)"""
        twin = object.__new__(SimDevices)
        twin.__dict__.update(self.__dict__)
        twin.fw = fw
        twin.counters = Counters(*_COUNTERS)
        twin.messages = []
        twin.peak = list(self.used)
        twin._store = [dict(stored) for stored in self._store]
        twin._tensors = {}
        return twin

    # -- memory ----------------------------------------------------------------------------

    @property
    def used(self) -> tuple[int, ...]:
        """Bytes currently stored on each device."""
        return tuple(sum(a.nbytes for a in stored.values()) for stored in self._store)

    def put(self, device: int, name: str, array: Any) -> Any:
        """Store ``array`` on ``device`` as ``name``; returns the stored tensor.

        Raises ``OutOfDeviceMemory`` (and stores nothing) if the device would then
        hold more than its budget.
        """
        index = self._device(device)
        if not isinstance(name, str) or not name:
            raise ValueError(f"a stored tensor needs a non-empty string name, got {name!r}")
        value = np.array(self._numpy(array, f"put({device}, {name!r})"))
        stored = self._store[index]
        after = self.used[index] - (stored[name].nbytes if name in stored else 0) + value.nbytes
        budget = self.memory_bytes[index]
        if budget is not None and after > budget:
            raise OutOfDeviceMemory(
                f"device {index}: storing {name!r} ({value.nbytes} bytes) would bring it to "
                f"{after} bytes, over its budget of {budget}"
            )
        value.setflags(write=False)
        stored[name] = value
        self._tensors.pop((index, name), None)
        self.counters.add("bytes_placed", value.nbytes)
        self.peak[index] = max(self.peak[index], after)
        return self.get(index, name)

    def get(self, device: int, name: str) -> Any:
        """The tensor stored on ``device`` as ``name`` (``KeyError`` if there is none)."""
        index = self._device(device)
        if name not in self._store[index]:
            raise KeyError(f"device {index} stores nothing called {name!r}")
        key = (index, name)
        if key not in self._tensors:
            array = self._store[index][name]
            self._tensors[key] = self.fw.tensor(array, dtype=array.dtype)
        return self._tensors[key]

    def names(self, device: int) -> list[str]:
        """What ``device`` stores, in the order it was put."""
        return list(self._store[self._device(device)])

    def free(self, device: int, name: str) -> None:
        """Release ``name`` on ``device`` (``KeyError`` if there is none)."""
        index = self._device(device)
        if name not in self._store[index]:
            raise KeyError(f"device {index} stores nothing called {name!r}")
        del self._store[index][name]
        self._tensors.pop((index, name), None)

    # -- communication ---------------------------------------------------------------------

    def send(self, src: int, dst: int, array: Any) -> Any:
        """One message carrying ``array`` from ``src`` to ``dst``; returns ``array``."""
        a, b = self._device(src), self._device(dst)
        if a == b:
            raise ValueError(f"send({a}, {b}): a device does not send to itself")
        nbytes = self._numpy(array, f"send({a}, {b})").nbytes
        self.counters.add("sends")
        self.counters.add("comm_bytes", nbytes)
        self.messages.append((a, b, nbytes))
        return array

    def sent(self, src: int, dst: int) -> int:
        """How many messages went from ``src`` to ``dst``."""
        pair = (self._device(src), self._device(dst))
        return sum(1 for a, b, _ in self.messages if (a, b) == pair)

    def all_reduce(self, parts: Sequence[Any]) -> Any:
        """The sum of one tensor per device (equal shapes), as every device then has it."""
        arrays = self._parts(parts, "all_reduce")
        if any(a.shape != arrays[0].shape or a.dtype != arrays[0].dtype for a in arrays):
            shapes = [(str(a.dtype), a.shape) for a in arrays]
            raise ValueError(f"all_reduce: the parts must have one shape and dtype, got {shapes}")
        total = arrays[0].copy()
        for array in arrays[1:]:  # in device order: the same float sum in every framework
            total += array
        self.counters.add("all_reduces")
        self.counters.add("comm_bytes", self.n_devices * arrays[0].nbytes)
        return self.fw.tensor(total, dtype=total.dtype)

    def all_gather(self, parts: Sequence[Any], axis: int = -1) -> Any:
        """One tensor per device, concatenated along ``axis``, as every device then has it."""
        arrays = self._parts(parts, "all_gather")
        try:
            whole = np.concatenate(arrays, axis=axis)
        except ValueError as exc:
            raise ValueError(f"all_gather: {exc}") from None
        self.counters.add("all_gathers")
        self.counters.add("comm_bytes", (self.n_devices - 1) * whole.nbytes)
        return self.fw.tensor(whole, dtype=whole.dtype)

    # -- helpers ---------------------------------------------------------------------------

    def _parts(self, parts: Sequence[Any], what: str) -> list[np.ndarray]:
        if isinstance(parts, (str, bytes)) or not hasattr(parts, "__len__"):
            raise ValueError(f"{what} takes a list with one tensor per device")
        if len(parts) != self.n_devices:
            raise ValueError(
                f"{what} takes one tensor per device ({self.n_devices}), got {len(parts)}"
            )
        return [self._numpy(part, what) for part in parts]

    def _numpy(self, array: Any, what: str) -> np.ndarray:
        if self.fw.is_tensor(array):
            return np.asarray(self.fw.numpy(array))
        if isinstance(array, np.ndarray):
            return array
        raise ValueError(f"{what}: expected a tensor, got {type(array).__name__}")

    def _device(self, device: Any) -> int:
        if isinstance(device, bool) or not isinstance(device, (int, np.integer)):
            raise TypeError(f"a device is an int index, got {device!r}")
        if not 0 <= device < self.n_devices:
            raise ValueError(f"device {device} is outside [0, {self.n_devices})")
        return int(device)

    def __repr__(self) -> str:
        return f"SimDevices(n_devices={self.n_devices}, used={self.used}, {self.counters!r})"
