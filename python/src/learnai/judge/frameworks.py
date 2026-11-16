"""Framework adapters: numpy in, framework tensor out, and back.

Tests are written once against numpy. An adapter turns numpy arrays into the tensors
of one framework before a call and turns results back into numpy afterwards, so the
same test judges a torch, JAX, MLX, or plain-Python solution.

Conversion rules (``tensor``):

* floating arrays become float32 (the house dtype; pass ``dtype=`` to override);
* integer arrays keep their dtype: an int8 array arrives as an int8 tensor, uint8 as
  uint8, int32 as int32. The one exception is 64-bit integers, numpy's default (what
  ``np.array([1, 2])`` and ``ctx.randint`` produce): they become the framework's
  default integer type, which is int64 in torch and int32 in JAX and MLX (uint64
  likewise stays uint64 in torch and becomes uint32 in JAX and MLX). Bool stays bool;
* the ``python`` adapter turns arrays into nested lists and numpy scalars into
  Python scalars.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from learnai.judge.errors import CheckFailure


class FrameworkUnavailable(Exception):
    """The framework is not installed in this environment."""


class Framework:
    """Base adapter. Subclasses fill in the four primitive operations."""

    name = "abstract"
    tensor_type_name = "tensor"
    version = ""
    device = "cpu"
    #: The array namespace: ``torch``, ``jax.numpy``, ``mlx.core``; None for python.
    xp: Any = None

    # -- primitives ------------------------------------------------------------

    def is_tensor(self, value: Any) -> bool:
        raise NotImplementedError

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        """Convert a numpy array (or anything ``np.asarray`` accepts) to a tensor."""
        raise NotImplementedError

    def numpy(self, value: Any) -> np.ndarray:
        """Convert one tensor of this framework to a numpy array."""
        raise NotImplementedError

    def sync(self, *values: Any) -> None:
        """Block until every tensor in ``values`` has actually been computed.

        Required before stopping a timer: MLX is lazy, JAX dispatches asynchronously,
        and torch on a GPU queues work.
        """

    # -- recursive conversion --------------------------------------------------

    def to_framework(self, obj: Any, *, fresh_kit: bool = False) -> Any:
        """Convert every numpy array inside ``obj`` (lists, tuples, dicts) to a tensor.

        Other values pass through untouched, except kit objects (anything with a
        ``with_framework`` method), which are rebound to this framework so they hand
        out the right tensor type. ``fresh_kit=True`` rebinds them even when they are
        already bound to it, which gives reference calls their own instrumentation
        counters instead of polluting the ones the solution is judged by.
        """
        if isinstance(obj, np.ndarray):
            return self.tensor(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, tuple):
            return tuple(self.to_framework(item, fresh_kit=fresh_kit) for item in obj)
        if isinstance(obj, list):
            return [self.to_framework(item, fresh_kit=fresh_kit) for item in obj]
        if isinstance(obj, dict):
            return {k: self.to_framework(v, fresh_kit=fresh_kit) for k, v in obj.items()}
        rebind = getattr(obj, "with_framework", None)
        if callable(rebind) and (fresh_kit or getattr(obj, "fw", self) is not self):
            return rebind(self)
        return obj

    def to_numpy(self, obj: Any, *, what: str = "the solution") -> Any:
        """Convert every tensor inside ``obj`` to numpy, complaining about foreign types.

        A solution that returns a numpy array or another framework's tensor where this
        framework's tensor is expected fails with a message naming both types.
        """
        if self.is_tensor(obj):
            return self.numpy(obj)
        if isinstance(obj, tuple):
            return tuple(self.to_numpy(item, what=what) for item in obj)
        if isinstance(obj, list):
            return [self.to_numpy(item, what=what) for item in obj]
        if isinstance(obj, dict):
            return {key: self.to_numpy(item, what=what) for key, item in obj.items()}
        if self.name != "python" and _looks_like_foreign_tensor(obj):
            got = f"{type(obj).__module__}.{type(obj).__qualname__}"
            raise CheckFailure(
                f"{what} returned a {got}, expected a {self.tensor_type_name}",
                f"Return {self.tensor_type_name} values; do not convert to another array type.\n",
            )
        return obj

    def __repr__(self) -> str:
        return f"<Framework {self.name}>"


def _looks_like_foreign_tensor(obj: Any) -> bool:
    if isinstance(obj, (np.ndarray, np.generic)):
        return True
    module = type(obj).__module__ or ""
    return module.split(".")[0] in ("torch", "jax", "jaxlib", "mlx")


def _prepare(value: Any, dtype: Any, *, wide_ints: bool) -> np.ndarray:
    """The numpy array to hand to the framework, in the dtype the tensor will have.

    ``wide_ints`` says whether the framework's default integer is 64 bits wide (torch)
    or 32 (JAX without x64, MLX). Narrower integers are never widened: a problem about
    int8 codes must be able to hand the solution int8 codes.
    """
    array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype)
    if array.dtype.kind == "f" and array.dtype != np.float32:
        return array.astype(np.float32)
    if array.dtype.kind in "iu" and array.dtype.itemsize == 8 and not wide_ints:
        return array.astype(np.int32 if array.dtype.kind == "i" else np.uint32)
    return array


class TorchFramework(Framework):
    name = "torch"
    tensor_type_name = "torch.Tensor"

    def __init__(self) -> None:
        import torch

        self.xp = torch
        self.version = torch.__version__

    def is_tensor(self, value: Any) -> bool:
        return isinstance(value, self.xp.Tensor)

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        array = _prepare(value, dtype, wide_ints=True)
        # torch.tensor copies: a solution that works in place must not be able to
        # change the numpy array the test is about to hand to the oracle.
        return self.xp.tensor(array)

    def numpy(self, value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()

    def sync(self, *values: Any) -> None:
        kinds = {t.device.type for t in _leaves(values) if self.is_tensor(t)}
        if "mps" in kinds:
            self.xp.mps.synchronize()
        if "cuda" in kinds:
            self.xp.cuda.synchronize()


class JaxFramework(Framework):
    name = "jax"
    tensor_type_name = "jax.Array"

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self.xp = jnp
        self.version = jax.__version__

    def is_tensor(self, value: Any) -> bool:
        return isinstance(value, self._jax.Array)

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        array = _prepare(value, dtype, wide_ints=False)
        return self.xp.asarray(array)

    def numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def sync(self, *values: Any) -> None:
        self._jax.block_until_ready(values)


class MlxFramework(Framework):
    name = "mlx"
    tensor_type_name = "mlx.core.array"

    def __init__(self) -> None:
        import mlx.core as mx

        self.xp = mx
        try:
            from importlib.metadata import version

            self.version = version("mlx")
        except Exception:  # noqa: BLE001
            self.version = getattr(mx, "__version__", "")
        self.device = "gpu" if "gpu" in str(mx.default_device()).lower() else "cpu"

    def is_tensor(self, value: Any) -> bool:
        return isinstance(value, self.xp.array)

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        array = _prepare(value, dtype, wide_ints=False)
        return self.xp.array(array)

    def numpy(self, value: Any) -> np.ndarray:
        return np.array(value)

    def sync(self, *values: Any) -> None:
        """Force every array reachable from ``values``, then wait for the GPU.

        ``mx.eval`` computes the arrays (MLX builds a graph and does no work until
        asked); ``mx.synchronize`` then waits for anything still in flight, which is
        what a solution leaves behind when it schedules work with ``mx.async_eval``
        and returns. Arrays are found inside tuples, lists, and dicts, in the
        parameters of an ``mlx.nn.Module``, and in the attributes of plain objects (a
        cache class): an unevaluated cache must not make a solution look fast.
        """
        tensors = [t for t in _leaves(values, deep=True) if self.is_tensor(t)]
        if tensors:
            self.xp.eval(*tensors)
        self.xp.synchronize()

    def memory(self) -> Any:
        """A context manager measuring MLX memory (see ``learnai.kit.mlx_memory``)."""
        from learnai.kit.mlxtools import mlx_memory

        return mlx_memory()


class PythonFramework(Framework):
    """Framework-free problems: lists and scalars instead of tensors."""

    name = "python"
    tensor_type_name = "list"

    def __init__(self) -> None:
        import platform

        self.version = platform.python_version()

    def is_tensor(self, value: Any) -> bool:
        return False

    def tensor(self, value: Any, dtype: Any = None) -> Any:
        array = np.asarray(value) if dtype is None else np.asarray(value).astype(dtype)
        return array.tolist()

    def numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)


def _leaves(obj: Any, *, deep: bool = False, _depth: int = 0, _seen: set[int] | None = None):
    """Yield the non-container values inside nested tuples, lists, and dicts.

    ``deep=True`` also looks inside the instance attributes of ordinary objects (a
    solution's cache class, an ``mlx.nn.Module``, which is a dict), a few levels down
    and never twice into the same object.
    """
    if isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _leaves(item, deep=deep, _depth=_depth, _seen=_seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _leaves(item, deep=deep, _depth=_depth, _seen=_seen)
    else:
        yield obj
        attributes = getattr(obj, "__dict__", None) if deep and _depth < 4 else None
        plain = not isinstance(obj, (type, ModuleType)) and not callable(obj)
        if isinstance(attributes, dict) and attributes and plain:
            seen = set() if _seen is None else _seen
            if id(obj) not in seen:
                seen.add(id(obj))
                yield from _leaves(attributes, deep=True, _depth=_depth + 1, _seen=seen)


#: Upper bound on the on-disk JAX compilation cache. Beyond it the cache is emptied.
JAX_CACHE_MAX_BYTES = 512 * 1024 * 1024


def _tidy_jax_cache(path: str) -> None:
    """Remove what a killed process may have left, and keep the directory bounded.

    JAX writes an entry with a plain open-and-write. A worker killed between the two
    (the hard time limit) leaves an empty file that every later run would fail to
    read, so empty entries are removed. JAX's own size bound rescans the directory on
    every write, which doubles the time of a first run; one scan per process and
    "empty it when it is too large" costs nothing and is enough for a cache.
    """
    import os

    total = 0
    entries = []
    with os.scandir(path) as listing:
        for entry in listing:
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            if size == 0:
                _unlink(entry.path)
                continue
            total += size
            entries.append(entry.path)
    if total > JAX_CACHE_MAX_BYTES:
        for entry_path in entries:
            _unlink(entry_path)


def _unlink(path: str) -> None:
    import os

    try:
        os.unlink(path)
    except OSError:  # another run removed it first
        pass


def use_jax_compilation_cache(directory: Any) -> bool:
    """Keep JAX's compiled kernels on disk, so only the first run pays for them.

    JAX compiles every operation once per input shape and dtype, in every process: a
    real-model reference spends seconds of each run compiling the same few hundred
    kernels again. JAX's persistent compilation cache stores the compiled executables,
    keyed by the computation, the JAX and XLA versions, and the compile options; a
    later run loads them instead. Results are unchanged (the same executables run) and
    nothing a case measures moves: speed cases compile during warm-up either way.

    ``directory`` is ``<root>/.learnai/cache/jax``. ``LEARNAI_JAX_CACHE`` overrides
    it with another directory, or disables the cache with ``0`` or ``off``; a
    ``JAX_COMPILATION_CACHE_DIR`` set by the user is left alone. Returns whether the
    cache was configured. Never raises: a run without the cache is merely slower.

    An entry that cannot be read (two runs racing on one file) is recompiled by JAX,
    which warns about it; the warning is silenced because the learner can do nothing
    with it and the result is the same.
    """
    import os
    import warnings

    choice = os.environ.get("LEARNAI_JAX_CACHE", "")
    if choice.lower() in ("0", "off", "false", "no"):
        return False
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return False
    try:
        import jax

        path = os.path.abspath(choice or str(directory))
        os.makedirs(path, exist_ok=True)
        _tidy_jax_cache(path)
        jax.config.update("jax_compilation_cache_dir", path)
        # The defaults keep only kernels that took a second to compile; the cost here
        # is a thousand small ones.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        warnings.filterwarnings(
            "ignore", message="Error (reading|writing) persistent compilation cache entry"
        )
    except Exception:  # noqa: BLE001 - read-only disk, an older JAX: run without it
        return False
    return True


_CLASSES = {
    "torch": TorchFramework,
    "jax": JaxFramework,
    "mlx": MlxFramework,
    "python": PythonFramework,
}
_instances: dict[str, Framework] = {}


def get(name: str) -> Framework:
    """Return the (cached) adapter for ``name``: torch, jax, mlx, or python."""
    if name not in _CLASSES:
        raise ValueError(f"unknown framework {name!r}")
    if name not in _instances:
        try:
            _instances[name] = _CLASSES[name]()
        except ImportError as exc:
            raise FrameworkUnavailable(
                f"framework {name!r} is not installed in this environment ({exc}); "
                f"enable it with `uv sync --extra {name}` plus the other extras you use"
            ) from exc
    return _instances[name]
