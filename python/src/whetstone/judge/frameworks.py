"""Framework adapters: numpy in, framework tensor out, and back.

Tests are written once against numpy. An adapter turns numpy arrays into the tensors
of one framework before a call and turns results back into numpy afterwards, so the
same test judges a torch, JAX, MLX, or plain-Python solution.

Conversion rules (``tensor``):

* floating arrays become float32 (the house dtype; pass ``dtype=`` to override);
* integer arrays become the framework's default integer type: int64 in torch,
  int32 in JAX and MLX; bool stays bool;
* the ``python`` adapter turns arrays into nested lists and numpy scalars into
  Python scalars.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from whetstone.judge.errors import CheckFailure


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


def _prepare(value: Any, dtype: Any) -> np.ndarray:
    array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype)
    if array.dtype.kind == "f" and array.dtype != np.float32:
        return array.astype(np.float32)
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
        array = _prepare(value, dtype)
        if array.dtype.kind in "iu" and dtype is None:
            array = array.astype(np.int64)
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
        array = _prepare(value, dtype)
        if array.dtype.kind in "iu" and dtype is None:
            array = array.astype(np.int32)
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
        array = _prepare(value, dtype)
        if array.dtype.kind in "iu" and dtype is None:
            array = array.astype(np.int32)
        return self.xp.array(array)

    def numpy(self, value: Any) -> np.ndarray:
        return np.array(value)

    def sync(self, *values: Any) -> None:
        tensors = [t for t in _leaves(values) if self.is_tensor(t)]
        if tensors:
            self.xp.eval(*tensors)


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


def _leaves(obj: Any):
    """Yield the non-container values inside nested tuples, lists, and dicts."""
    if isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _leaves(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _leaves(item)
    else:
        yield obj


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
