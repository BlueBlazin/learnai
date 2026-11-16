"""A numpy stand-in for the part of ``triton.language`` the CUDA track uses.

    from learnai.kit import tritonshim as tl

    @tl.jit
    def scale_kernel(x_ptr, out_ptr, n, factor, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * factor, mask=mask)

    tl.launch(scale_kernel, (tl.cdiv(n, 64),), x, out, n, 2.0, BLOCK=64)
    # the same thing, spelled as Triton spells it:
    scale_kernel[(tl.cdiv(n, 64),)](x, out, n, 2.0, BLOCK=64)

Triton (https://triton-lang.org) compiles a Python function into a GPU kernel. The
kernel is launched over a *grid*; every point of the grid is a *program* that runs the
same code, finds out which one it is with ``tl.program_id``, and reads and writes
memory through *pointers*: ``tl.load(ptr + offsets, mask=...)`` reads a block of
elements at once. Triton runs only on Linux with an NVIDIA or AMD GPU. This module
runs the same kernel source in numpy, one program after another, so that a kernel can
be written and judged on a CPU.

What it models, following the Triton documentation:

- ``program_id(axis)``, ``num_programs(axis)``; a 1-D to 3-D grid.
- Pointers: a tensor argument arrives as a pointer to its first element; ``ptr +
  offsets`` (in elements) is a block of pointers; ``load(pointer, mask=None,
  other=None)`` and ``store(pointer, value, mask=None)`` with Triton's keyword-only
  options accepted and ignored.
- ``arange``, ``zeros``, ``zeros_like``, ``full``, ``where``, ``cast`` (and
  ``x.to(dtype)``), ``sum``, ``max``, ``min``, ``maximum``, ``minimum``, ``clamp``,
  ``abs``, ``sqrt``, ``rsqrt``, ``exp``, ``log``, ``sigmoid``; ``+ - * / // % & | ^
  << >> ~`` and comparisons; ``x[:, None]`` and ``x[None, :]`` to add an axis.
- Triton's type rules: promotion by kind, then width, then unsigned; a Python number
  next to a tensor takes the tensor's dtype unless its kind is higher; ``/`` and
  ``%`` with a float operand compute in at least float32 (float16 is promoted);
  ``/`` on integers gives float32; integer ``//`` and ``%`` round toward zero, as in
  C; ``sum`` widens small integers to 32 bits; scalar arguments arrive as int32
  (int64 when they do not fit), float32, or int1.
- ``BLOCK: tl.constexpr`` parameters stay Python numbers and can be used as shapes;
  a shape is a tuple or, as the layer-norm tutorial writes it, a list
  (``tl.zeros([BLOCK], dtype=tl.float32)``); shapes must be powers of two.
- ``tl.arange(start, end)``: the page says ``start`` and ``end`` must be powers of
  two, and every tutorial writes ``tl.arange(0, BLOCK)``. The shim accepts what
  satisfies both that wording (with 0 allowed) and a power-of-two length:
  ``(0, 2**m)`` and ``(2**k, 2**(k + 1))``.
- ``<<`` and ``>>``: the documentation only says the operators exist. A shift count
  outside ``[0, bits)`` raises, and so does ``>>`` on a signed block that holds a
  negative value, because whether the sign bit is copied is not documented; shift
  unsigned values (zeros come in) or mask the bits you need.

Where it is deliberately strict, because a GPU would not tell you:

- **An unmasked access outside the tensor raises** ``OutOfBounds``. On a GPU it reads
  or overwrites whatever is next to the tensor.
- **A masked-off element of a load without ``other=`` is poison** (NaN for floats, the
  largest value for integers). Triton documents that value as undefined.
- **Two programs storing to the same element raise** ``DataRace``: programs run
  concurrently on a GPU, so the result would depend on timing. Programs run here in
  a shuffled, fixed order, so a kernel that depends on the order of programs fails.
- **Only kernel code**: ``jit`` checks the source and rejects what Triton cannot
  compile from the subset used here (calling numpy, lists, comprehensions, ``with``,
  ``try``, imports).

What it does not model: speed, ``tl.dot``, atomics, tensor descriptors, autotuning,
float semantics on the GPU beyond IEEE float16/32/64, and bfloat16 (numpy has none).

**This shim has not been run against real Triton**: the machine it was written on
cannot install Triton. Signatures and semantics were taken from the documentation at
triton-lang.org (``triton.language``, ``tensor``, ``load``, ``store``, ``sum``,
``max``, ``cast``, ``arange``, ``zeros``, ``where``, ``full``, ``program_id``, "Triton
Semantics", and the vector-add and layer-norm tutorials). ``launch`` hands a kernel to real Triton when its tensors are CUDA tensors
(``compile_with_triton``); that path is untested until someone runs it on a CUDA
machine.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import itertools
import math
import sys
import textwrap
import types
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from learnai.kit.counters import Counters

MAX_TENSOR_NUMEL = 1048576  # TRITON_MAX_TENSOR_NUMEL


class TritonShimError(Exception):
    """The kernel did something Triton does not allow, or that a GPU would not survive."""


class OutOfBounds(TritonShimError):
    """An unmasked load or store outside the tensor behind the pointer."""


class DataRace(TritonShimError):
    """Two programs of one launch stored to the same element."""


# --------------------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------------------


class dtype:
    """A Triton element type: ``tl.float32``, ``tl.int32``, ``tl.uint8``, ``tl.int1``."""

    def __init__(self, name: str, np_dtype: Any) -> None:
        self.name = name
        self.np = np.dtype(np_dtype)
        self.kind = {"b": 0, "i": 1, "u": 1, "f": 2}[self.np.kind]  # bool < int < float
        self.bits = 1 if self.np.kind == "b" else self.np.itemsize * 8
        self.signed = self.np.kind == "i"

    def is_floating(self) -> bool:
        return self.kind == 2

    def is_int(self) -> bool:
        return self.kind == 1

    def is_bool(self) -> bool:
        return self.kind == 0

    def __repr__(self) -> str:
        return self.name


int1 = dtype("int1", np.bool_)
int8, int16, int32, int64 = (dtype(f"int{b}", f"int{b}") for b in (8, 16, 32, 64))
uint8, uint16, uint32, uint64 = (dtype(f"uint{b}", f"uint{b}") for b in (8, 16, 32, 64))
float16, float32, float64 = (dtype(f"float{b}", f"float{b}") for b in (16, 32, 64))
bool = int1
_DTYPES = {
    d.np: d
    for d in (int1, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
              float16, float32, float64)
}  # fmt: skip
_pybool = builtins.bool


def _dtype_of(np_dtype: Any) -> dtype:
    try:
        return _DTYPES[np.dtype(np_dtype)]
    except KeyError:
        raise TritonShimError(f"the shim has no Triton dtype for {np_dtype}") from None


class constexpr:
    """Annotation for a compile-time constant: ``BLOCK: tl.constexpr``.

    A constexpr parameter must be passed by keyword at launch, stays a Python number
    inside the kernel, and may be used where Triton needs a constant (a shape, the
    bounds of ``arange``).
    """

    def __init__(self, value: Any) -> None:
        self.value = value


# --------------------------------------------------------------------------------------
# tensors
# --------------------------------------------------------------------------------------

_SCALARS = (int, float, _pybool, np.integer, np.floating, np.bool_)


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _const_int(value: Any, what: str) -> int:
    if isinstance(value, tensor):
        raise TritonShimError(
            f"{what} must be a compile-time constant (a literal or a `tl.constexpr` "
            "parameter), not a value computed at run time"
        )
    if isinstance(value, (_pybool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TritonShimError(f"{what} must be an integer constant, got {value!r}")
    return int(value)


def _scalar_dtype(value: Any) -> dtype:
    """The dtype a Python number takes when its kind is higher than the tensor's."""
    if isinstance(value, (_pybool, np.bool_)):
        return int1
    if isinstance(value, (int, np.integer)):
        v = int(value)
        for candidate in (int32, uint32, int64, uint64):
            info = np.iinfo(candidate.np)
            if info.min <= v <= info.max:
                return candidate
        raise TritonShimError(f"integer constant {v} does not fit in 64 bits")
    v = float(value)
    fits = not math.isfinite(v) or math.fabs(v) <= float(np.finfo(np.float32).max)
    return float32 if fits else float64


def _promote(a: dtype, b: dtype) -> dtype:
    """Tensor with tensor: kind, then width, then unsigned (Triton Semantics)."""
    if a.kind != b.kind:
        return a if a.kind > b.kind else b
    if a.bits != b.bits:
        return a if a.bits > b.bits else b
    if a.signed != b.signed:
        return b if a.signed else a
    return a


class tensor:
    """A block of values (or one scalar) inside a kernel. Created by ``tl`` functions."""

    __slots__ = ("dtype", "value")
    __array_ufunc__ = None  # numpy scalars defer to the reflected operators below

    def __init__(self, value: Any, ty: dtype) -> None:
        self.value = np.asarray(value, dtype=ty.np)
        self.dtype = ty

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def type(self) -> dtype:
        return self.dtype

    @property
    def T(self) -> tensor:
        if self.value.ndim != 2:
            raise TritonShimError(".T transposes a 2-D tensor")
        return tensor(self.value.T, self.dtype)

    # -- conversions -------------------------------------------------------------------

    def to(
        self, ty: dtype, fp_downcast_rounding: str | None = None, bitcast: Any = False
    ) -> tensor:
        return cast(self, ty, fp_downcast_rounding, bitcast)

    cast = to

    def __index__(self) -> int:
        # `range(0, n, BLOCK)` with a run-time scalar `n`: Triton compiles the loop.
        if self.value.ndim or not self.dtype.is_int():
            raise TritonShimError("only an integer scalar can be a loop bound or an index")
        return int(self.value)

    def __bool__(self) -> _pybool:
        if self.value.ndim:
            raise TritonShimError(
                "the truth value of a block is ambiguous: `if` and `while` need a scalar "
                "(use tl.where for elementwise choices)"
            )
        return _pybool(self.value)

    def __len__(self) -> int:
        raise TritonShimError("len() is not available inside a Triton kernel")

    def __iter__(self) -> Any:
        raise TritonShimError("a block cannot be iterated inside a Triton kernel")

    def __getitem__(self, index: Any) -> tensor:
        parts = index if isinstance(index, tuple) else (index,)
        full = slice(None)
        if not all(p is None or (isinstance(p, slice) and p == full) for p in parts):
            raise TritonShimError(
                "the shim supports only `x[:, None]` / `x[None, :]` (adding an axis); "
                "Triton has no general indexing of blocks"
            )
        return tensor(self.value[index], self.dtype)

    def __repr__(self) -> str:
        return f"tensor(shape={self.shape}, dtype={self.dtype})"

    # -- arithmetic --------------------------------------------------------------------
    # The binary operators (+ - * / // % & | ^ << >> and the comparisons) are added
    # below the class by `_install_operators`; all of them go through `_binary`.

    def __add__(self, other: Any) -> Any:
        if isinstance(other, pointer):
            return other + self  # offsets + pointer
        return _binary("add", self, other)

    __hash__ = None  # type: ignore[assignment]  # `==` compares elementwise

    def __neg__(self) -> tensor:
        if self.dtype.is_bool():
            raise TritonShimError("cannot negate a boolean block")
        with np.errstate(all="ignore"):
            return tensor(-self.value, self.dtype)

    def __invert__(self) -> tensor:
        if self.dtype.is_floating():
            raise TritonShimError("`~` needs an integer or boolean block")
        return tensor(~self.value, self.dtype)

    def __abs__(self) -> tensor:
        return abs(self)

    # -- methods Triton also offers on tensors -----------------------------------------

    def sum(self, axis: Any = None, keep_dims: Any = False, dtype: Any = None) -> tensor:
        return sum(self, axis, keep_dims, dtype)

    def max(self, axis: Any = None, **options: Any) -> tensor:
        return max(self, axis, **options)

    def min(self, axis: Any = None, **options: Any) -> tensor:
        return min(self, axis, **options)

    def store(self, value: Any, mask: Any = None, **options: Any) -> None:
        raise TritonShimError("only a pointer can be stored to")


_COMPARE = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
            "eq": np.equal, "ne": np.not_equal}  # fmt: skip
_ARITH = {"add": np.add, "sub": np.subtract, "mul": np.multiply}
_BITS = {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
         "lshift": np.left_shift, "rshift": np.right_shift}  # fmt: skip


def _install_operators() -> None:
    def forward(op: str) -> Any:
        return lambda self, other: _binary(op, self, other)

    def reflected(op: str) -> Any:
        return lambda self, other: _binary(op, other, self)

    names = {"add": "add", "sub": "sub", "mul": "mul", "truediv": "truediv",
             "floordiv": "floordiv", "mod": "mod", "and": "and", "or": "or", "xor": "xor",
             "lshift": "lshift", "rshift": "rshift"}  # fmt: skip
    for dunder, op in names.items():
        if dunder != "add":
            setattr(tensor, f"__{dunder}__", forward(op))
        setattr(tensor, f"__r{dunder}__", reflected(op))
    for op in _COMPARE:
        setattr(tensor, f"__{op}__", forward(op))


_install_operators()


def _operands(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray, dtype]:
    """Both operands as arrays of the common dtype, by Triton's promotion rules."""
    for operand in (a, b):
        if isinstance(operand, pointer):
            raise TritonShimError("pointers support only `pointer + offsets`")
        if not isinstance(operand, (tensor, *_SCALARS)):
            raise TritonShimError(f"a {type(operand).__name__} cannot be used in kernel arithmetic")
    if isinstance(a, tensor) and isinstance(b, tensor):
        ty = _promote(a.dtype, b.dtype)
    else:
        block, scalar = (a, b) if isinstance(a, tensor) else (b, a)
        kind = _scalar_dtype(scalar)
        # A scalar of the same or a lower kind does not take part in promotion.
        ty = block.dtype if kind.kind <= block.dtype.kind else _promote(block.dtype, kind)
    return _as_array(a, ty), _as_array(b, ty), ty


def _as_array(x: Any, ty: dtype) -> np.ndarray:
    value = x.value if isinstance(x, tensor) else np.asarray(x)
    with np.errstate(all="ignore"):
        return value.astype(ty.np, copy=False)


def _binary(op: str, a: Any, b: Any) -> tensor:
    x, y, ty = _operands(a, b)
    try:
        np.broadcast_shapes(x.shape, y.shape)
    except ValueError:
        raise TritonShimError(f"cannot broadcast shapes {x.shape} and {y.shape}") from None
    with np.errstate(all="ignore"):
        if op in _COMPARE:
            return tensor(_COMPARE[op](x, y), int1)
        if op in _ARITH:
            if ty.is_bool():
                x, y, ty = x.astype(np.int32), y.astype(np.int32), int32
            return tensor(_ARITH[op](x, y), ty)
        if op in _BITS:
            if ty.is_floating():
                raise TritonShimError(f"`{op}` needs integer or boolean operands, got {ty}")
            if op in ("lshift", "rshift"):
                _check_shift(op, a, y, ty)
            return tensor(_BITS[op](x, y), ty)
        if op in ("truediv", "mod") and (ty.is_floating() or op == "truediv"):
            # Division and modulo do not exist for floats narrower than float32, so a
            # float operand of any narrower width is promoted to float32 (Triton
            # Semantics); integers divide with `/` as float32.
            if not ty.is_floating() or ty.bits < 32:
                x, y, ty = x.astype(np.float32), y.astype(np.float32), float32
            return tensor(np.divide(x, y) if op == "truediv" else np.fmod(x, y), ty)
        if ty.is_bool():
            raise TritonShimError(f"`{op}` is not defined for boolean blocks")
        if op == "floordiv":
            if ty.is_floating():
                raise TritonShimError("`//` needs integer operands in a Triton kernel")
            _no_zero_divisor(y)
            # C semantics: the quotient is rounded toward zero, not toward minus infinity.
            quotient = np.abs(x.astype(np.int64)) // np.abs(y.astype(np.int64))
            sign = np.where((x < 0) != (y < 0), -1, 1) if ty.signed else 1
            return tensor((quotient * sign).astype(ty.np), ty)
        if op == "mod":
            _no_zero_divisor(y)
            return tensor(np.fmod(x, y), ty)  # the sign of the dividend, as in C
    raise AssertionError(op)


def _check_shift(op: str, left: Any, y: np.ndarray, ty: dtype) -> None:
    """Refuse the shifts whose result Triton's documentation does not define.

    The documentation says that ``<<`` and ``>>`` exist and nothing more. Two things
    are therefore not relied on: a count outside ``[0, bits)`` (undefined in C and in
    LLVM, which Triton compiles to), and ``>>`` on a negative signed value, where an
    arithmetic shift copies the sign bit and a logical shift brings in zeros.
    """
    symbol = "<<" if op == "lshift" else ">>"
    if ty.is_bool():
        raise TritonShimError(f"`{symbol}` is not defined for boolean blocks")
    if np.any(y < 0) or np.any(y.astype(np.uint64) >= ty.bits):
        raise TritonShimError(
            f"`{symbol}`: a shift count outside [0, {ty.bits}) is undefined for {ty}"
        )
    # Negative in its own type, whatever the count's type promotes the pair to.
    shifted = left.value if isinstance(left, tensor) else np.asarray(left)
    if op == "rshift" and shifted.dtype.kind == "i" and np.any(shifted < 0):
        raise TritonShimError(
            "`>>` on a negative signed value: Triton's documentation does not say whether "
            "a signed right shift copies the sign bit or brings in zeros, so the shim "
            "refuses it. Keep the value unsigned (tl.uint32: zeros come in), or mask "
            "the bits you need after shifting a non-negative value."
        )


def _no_zero_divisor(y: np.ndarray) -> None:
    if np.any(y == 0):
        raise TritonShimError("integer division by zero (undefined on a GPU)")


def _as_tensor(x: Any) -> tensor:
    if isinstance(x, tensor):
        return x
    if isinstance(x, _SCALARS):
        return tensor(x, _scalar_dtype(x))
    raise TritonShimError(f"expected a block or a number, got {type(x).__name__}")


# --------------------------------------------------------------------------------------
# pointers, load, store
# --------------------------------------------------------------------------------------


class pointer:
    """A pointer, or a block of pointers, into one tensor's memory.

    A tensor passed to a kernel arrives as a pointer to its first element.
    ``ptr + offsets`` moves it by whole elements, as typed pointers do in C; the
    result has the shape of ``offsets``.
    """

    __slots__ = ("buffer", "element_ty", "name", "offsets")
    __array_ufunc__ = None

    def __init__(self, buffer: np.ndarray, offsets: np.ndarray, name: str) -> None:
        self.buffer = buffer  # 1-D view of the tensor's memory
        self.offsets = offsets  # int64, a scalar or a block
        self.element_ty = _dtype_of(buffer.dtype)
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.offsets.shape

    def __add__(self, other: Any) -> pointer:
        if isinstance(other, tensor):
            if not other.dtype.is_int():
                raise TritonShimError("a pointer can be offset only by integers")
            step = other.value.astype(np.int64)
        elif isinstance(other, (int, np.integer)) and not isinstance(other, _pybool):
            step = np.int64(other)
        else:
            raise TritonShimError("a pointer can be offset only by integers")
        return pointer(self.buffer, self.offsets + step, self.name)

    __radd__ = __add__

    def __sub__(self, other: Any) -> pointer:
        return self + (-other)

    def __repr__(self) -> str:
        return f"pointer({self.name}, {self.element_ty}, shape={self.shape})"


class _Launch:
    """State of the launch that is running: which program, and what it touched."""

    def __init__(self, grid: tuple[int, ...]) -> None:
        self.grid = grid
        self.program: tuple[int, ...] = (0,) * len(grid)
        self.counters = Counters(
            "programs", "loads", "stores", "elements_loaded", "elements_stored"
        )
        #: id(buffer) -> for every element, the index of the program that stored to it.
        self.writers: dict[int, np.ndarray] = {}
        self.index = 0


_launch: _Launch | None = None


def _current() -> _Launch:
    if _launch is None:
        raise TritonShimError("this can only be called inside a kernel that is being launched")
    return _launch


def _mask_for(mask: Any, shape: tuple[int, ...], what: str) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=np.bool_)
    if isinstance(mask, (_pybool, np.bool_)):
        return np.full(shape, _pybool(mask))
    if not isinstance(mask, tensor) or not mask.dtype.is_bool():
        raise TritonShimError(f"{what}: mask must be a boolean block (tl.int1)")
    try:
        return np.broadcast_to(mask.value, shape)
    except ValueError:
        raise TritonShimError(
            f"{what}: mask of shape {mask.shape} does not broadcast to the pointer's {shape}"
        ) from None


def _check_bounds(ptr: pointer, active: np.ndarray, what: str) -> None:
    offsets = np.broadcast_to(ptr.offsets, active.shape)
    bad = active & ((offsets < 0) | (offsets >= ptr.buffer.size))
    if bad.any():
        first = int(offsets[bad].flat[0])
        program = _current().program
        raise OutOfBounds(
            f"{what} out of bounds: program {program} accessed element {first} of "
            f"`{ptr.name}`, which has {ptr.buffer.size} elements "
            f"({int(bad.sum())} of {bad.size} elements of the block are outside). "
            "On a GPU this reads or overwrites other memory; guard the access with mask=."
        )


def load(
    pointer: pointer,
    mask: Any = None,
    other: Any = None,
    *,
    cache_modifier: str = "",
    eviction_policy: str = "",
    volatile: Any = False,
) -> tensor:
    """Read the elements ``pointer`` points to.

    Where ``mask`` is false nothing is read and the result is ``other`` (cast to the
    element type). Without ``other`` the masked-off values are undefined in Triton;
    here they are poison (NaN, or the largest integer) so that using them shows.
    """
    ptr = _need_pointer(pointer, "tl.load")
    active = _mask_for(mask, ptr.shape, "tl.load")
    _check_bounds(ptr, active, "tl.load")
    ty = ptr.element_ty
    if other is None:
        fill = np.nan if ty.is_floating() else True if ty.is_bool() else np.iinfo(ty.np).max
        out = np.full(ptr.shape, fill, dtype=ty.np)
    else:
        if not isinstance(other, (tensor, *_SCALARS)):
            raise TritonShimError("tl.load: other must be a number or a block")
        out = np.array(np.broadcast_to(_as_array(other, ty), ptr.shape))
    offsets = np.broadcast_to(ptr.offsets, ptr.shape)
    out[active] = ptr.buffer[offsets[active]]
    launch = _current()
    launch.counters.add("loads")
    launch.counters.add("elements_loaded", int(active.sum()))
    return tensor(out, ty)


def store(
    pointer: pointer,
    value: Any,
    mask: Any = None,
    *,
    cache_modifier: str = "",
    eviction_policy: str = "",
) -> None:
    """Write ``value`` (broadcast and cast to the element type) where ``mask`` is true."""
    ptr = _need_pointer(pointer, "tl.store")
    active = _mask_for(mask, ptr.shape, "tl.store")
    _check_bounds(ptr, active, "tl.store")
    if not isinstance(value, (tensor, *_SCALARS)):
        raise TritonShimError("tl.store: value must be a number or a block")
    try:
        values = np.broadcast_to(_as_array(value, ptr.element_ty), ptr.shape)
    except ValueError:
        shape = getattr(value, "shape", ())
        raise TritonShimError(
            f"tl.store: value of shape {shape} does not broadcast to the pointer's {ptr.shape}"
        ) from None
    offsets = np.broadcast_to(ptr.offsets, ptr.shape)[active]
    launch = _current()
    writers = launch.writers.setdefault(
        id(ptr.buffer), np.full(ptr.buffer.size, -1, dtype=np.int64)
    )
    clash = (writers[offsets] != -1) & (writers[offsets] != launch.index)
    if clash.any():
        element = int(offsets[clash][0])
        raise DataRace(
            f"tl.store: program {launch.program} stored to element {element} of "
            f"`{ptr.name}`, which another program of this launch had already written. "
            "Programs run concurrently on a GPU, so the result would depend on timing."
        )
    writers[offsets] = launch.index
    if not ptr.buffer.flags.writeable:
        raise TritonShimError(f"tl.store: `{ptr.name}` is read-only")
    ptr.buffer[offsets] = values[active]
    launch.counters.add("stores")
    launch.counters.add("elements_stored", int(active.sum()))


def _need_pointer(value: Any, what: str) -> pointer:
    if not isinstance(value, pointer):
        raise TritonShimError(
            f"{what} needs a pointer (a tensor argument of the kernel, plus offsets), "
            f"got {type(value).__name__}"
        )
    return value


# --------------------------------------------------------------------------------------
# the language
# --------------------------------------------------------------------------------------


def program_id(axis: int) -> tensor:
    """Which program this is along ``axis`` (0, 1, or 2) of the launch grid: an int32."""
    return tensor(_grid_axis(axis, _current().program, "program_id"), int32)


def num_programs(axis: int) -> tensor:
    """The size of the launch grid along ``axis``: an int32."""
    return tensor(_grid_axis(axis, _current().grid, "num_programs"), int32)


def _grid_axis(axis: Any, values: tuple[int, ...], what: str) -> int:
    if axis not in (0, 1, 2):
        raise TritonShimError(f"{what}: axis must be 0, 1 or 2")
    return values[axis] if axis < len(values) else (0 if what == "program_id" else 1)


def _shape(shape: Any, what: str) -> tuple[int, ...]:
    dims = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    dims = tuple(_const_int(d, f"{what}: a shape") for d in dims)
    if not all(_is_power_of_two(d) for d in dims):
        raise TritonShimError(f"{what}: every dimension must be a power of two, got {dims}")
    if math.prod(dims) > MAX_TENSOR_NUMEL:
        raise TritonShimError(f"{what}: more than {MAX_TENSOR_NUMEL} elements")
    return dims


def arange(start: int, end: int) -> tensor:
    """``[start, end)`` as an int32 block; both bounds are constants.

    Triton documents: "start: Must be a power of two", "end: Must be a power of two
    greater than start", and its tutorials all write ``tl.arange(0, BLOCK)``. So 0 is
    accepted as a start. The shim also requires the *length* to be a power of two,
    which is the rule for every other block shape; ``(0, 2**m)`` and ``(2**k,
    2**(k + 1))`` are the ranges that pass under both readings. Offsets that start
    elsewhere are written ``start + tl.arange(0, BLOCK)``.
    """
    lo, hi = _const_int(start, "tl.arange: start"), _const_int(end, "tl.arange: end")
    if hi <= lo:
        raise TritonShimError("tl.arange: end must be greater than start")
    if lo != 0 and not _is_power_of_two(lo):
        raise TritonShimError(
            f"tl.arange: start must be 0 or a power of two, got {lo} "
            "(add the offset to the block instead: `start + tl.arange(0, BLOCK)`)"
        )
    if not _is_power_of_two(hi):
        raise TritonShimError(f"tl.arange: end must be a power of two, got {hi}")
    if not _is_power_of_two(hi - lo):
        raise TritonShimError(
            f"tl.arange: the number of elements must be a power of two, got {hi - lo}"
        )
    if hi - lo > MAX_TENSOR_NUMEL:
        raise TritonShimError(f"tl.arange: more than {MAX_TENSOR_NUMEL} elements")
    return tensor(np.arange(lo, hi), int32)


def _need_dtype(ty: Any, what: str) -> dtype:
    if not isinstance(ty, dtype):
        raise TritonShimError(f"{what}: dtype must be a tl dtype such as tl.float32")
    return ty


def full(shape: Any, value: Any, dtype: dtype) -> tensor:
    """A block of ``shape`` filled with ``value``.

    ``shape`` is a tuple of constants (as documented) or a list of them (as the
    layer-norm tutorial writes), each a power of two.
    """
    ty = _need_dtype(dtype, "tl.full")
    if isinstance(value, tensor) and value.shape != ():
        raise TritonShimError("tl.full: value must be a scalar")
    return tensor(np.full(_shape(shape, "tl.full"), _as_array(value, ty)), ty)


def zeros(shape: Any, dtype: dtype) -> tensor:
    """A block of zeros: ``tl.zeros((BLOCK,), tl.float32)``; see ``full`` for ``shape``."""
    return full(shape, 0, dtype)


def zeros_like(input: tensor) -> tensor:
    return tensor(np.zeros(input.shape, input.dtype.np), input.dtype)


def cast(
    input: Any,
    dtype: dtype,
    fp_downcast_rounding: str | None = None,
    bitcast: Any = False,
) -> tensor:
    """Convert to ``dtype`` (``bitcast=True`` reinterprets the bits instead)."""
    ty = _need_dtype(dtype, "tl.cast")
    x = _as_tensor(input)
    if bitcast:
        if x.dtype.np.itemsize != ty.np.itemsize:
            raise TritonShimError("tl.cast(bitcast=True): the two types differ in size")
        return tensor(x.value.view(ty.np), ty)
    with np.errstate(all="ignore"):
        if ty.is_bool():
            return tensor(x.value != 0, ty)
        return tensor(x.value.astype(ty.np), ty)


def where(condition: Any, x: Any, y: Any) -> tensor:
    """Elementwise ``x`` where ``condition`` is true, else ``y``. Both are evaluated."""
    cond = _as_tensor(condition)
    if not cond.dtype.is_bool():
        cond = cast(cond, int1)
    a, b, ty = _operands(x, y) if not _both_scalars(x, y) else _scalar_pair(x, y)
    try:
        return tensor(np.where(cond.value, a, b), ty)
    except ValueError:
        raise TritonShimError("tl.where: the shapes do not broadcast") from None


def _both_scalars(x: Any, y: Any) -> _pybool:
    return not isinstance(x, tensor) and not isinstance(y, tensor)


def _scalar_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray, dtype]:
    ty = _promote(_scalar_dtype(x), _scalar_dtype(y))
    return _as_array(x, ty), _as_array(y, ty), ty


def _axis(axis: Any, ndim: int, what: str) -> int | None:
    if axis is None:
        return None
    axis = _const_int(axis, f"{what}: axis")
    if not -ndim <= axis < ndim:
        raise TritonShimError(f"{what}: axis {axis} is out of range for {ndim} dimensions")
    return axis


def sum(input: Any, axis: Any = None, keep_dims: Any = False, dtype: Any = None) -> tensor:
    """Sum along ``axis`` (all axes when None). Integers narrower than 32 bits are
    widened to int32 or uint32 first; ``dtype`` casts the input before summing."""
    x = _as_tensor(input)
    if dtype is not None:
        x = cast(x, _need_dtype(dtype, "tl.sum"))
    elif x.dtype.bits < 32 and not x.dtype.is_floating():
        x = cast(x, int32 if x.dtype.signed else uint32)
    with np.errstate(all="ignore"):
        out = np.sum(x.value, axis=_axis(axis, x.value.ndim, "tl.sum"), keepdims=_pybool(keep_dims),
                     dtype=x.dtype.np)  # fmt: skip
    return tensor(out, x.dtype)


def _reduce(fn: Any, what: str, input: Any, axis: Any, options: dict[str, Any]) -> tensor:
    keep = _pybool(options.pop("keep_dims", False))
    if options.pop("return_indices", False):
        raise TritonShimError(f"{what}: return_indices is not supported by the shim")
    options.pop("return_indices_tie_break_left", None)
    options.pop("propagate_nan", None)
    if options:
        raise TritonShimError(f"{what}: unexpected arguments {sorted(options)}")
    x = _as_tensor(input)
    return tensor(fn(x.value, axis=_axis(axis, x.value.ndim, what), keepdims=keep), x.dtype)


def max(input: Any, axis: Any = None, **options: Any) -> tensor:
    """The maximum along ``axis`` (all axes when None); ``keep_dims=`` as in Triton."""
    return _reduce(np.max, "tl.max", input, axis, options)


def min(input: Any, axis: Any = None, **options: Any) -> tensor:
    return _reduce(np.min, "tl.min", input, axis, options)


def maximum(x: Any, y: Any, propagate_nan: Any = None) -> tensor:
    a, b, ty = _operands(x, y) if not _both_scalars(x, y) else _scalar_pair(x, y)
    return tensor(np.fmax(a, b) if ty.is_floating() else np.maximum(a, b), ty)


def minimum(x: Any, y: Any, propagate_nan: Any = None) -> tensor:
    a, b, ty = _operands(x, y) if not _both_scalars(x, y) else _scalar_pair(x, y)
    return tensor(np.fmin(a, b) if ty.is_floating() else np.minimum(a, b), ty)


def clamp(x: Any, min: Any, max: Any) -> tensor:
    return minimum(maximum(x, min), max)


def abs(x: Any) -> tensor:
    t = _as_tensor(x)
    with np.errstate(all="ignore"):
        return tensor(np.abs(t.value), t.dtype)


def _float_fn(fn: Any, name: str) -> Callable[[Any], tensor]:
    def op(x: Any) -> tensor:
        t = _as_tensor(x)
        if not t.dtype.is_floating():
            raise TritonShimError(f"tl.{name} needs a floating-point block, got {t.dtype}")
        with np.errstate(all="ignore"):
            return tensor(fn(t.value), t.dtype)

    op.__name__ = name
    op.__doc__ = f"Elementwise {name} of a floating-point block."
    return op


sqrt = _float_fn(np.sqrt, "sqrt")
rsqrt = _float_fn(lambda v: (1 / np.sqrt(v)).astype(v.dtype), "rsqrt")
exp = _float_fn(np.exp, "exp")
log = _float_fn(np.log, "log")
sigmoid = _float_fn(lambda v: (1 / (1 + np.exp(-v))).astype(v.dtype), "sigmoid")


def cdiv(x: int, y: int) -> int:
    """``triton.cdiv``: ceiling division, for computing a grid."""
    return (x + y - 1) // y


def next_power_of_2(n: int) -> int:
    """``triton.next_power_of_2``."""
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


# --------------------------------------------------------------------------------------
# jit and launch
# --------------------------------------------------------------------------------------

#: The functions of the language, as a kernel may call them: ``tl.<name>(...)``.
LANGUAGE = frozenset({
    "program_id", "num_programs", "arange", "full", "zeros", "zeros_like", "cast", "where",
    "sum", "max", "min", "maximum", "minimum", "clamp", "abs", "sqrt", "rsqrt", "exp", "log",
    "sigmoid", "load", "store", "static_range",
})  # fmt: skip
#: Attributes and methods a kernel may use on a block.
_TENSOR_ATTRIBUTES = frozenset({"to", "cast", "sum", "max", "min", "shape", "dtype", "type", "T"})
#: Builtins a kernel may call (Triton compiles these for scalars and constants).
_ALLOWED_BUILTINS = frozenset({"range", "min", "max", "int", "float", "bool", "abs", "print"})
#: Syntax Triton's compiler does not accept in the subset this shim covers.
_FORBIDDEN_SYNTAX = (
    ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Lambda, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.List, ast.Dict, ast.Set, ast.Starred, ast.Yield,
    ast.YieldFrom, ast.Await, ast.ClassDef, ast.Global, ast.Nonlocal, ast.JoinedStr,
    ast.Delete, ast.Raise, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.NamedExpr,
)  # fmt: skip

static_range = builtins.range  # ``tl.static_range``: a loop unrolled at compile time


def _check_source(fn: Callable[..., Any]) -> None:
    """Reject kernel source that Triton could not compile (see the module docstring).

    Allowed: assignments, ``for ... in range(...)``, ``while``, ``if``, arithmetic and
    comparisons, ``tl.<function>(...)`` for the functions in ``LANGUAGE``, ``tl.<dtype>``,
    the block methods ``to``, ``cast``, ``sum``, ``max``, ``min``, the builtins
    ``range``, ``min``, ``max``, ``int``, ``float``, ``bool``, ``abs``, ``print``, and
    calls of other ``@tl.jit`` functions.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except OSError, TypeError, SyntaxError:
        return  # no source (defined in a REPL): nothing to check
    function = tree.body[0]
    scope = fn.__globals__
    this = sys.modules[__name__]

    def fail(node: ast.AST, what: str) -> None:
        line = fn.__code__.co_firstlineno + getattr(node, "lineno", 1) - 1
        raise TritonShimError(
            f"{fn.__name__} (line {line}): {what} cannot be used inside a Triton kernel"
        )

    decorators = {id(n) for d in function.decorator_list for n in ast.walk(d)}
    annotations = {
        id(n)
        for argument in (*function.args.args, *function.args.kwonlyargs)
        if argument.annotation is not None
        for n in ast.walk(argument.annotation)
    }
    # A list is allowed in one place: as the shape of `tl.zeros` or `tl.full`, the way
    # Triton's layer-norm tutorial writes it (`tl.zeros([BLOCK_SIZE], dtype=...)`).
    shapes = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("zeros", "full")
            and isinstance(node.func.value, ast.Name)
            and scope.get(node.func.value.id) is this
        ):
            given = node.args[:1] + [k.value for k in node.keywords if k.arg == "shape"]
            shapes.update(id(n) for n in given if isinstance(n, ast.List))
    for node in ast.walk(function):
        if id(node) in decorators or id(node) in annotations or id(node) in shapes:
            continue
        if node is not function and isinstance(node, ast.FunctionDef):
            fail(node, "a nested function")
        if isinstance(node, _FORBIDDEN_SYNTAX):
            fail(node, f"`{type(node).__name__}` syntax")
        if isinstance(node, ast.Attribute):
            owner = scope.get(node.value.id) if isinstance(node.value, ast.Name) else None
            if owner is this:
                is_dtype = isinstance(getattr(this, node.attr, None), dtype)
                if node.attr not in LANGUAGE and not is_dtype:
                    fail(node, f"`{node.value.id}.{node.attr}` (not in the shim's subset)")
            elif isinstance(owner, types.ModuleType):
                fail(node, f"`{node.value.id}.{node.attr}`")
            elif node.attr not in _TENSOR_ATTRIBUTES:
                fail(node, f"the attribute `.{node.attr}`")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            target = scope.get(name)
            builtin = name in _ALLOWED_BUILTINS and target is None
            if not (isinstance(target, JITFunction) or builtin):
                fail(node, f"calling `{name}`")


class JITFunction:
    """What ``@tl.jit`` returns: the kernel, launched with ``kernel[grid](*args)``."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        _check_source(fn)
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.signature = inspect.signature(fn)
        self.constexprs = frozenset(
            name
            for name, parameter in self.signature.parameters.items()
            if "constexpr" in _annotation_text(parameter.annotation)
        )
        #: Counters of the most recent launch (None before the first).
        self.last: Counters | None = None

    def __getitem__(self, grid: Any) -> Callable[..., Counters]:
        return lambda *args, **kwargs: launch(self, grid, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _launch is None:
            raise TritonShimError(
                f"{self.__name__} is a kernel: launch it with tl.launch({self.__name__}, grid, "
                f"...) or {self.__name__}[grid](...)"
            )
        return self.fn(*args, **kwargs)  # a kernel calling a helper kernel function

    def __repr__(self) -> str:
        return f"JITFunction({self.__name__})"


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", repr(annotation))


def jit(fn: Callable[..., Any]) -> JITFunction:
    """Mark ``fn`` as a kernel: the stand-in for ``@triton.jit``."""
    if isinstance(fn, JITFunction):
        return fn
    return JITFunction(fn)


def _bind(kernel: JITFunction, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = kernel.signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TritonShimError(f"{kernel.__name__}: {exc}") from None
    out: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in kernel.constexprs:
            if not isinstance(value, _SCALARS):
                raise TritonShimError(f"{kernel.__name__}: constexpr `{name}` must be a number")
            out[name] = value.item() if isinstance(value, np.generic) else value
        elif isinstance(value, (_pybool, np.bool_)):
            out[name] = tensor(value, int1)
        elif isinstance(value, (int, np.integer)):
            # Triton types an integer argument int32, or int64 when it does not fit.
            out[name] = tensor(value, int32 if -(2**31) <= int(value) < 2**31 else int64)
        elif isinstance(value, (float, np.floating)):
            out[name] = tensor(value, float32)  # a Python float arrives as float32
        else:
            out[name] = pointer(_memory_of(value, name, kernel), np.int64(0), name)
    return out


def _memory_of(value: Any, name: str, kernel: JITFunction) -> np.ndarray:
    """The tensor's memory as a flat numpy view: stores land in the caller's tensor."""
    array = value
    if type(value).__module__.split(".")[0] == "torch":
        if value.device.type != "cpu":
            raise TritonShimError(
                f"{kernel.__name__}: `{name}` is on {value.device}; the shim runs on CPU tensors"
            )
        if not value.is_contiguous():
            raise TritonShimError(
                f"{kernel.__name__}: `{name}` is not contiguous; a kernel sees raw memory, "
                "so pass `.contiguous()` tensors"
            )
        array = value.detach().numpy()
    if not isinstance(array, np.ndarray):
        raise TritonShimError(
            f"{kernel.__name__}: argument `{name}` must be a tensor, an array, or a number, "
            f"got {type(value).__name__}"
        )
    if not array.flags.c_contiguous:
        raise TritonShimError(f"{kernel.__name__}: `{name}` is not contiguous")
    _dtype_of(array.dtype)
    return array.reshape(-1)


def _is_cuda(value: Any) -> _pybool:
    device = getattr(value, "device", None)
    return getattr(device, "type", None) == "cuda"


def launch(kernel: JITFunction, grid: Any, *args: Any, **kwargs: Any) -> Counters | None:
    """Run ``kernel`` once for every point of ``grid``.

    ``grid`` is a tuple of one to three ints, or a function of the keyword
    (meta-)parameters returning one, as with ``kernel[grid](...)`` in Triton. Tensor
    arguments are CPU torch tensors or numpy arrays; the kernel writes into them in
    place. Returns the launch's counters: ``programs``, ``loads``, ``stores``,
    ``elements_loaded``, ``elements_stored`` (masked-off elements are not counted).

    When the tensors are CUDA tensors, the kernel is compiled and launched by real
    Triton instead (``compile_with_triton``) and nothing is returned.
    """
    global _launch
    if not isinstance(kernel, JITFunction):
        raise TritonShimError("tl.launch needs a kernel decorated with @tl.jit")
    if any(_is_cuda(value) for value in (*args, *kwargs.values())):
        compile_with_triton(kernel)[grid](*args, **kwargs)
        return None
    if _launch is not None:
        raise TritonShimError("a kernel cannot launch another kernel")
    if callable(grid):
        grid = grid(dict(kwargs))
    dims = tuple(grid) if isinstance(grid, Sequence) else (grid,)
    whole = all(isinstance(d, (int, np.integer)) and d >= 0 for d in dims)
    if not 1 <= len(dims) <= 3 or not whole:
        raise TritonShimError(f"the grid must be one to three non-negative ints, got {grid!r}")
    dims = tuple(int(d) for d in dims)
    arguments = _bind(kernel, args, kwargs)
    state = _Launch(dims)
    programs = list(itertools.product(*(range(d) for d in reversed(dims))))
    # A fixed shuffle: nothing may depend on the order in which programs run.
    np.random.default_rng(0).shuffle(programs)
    _launch = state
    try:
        for index, reversed_id in enumerate(programs):
            state.program, state.index = tuple(reversed(reversed_id)), index
            state.counters.add("programs")
            kernel.fn(**arguments)
    finally:
        _launch = None
    kernel.last = state.counters
    return state.counters


def compile_with_triton(kernel: JITFunction) -> Any:
    """The same kernel function, compiled by real Triton (needs ``triton`` and a GPU).

    The function's source is not touched. Its globals are copied with every name that
    refers to this module replaced by ``triton.language``, and its ``constexpr``
    annotations by Triton's, and ``triton.jit`` is applied to the result. Helper
    kernels it calls are converted the same way. **Untested**: written from Triton's
    documentation on a machine that cannot install Triton.
    """
    try:
        import triton
        import triton.language as real
    except ImportError as exc:
        raise TritonShimError(
            "these tensors are on a CUDA device, which needs the real Triton compiler: "
            "install `triton` (Linux only)"
        ) from exc
    cached = getattr(kernel, "_compiled", None)
    if cached:
        return cached
    this = sys.modules[__name__]
    scope = dict(kernel.fn.__globals__)
    kernel._compiled = False  # type: ignore[attr-defined]  # in progress
    for name in kernel.fn.__code__.co_names:  # only what this kernel refers to
        value = scope.get(name)
        if value is this:
            scope[name] = real
        elif isinstance(value, JITFunction):
            if getattr(value, "_compiled", None) is False:
                raise TritonShimError(
                    f"{kernel.__name__}: kernels cannot call each other in a cycle"
                )
            scope[name] = compile_with_triton(value)
    twin = types.FunctionType(
        kernel.fn.__code__, scope, kernel.fn.__name__, kernel.fn.__defaults__, kernel.fn.__closure__
    )
    twin.__kwdefaults__ = kernel.fn.__kwdefaults__
    twin.__module__ = kernel.fn.__module__
    twin.__qualname__ = kernel.fn.__qualname__
    twin.__annotations__ = {
        name: real.constexpr if name in kernel.constexprs else annotation
        for name, annotation in getattr(kernel.fn, "__annotations__", {}).items()
    }
    kernel._compiled = triton.jit(twin)  # type: ignore[attr-defined]
    return kernel._compiled
