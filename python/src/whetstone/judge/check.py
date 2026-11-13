"""Assertions for test cases, with failure messages a learner can act on.

Every helper raises ``CheckFailure(message, detail)`` when the check does not hold.
The ``message`` is one line and never contains expected values (it is shown for hidden
cases too); the ``detail`` has the numbers and excerpts and is dropped for hidden cases.

Use as ``from whetstone.judge import check`` and then ``check.allclose(...)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from whetstone.judge.context import current, current_or_none
from whetstone.judge.errors import CheckFailure

#: Defaults for float32 results: |got - expected| <= ATOL + RTOL * |expected|.
#: float32 carries about 7 significant digits; a correct implementation that orders
#: its operations differently from the reference typically lands within 1e-6 relative,
#: so 1e-4 leaves two orders of magnitude of headroom while still catching real bugs.
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-6

#: Default false-failure probability for one statistical check. A judge run must stay
#: below 1e-9 in total, so the default leaves room for ten statistical checks.
DEFAULT_DELTA = 1e-10


# -- small utilities ---------------------------------------------------------------


def that(condition: Any, message: str, detail: str | None = None) -> None:
    """Fail with ``message`` unless ``condition`` is true. The escape hatch."""
    if not condition:
        raise CheckFailure(message, detail)


def _join(*parts: str | None) -> str:
    return "\n".join(part for part in parts if part) + "\n"


def _index(idx: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(i)) for i in idx) + "]" if len(idx) else "(scalar)"


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.7g}"
    return str(value)


def _array_text(array: np.ndarray) -> str:
    return np.array2string(array, precision=6, separator=", ", max_line_width=96, threshold=64)


def _coerce(got: Any, label: str, context: str | None) -> np.ndarray:
    """Turn a solution's result into a numeric numpy array or fail clearly."""
    if got is None:
        raise CheckFailure(
            f"{label}: got None (does the function return its result?)", _join(context)
        )
    try:
        array = np.asarray(got)
    except Exception as exc:  # noqa: BLE001
        raise CheckFailure(
            f"{label}: got a {type(got).__name__} that is not array-like", _join(context, str(exc))
        ) from None
    if array.dtype.kind not in "fiub":
        raise CheckFailure(
            f"{label}: got a {type(got).__name__} with non-numeric contents (dtype {array.dtype})",
            _join(context),
        )
    return array


def _show(name: str, array: np.ndarray) -> str:
    text = _array_text(array)
    if "\n" not in text:
        return f"{name} = {text}"
    return f"{name} =\n" + "\n".join("  " + line for line in text.splitlines())


def _excerpt(got: np.ndarray, expected: np.ndarray, idx: tuple[int, ...], width: int = 8) -> str:
    """Expected and got around ``idx``: whole arrays if tiny, else a slice of the last axis."""
    if expected.ndim == 0:
        return f"expected = {_fmt(expected[()])}\n     got = {_fmt(got[()])}"
    if expected.size <= 12:
        return f"{_show('expected', expected)}\n{_show('got', got)}"
    row, j = idx[:-1], idx[-1]
    n = expected.shape[-1]
    hi = min(n, max(0, j - width // 2) + width)
    lo = max(0, hi - width)
    where = "[" + "".join(f"{i}, " for i in row) + f"{lo}:{hi}]"
    return (
        f"expected{where} = {_array_text(expected[row][lo:hi])}\n"
        f"     got{where} = {_array_text(got[row][lo:hi])}"
    )


def _check_shape(got: np.ndarray, expected: np.ndarray, label: str, context: str | None) -> None:
    if got.shape != expected.shape:
        raise CheckFailure(
            f"{label}: shape mismatch: expected {tuple(expected.shape)}, got {tuple(got.shape)}",
            _join(context, f"expected shape: {tuple(expected.shape)}", f"got shape: {got.shape}"),
        )


def _check_dtype(got: np.ndarray, expected: np.ndarray, label: str, context: str | None) -> None:
    def family(dtype: np.dtype) -> str:
        return "int" if dtype.kind in "iu" else dtype.kind

    same = family(got.dtype) == family(expected.dtype)
    if same and expected.dtype.kind == "f":
        same = got.dtype.itemsize == expected.dtype.itemsize
    if not same:
        raise CheckFailure(
            f"{label}: dtype mismatch: expected {expected.dtype}, got {got.dtype}",
            _join(context, "The result must have the same dtype as the reference's result."),
        )


# -- numeric judge -------------------------------------------------------------------


def shape(got: Any, expected_shape: Sequence[int], *, label: str = "output") -> None:
    """Fail unless ``got`` has exactly ``expected_shape``."""
    array = _coerce(got, label, None)
    if tuple(array.shape) != tuple(expected_shape):
        raise CheckFailure(
            f"{label}: shape mismatch: expected {tuple(expected_shape)}, got {tuple(array.shape)}"
        )


def finite(got: Any, *, label: str = "output") -> None:
    """Fail if ``got`` contains NaN or infinity."""
    array = _coerce(got, label, None)
    bad = ~np.isfinite(array.astype(np.float64))
    if bad.any():
        first = tuple(np.argwhere(bad)[0])
        raise CheckFailure(
            f"{label}: contains non-finite values "
            f"({int(bad.sum())} of {array.size}, first at index {_index(first)})",
            f"value at {_index(first)}: {_fmt(array[first])}\n",
        )


def allclose(
    got: Any,
    expected: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    label: str = "output",
    dtype: bool = False,
    context: str | None = None,
) -> None:
    """Fail unless ``|got - expected| <= atol + rtol * |expected|`` everywhere.

    Defaults (rtol 1e-4, atol 1e-6) suit float32 results of order one. NaN matches NaN
    and infinities must match exactly, so an expected ``-inf`` is checkable. The shape
    is compared first, then (``dtype=True``) the dtype, then the values.
    """
    rtol = DEFAULT_RTOL if rtol is None else rtol
    atol = DEFAULT_ATOL if atol is None else atol
    want = np.asarray(expected)
    have = _coerce(got, label, context)
    _check_shape(have, want, label, context)
    if dtype:
        _check_dtype(have, want, label, context)

    g = have.astype(np.float64)
    e = want.astype(np.float64)
    g_nan, e_nan = np.isnan(g), np.isnan(e)
    same_special = (g_nan & e_nan) | (np.isinf(e) & (g == e))
    structural = (g_nan != e_nan) | (np.isinf(g) != np.isinf(e)) | (np.isinf(e) & (g != e))
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs(g - e)
        tol = atol + rtol * np.abs(e)
    err = np.where(same_special, 0.0, err)
    err = np.where(structural, np.inf, err)
    bad = structural | (np.isfinite(e) & (err > tol))
    if not bad.any():
        return

    n_bad, size = int(bad.sum()), int(bad.size)
    worst = tuple(int(i) for i in np.unravel_index(np.argmax(np.where(bad, err, -1.0)), bad.shape))
    unexpected_nan = g_nan & ~e_nan
    unexpected_inf = np.isinf(g) & ~np.isinf(e)
    if unexpected_nan.any():
        first = tuple(np.argwhere(unexpected_nan)[0])
        message = (
            f"{label}: got NaN where a number is expected "
            f"({int(unexpected_nan.sum())} of {size} values, first at index {_index(first)})"
        )
        worst = tuple(int(i) for i in first)
    elif unexpected_inf.any():
        first = tuple(np.argwhere(unexpected_inf)[0])
        message = (
            f"{label}: got infinity where a finite number is expected "
            f"({int(unexpected_inf.sum())} of {size} values, first at index {_index(first)})"
        )
        worst = tuple(int(i) for i in first)
    else:
        message = (
            f"{label}: {n_bad} of {size} values differ from the expected result "
            f"(max abs error {_fmt(err[worst])} at index {_index(worst)})"
        )

    lines = [
        f"shape {tuple(want.shape)}, dtype {have.dtype}",
        f"tolerance: |got - expected| <= {atol:g} + {rtol:g} * |expected|",
        f"mismatched values: {n_bad} of {size}",
        (
            f"max abs error: {_fmt(err[worst])} at index {_index(worst)} "
            f"(expected {_fmt(want[worst])}, got {_fmt(have[worst])})"
        ),
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.isfinite(err) & np.isfinite(e) & (e != 0), err / np.abs(e), -1.0)
    if rel.max() > 0:
        at = tuple(int(i) for i in np.unravel_index(np.argmax(rel), rel.shape))
        lines.append(
            f"max rel error: {_fmt(rel[at])} at index {_index(at)} "
            f"(expected {_fmt(want[at])}, got {_fmt(have[at])})"
        )
    lines.append(_excerpt(have, want, worst))
    raise CheckFailure(message, _join(context, *lines))


def equal(got: Any, expected: Any, *, label: str = "output", context: str | None = None) -> None:
    """Fail unless ``got`` equals ``expected`` exactly.

    For integer, bool, and token arrays (shape first, then values, integer width
    ignored) and for plain Python values (lists of tokens, strings, dicts).

    A list or tuple of 0-d arrays (what a solution returns when it collects scalar
    tensors, ``[logits.argmax() for ...]``) is compared as a list of scalars: each 0-d
    array counts as the number it holds. An array with one or more dimensions inside a
    list is never squeezed into a scalar; it is compared as an array, shape first.
    """
    got, expected = _scalars(got), _scalars(expected)
    if isinstance(expected, np.ndarray) or isinstance(got, np.ndarray):
        want = np.asarray(expected)
        have = _coerce(got, label, context)
        _check_shape(have, want, label, context)
        with np.errstate(invalid="ignore"):
            differs = ~((have == want) | (_isnan(have) & _isnan(want)))
        if not differs.any():
            return
        first = tuple(int(i) for i in np.argwhere(differs)[0])
        raise CheckFailure(
            f"{label}: {int(differs.sum())} of {differs.size} values differ from the expected "
            f"result (first at index {_index(first)})",
            _join(
                context,
                f"first difference at index {_index(first)}: "
                f"expected {_fmt(want[first])}, got {_fmt(have[first])}",
                _excerpt(have, want, first),
            ),
        )

    if got is None and expected is not None:
        raise CheckFailure(
            f"{label}: got None (does the function return its result?)", _join(context)
        )
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(got) != len(expected):
            raise CheckFailure(
                f"{label}: expected {len(expected)} item{'' if len(expected) == 1 else 's'}, "
                f"got {len(got)}",
                _join(context, f"expected: {_short(expected)}", f"     got: {_short(got)}"),
            )
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            if _holds_array(g) or _holds_array(e):
                # Arrays inside a list: compare them as arrays (shape, then values), or
                # element by element when they are nested deeper.
                equal(g, e, label=f"{label}[{i}]", context=context)
                continue
            if not _py_equal(g, e):
                lo = max(0, i - 4)
                raise CheckFailure(
                    f"{label}: differs from the expected result (first difference at position {i})",
                    _join(
                        context,
                        f"position {i}: expected {_short(e)}, got {_short(g)}",
                        f"expected[{lo}:{lo + 10}] = {_short(list(expected[lo : lo + 10]))}",
                        f"     got[{lo}:{lo + 10}] = {_short(list(got[lo : lo + 10]))}",
                    ),
                )
        return
    if not _py_equal(got, expected):
        raise CheckFailure(
            f"{label}: differs from the expected result",
            _join(context, f"expected: {_short(expected)}", f"     got: {_short(got)}"),
        )


def _scalars(value: Any) -> Any:
    """Replace 0-d arrays and numpy scalars inside lists and tuples by Python scalars.

    The top-level value itself is left alone (a 0-d array result is an array result).
    """
    if isinstance(value, (list, tuple)):
        return type(value)(_scalar_item(item) for item in value)
    return value


def _scalar_item(item: Any) -> Any:
    if isinstance(item, np.generic) or (isinstance(item, np.ndarray) and item.ndim == 0):
        return item.item()
    if isinstance(item, (list, tuple)):
        return type(item)(_scalar_item(inner) for inner in item)
    return item


def _holds_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    return isinstance(value, (list, tuple)) and any(_holds_array(item) for item in value)


def _isnan(array: np.ndarray) -> np.ndarray:
    return np.isnan(array) if array.dtype.kind == "f" else np.zeros(array.shape, dtype=bool)


def _py_equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - e.g. comparing arrays inside containers
        return False


def _short(value: Any, limit: int = 400) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def match(
    got: Any,
    expected: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    dtype: bool = False,
    label: str = "output",
    context: str | None = None,
) -> None:
    """Compare a result with the oracle's result, whatever its structure.

    Float arrays and floats use ``allclose``; integer and bool arrays and everything
    else use ``equal``; tuples, dicts, and lists that contain arrays are compared
    element by element with the position in the label (``output[1]``).
    """
    if isinstance(expected, np.ndarray):
        if isinstance(got, (list, tuple, dict)):
            raise CheckFailure(
                f"{label}: expected a tensor, got a {type(got).__name__}", _join(context)
            )
        if expected.dtype.kind == "f":
            allclose(got, expected, rtol=rtol, atol=atol, label=label, dtype=dtype, context=context)
        else:
            if dtype:
                have = _coerce(got, label, context)
                _check_shape(have, expected, label, context)
                _check_dtype(have, expected, label, context)
            equal(got, expected, label=label, context=context)
        return
    if isinstance(expected, dict):
        if not isinstance(got, dict) or set(got) != set(expected):
            keys = sorted(map(str, got)) if isinstance(got, dict) else type(got).__name__
            raise CheckFailure(
                f"{label}: expected a dict with keys {sorted(map(str, expected))}, got {keys}",
                _join(context),
            )
        for key, value in expected.items():
            match(got[key], value, rtol=rtol, atol=atol, dtype=dtype,
                  label=f"{label}[{key!r}]", context=context)  # fmt: skip
        return
    if (
        isinstance(expected, list)
        and expected
        and all(isinstance(item, (np.ndarray, np.generic)) and item.ndim == 0 for item in expected)
    ):
        # A list of scalar tensors is a list of scalars. Compare it as one sequence
        # (one message, with the position) instead of as len(expected) tiny tensors.
        if not isinstance(got, (list, tuple)):
            raise CheckFailure(
                f"{label}: expected a list of {len(expected)} scalars, got a "
                f"{type(got).__name__}",
                _join(context),
            )
        want = np.array([item.item() for item in expected])
        have = _scalars(list(got))
        if want.dtype.kind == "f":
            if len(have) != len(want):
                raise CheckFailure(
                    f"{label}: expected {len(want)} values, got {len(have)}", _join(context)
                )
            allclose(have, want, rtol=rtol, atol=atol, label=label, context=context)
        else:
            equal(have, want.tolist(), label=label, context=context)
        return
    nested = isinstance(expected, tuple) or (
        isinstance(expected, list)
        and any(isinstance(item, (np.ndarray, tuple, list, dict)) for item in expected)
    )
    if nested:
        if not isinstance(got, (tuple, list)) or len(got) != len(expected):
            what = f"{len(got)} values" if isinstance(got, (tuple, list)) else type(got).__name__
            raise CheckFailure(
                f"{label}: expected {len(expected)} values, got {what}", _join(context)
            )
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            match(g, e, rtol=rtol, atol=atol, dtype=dtype, label=f"{label}[{i}]", context=context)
        return
    if isinstance(expected, float):
        allclose(got, expected, rtol=rtol, atol=atol, label=label, context=context)
        return
    equal(got, expected, label=label, context=context)


def raises(
    exception: type[BaseException], fn: Callable[..., Any], *args: Any, label: str = "call"
) -> None:
    """Fail unless ``fn(*args)`` raises ``exception`` (for specified error behavior)."""
    try:
        fn(*args)
    except exception:
        return
    except CheckFailure:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CheckFailure(
            f"{label}: expected {exception.__name__}, but {type(exc).__name__} was raised",
            f"{type(exc).__name__}: {exc}\n",
        ) from None
    raise CheckFailure(f"{label}: expected {exception.__name__}, but nothing was raised")


# -- property judge ------------------------------------------------------------------


def forall(prop: Callable[[np.random.Generator], None], *, trials: int = 25) -> None:
    """Run the invariant ``prop(rng)`` on ``trials`` seeded random inputs.

    Each trial gets its own generator (``ctx.child_rng(i)``), and ``ctx.rng`` (hence
    ``ctx.randn`` and friends) is pointed at it for the duration of the trial. The
    first failing trial is reported with its index and seed so it can be replayed with
    ``np.random.default_rng([seed, trial])``.
    """
    ctx = current()
    outer = ctx.rng
    try:
        for trial in range(trials):
            ctx.rng = ctx.child_rng(trial)
            try:
                prop(ctx.rng)
            except CheckFailure as failure:
                where = f"property failed on trial {trial + 1} of {trials}"
                raise CheckFailure(
                    f"{failure.message} ({where})",
                    _join(
                        f"{where}; replay with np.random.default_rng([{ctx.seed}, {trial}])",
                        failure.detail,
                    ),
                ) from None
    finally:
        ctx.rng = outer


# -- statistical judge ---------------------------------------------------------------


def _kl_bernoulli(a: float, p: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(p), in nats."""
    total = 0.0
    if a > 0:
        total += math.inf if p <= 0 else a * math.log(a / p)
    if a < 1:
        total += math.inf if p >= 1 else (1 - a) * math.log((1 - a) / (1 - p))
    return total


def _kl_interval(p: float, level: float) -> tuple[float, float]:
    """The frequencies ``a`` with ``KL(a || p) <= level``, as an interval around p."""

    def solve(inside: float, outside: float) -> float:
        for _ in range(60):
            mid = (inside + outside) / 2
            if _kl_bernoulli(mid, p) <= level:
                inside = mid
            else:
                outside = mid
        return inside

    return solve(p, 0.0), solve(p, 1.0)


def distribution(
    samples: Any = None,
    probs: Any = None,
    *,
    counts: Any = None,
    delta: float = DEFAULT_DELTA,
    label: str = "samples",
) -> None:
    """Fail unless the empirical distribution of ``samples`` is consistent with ``probs``.

    ``samples`` are outcomes in ``range(K)`` (or pass per-outcome ``counts``), produced
    by calling the solution N times with independent uniform numbers that the test
    supplied; ``probs`` is the exact target distribution over the K outcomes.

    The bound. For one outcome with target probability p, the observed frequency f is
    a mean of N independent Bernoulli(p) draws when the solution is correct. The
    Chernoff bound gives, for any a > p,  P(f >= a) <= exp(-N * KL(a || p)), and the
    mirror image for a < p, where KL is the Bernoulli relative entropy. So

        P( N * KL(f || p) >= t ) <= 2 * exp(-t).

    The check fails when any outcome has ``N * KL(f_i || p_i) >= t`` with
    ``t = ln(2K / delta)``. By the union bound over K outcomes, a correct solution
    fails with probability at most ``delta`` (default 1e-10) over the test's choice of
    uniform numbers. Outcomes with p = 0 must never occur. Unlike a chi-square test
    this is a finite-sample guarantee, and unlike Hoeffding it stays sharp for rare
    outcomes. A wrong distribution is detected once N is large enough that the true
    frequency of some outcome falls outside the acceptance interval, whose half-width
    is about ``sqrt(2 * p * (1 - p) * t / N)``.
    """
    target = np.asarray(probs, dtype=np.float64).reshape(-1)
    k = target.size
    if k < 2 or (target < 0).any() or abs(target.sum() - 1.0) > 1e-6:
        raise ValueError("probs must be a distribution over at least two outcomes")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if counts is None:
        values = _coerce(samples, label, None).reshape(-1)
        rounded = values.astype(np.int64)
        outside = (rounded != values) | (rounded < 0) | (rounded >= k)
        if outside.any():
            first = int(np.argmax(outside))
            raise CheckFailure(
                f"{label}: sample {first} is {_fmt(values[first])}, "
                f"which is not an outcome in range({k})"
            )
        observed = np.bincount(rounded, minlength=k).astype(np.int64)
    else:
        observed = np.asarray(counts, dtype=np.int64).reshape(-1)
        if observed.size != k:
            raise ValueError("counts and probs must have the same length")
    n = int(observed.sum())
    if n == 0:
        raise CheckFailure(f"{label}: no samples to compare")

    threshold = math.log(2 * k / delta)
    stats = np.array([n * _kl_bernoulli(observed[i] / n, float(target[i])) for i in range(k)])
    if not (stats >= threshold).any():
        return
    worst = int(np.argmax(stats))
    direction = "more" if observed[worst] / n > target[worst] else "less"
    rows = ["outcome   target  allowed frequency    observed   count"]
    order = range(k) if k <= 24 else np.argsort(-stats)[:12]
    for i in order:
        lo, hi = _kl_interval(float(target[i]), threshold / n)
        flag = "  <-- outside" if stats[i] >= threshold else ""
        rows.append(
            f"{int(i):7d}  {target[i]:7.4f}  [{lo:7.4f}, {hi:7.4f}]  "
            f"{observed[i] / n:9.4f}  {int(observed[i]):6d}{flag}"
        )
    raise CheckFailure(
        f"{label}: the empirical distribution does not match the target: outcome {worst} "
        f"occurs {direction} often than it should ({n} samples)",
        _join(
            f"{n} samples, {k} outcomes; allowed frequencies have false-failure "
            f"probability below {delta:g}",
            *rows,
        ),
    )


def mean(
    values: Any,
    expected: float,
    *,
    low: float,
    high: float,
    delta: float = DEFAULT_DELTA,
    label: str = "mean",
) -> None:
    """Fail unless the mean of ``values`` is consistent with an expectation of ``expected``.

    ``values`` must be N independent draws that are guaranteed to lie in
    ``[low, high]`` (an acceptance indicator is in [0, 1]). Hoeffding's inequality
    gives  P(|mean - expected| >= eps) <= 2 * exp(-2 * N * eps^2 / (high - low)^2),
    so the check allows  eps = (high - low) * sqrt(ln(2 / delta) / (2 * N))  and a
    correct solution fails with probability at most ``delta``.
    """
    array = _coerce(values, label, None).astype(np.float64).reshape(-1)
    n = array.size
    if n == 0:
        raise CheckFailure(f"{label}: no values to average")
    if array.min() < low or array.max() > high:
        raise CheckFailure(
            f"{label}: values fall outside the possible range [{low:g}, {high:g}]",
            f"min {array.min():g}, max {array.max():g}\n",
        )
    eps = (high - low) * math.sqrt(math.log(2 / delta) / (2 * n))
    observed = float(array.mean())
    if abs(observed - expected) > eps:
        raise CheckFailure(
            f"{label}: the average over {n} samples is not consistent with the target",
            f"expected {expected:.6g} +/- {eps:.3g} (false-failure probability below "
            f"{delta:g}), observed {observed:.6g}\n",
        )


# -- cost judge, and bounds in general -------------------------------------------------


def at_most(name: str, value: float, limit: float, *, unit: str | None = None) -> None:
    """Report metric ``name`` with its ``limit`` and fail if ``value`` exceeds it."""
    ctx = current_or_none()
    if ctx is not None:
        ctx.metric(name, value, unit, limit)
    if value > limit:
        suffix = f" {unit}" if unit else ""
        raise CheckFailure(
            f"{name}: {_fmt(value)}{suffix} exceeds the limit of {_fmt(limit)}{suffix}"
        )


def cost(
    counters: Any,
    name: str,
    *,
    at_most: float | None = None,
    exactly: float | None = None,
    at_least: float | None = None,
    unit: str | None = None,
) -> None:
    """Cost judge: assert on an instrumentation counter of a kit object.

    ``counters`` is a kit ``Counters`` object or a mapping; ``name`` is the counter.
    The value is reported as a metric (with ``at_most`` or ``exactly`` as its limit)
    whether or not the assertion holds. Counts are exact, so there is no tolerance.
    """
    if isinstance(counters, Mapping):
        value = counters[name]
    else:
        value = getattr(counters, name)
    limit = at_most if at_most is not None else exactly
    ctx = current_or_none()
    if ctx is not None:
        ctx.metric(name, value, unit, limit)
    suffix = f" {unit}" if unit else ""
    if exactly is not None and value != exactly:
        raise CheckFailure(f"{name}: expected exactly {_fmt(exactly)}{suffix}, got {_fmt(value)}")
    if at_most is not None and value > at_most:
        raise CheckFailure(
            f"{name}: {_fmt(value)}{suffix} exceeds the limit of {_fmt(at_most)}{suffix}"
        )
    if at_least is not None and value < at_least:
        raise CheckFailure(
            f"{name}: {_fmt(value)}{suffix} is below the required {_fmt(at_least)}{suffix}"
        )
