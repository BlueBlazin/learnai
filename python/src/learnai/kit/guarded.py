"""Guarded: a kit object the solution may use, and not look inside.

A cost judge hands the solution an object that counts the work asked of it. The count
means something only if the solution cannot reset the counter, read the arrays behind
it, or build an uncounted twin. A test makes its counting object guarded by mixing
``Guarded`` in, arms the guard before it calls the solution, and checks it afterward::

    from learnai.kit import CountingKV, Guarded, check_guard, guard

    class Store(Guarded, CountingKV):
        label = "store"                      # the name the learner's messages use

    store = Store(ctx.fw, keys, values)
    guard(ctx)                               # arm: this run's solution file
    got = ctx.call("sparse_attend", q, store, idx)
    check_guard()                            # fails the case if anything was refused
    check.cost(store.counters, "rows_read", at_most=20, unit="rows")   # tests may

Three rules, all about *who is asking* (the call stack decides, as for ``forbidden``):

1. Code in the solution file may read the names the class lets it read, and may set
   nothing. By default that is every public attribute except ``refused``; a class that
   sets ``allowed`` lets the solution read those names only.
2. Kit and test code is not restricted, with one exception: while the solution is
   running, the restricted names can be read only inside a call the solution made
   *through the object* (``store.gather(...)``). So the kit's own methods work, and
   borrowing one as a plain function (``CountingKV.reference(store)``,
   ``type(store).gather(store, ...)``) does not.
3. A guarded object cannot be copied.

A refused access raises ``CheckFailure`` and is recorded, so a solution that catches
the exception still fails at ``check_guard()``.

The attributes of a guarded object are not kept in the instance. They live in a table
in this module, which is why ``object.__getattribute__(store, "counters")`` and
``object.__getattribute__(store, "__dict__")`` find nothing, and ``vars(store)`` is
refused like ``store.__dict__``.

This is a guard against taking the shortcut, not a sandbox. The solution runs in this
process, and code that sets out to defeat the guard can: by importing this module and
reading its table or replacing its functions, by replacing a method on the guarded
class or ``sys._getframe``, by walking the garbage collector's object graph, or from
another thread. What the guard does stop is everything a learner writes by accident or
habit, and every cheat that needs only the object it was handed.
"""

from __future__ import annotations

import os
import sys
import types
import weakref
from collections.abc import Callable
from typing import Any, ClassVar

from learnai.judge.errors import CheckFailure

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep  # learnai/
#: id(object) -> its attributes. Entries are removed when the object is collected.
_STATE: dict[int, dict[str, Any]] = {}
_violations: list[str] = []
_solution_file: str | None = None
_test_file: str | None = None
_depth = 0  # > 0 while a call made through a guarded object is running


def guard(ctx: Any) -> None:
    """Arm the guards for the solution of this run and forget earlier violations.

    Call it in the test, right before the solution is called::

        guard(ctx)
        got = ctx.call("sparse_attend", q, store, idx)
        check_guard()

    The file that calls ``guard`` (the problem's ``tests.py``) is trusted like the kit.
    """
    global _solution_file, _test_file, _depth
    _solution_file = ctx.solution.__file__
    _test_file = sys._getframe(1).f_code.co_filename
    _depth = 0
    _violations.clear()


def check_guard() -> None:
    """Fail the case with the first access that was refused since ``guard(ctx)``.

    Call it right after the solution returns, before the other checks::

        got = ctx.call("forward_shared", dsa, h, types, k)
        check_guard()
    """
    if _violations:
        raise CheckFailure(_violations[0])


def _violation(message: str) -> None:
    _violations.append(message)
    raise CheckFailure(message)


def _asker(frame: types.FrameType | None) -> tuple[str, types.FrameType | None]:
    """Whose code is asking: ``"solution"`` or ``"trusted"``, and the frame that decided.

    The nearest frame that belongs to the solution file, to the kit and the judge, or
    to the test file decides. Frames of anything else (``copy``, ``operator``, a string
    given to ``eval``, a framework) are looked through: they act for whoever called them.
    """
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename == _solution_file:
            return "solution", frame
        if filename == _test_file or filename.startswith(_PACKAGE):
            return "trusted", frame
        frame = frame.f_back
    return "trusted", None


def _solution_is_running(frame: types.FrameType | None) -> bool:
    while frame is not None:
        if frame.f_code.co_filename == _solution_file:
            return True
        frame = frame.f_back
    return False


def _state(obj: Any) -> dict[str, Any]:
    try:
        return _STATE[id(obj)]
    except KeyError:
        state = _STATE[id(obj)] = {}
        weakref.finalize(obj, _STATE.pop, id(obj), None)
        return state


def _entering(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function``, marked as a call made through a guarded object while it runs."""

    def call(*args: Any, **kwargs: Any) -> Any:
        global _depth
        _depth += 1
        try:
            return function(*args, **kwargs)
        finally:
            _depth -= 1

    return call


class Guarded:
    """Mixin that guards a kit object handed to the solution. List it first::

        class Store(Guarded, CountingKV):
            label = "store"

        class Model(Guarded, BlockDenoiser):
            label = "model"
            allowed = frozenset({"encode", "denoise", "vocab_size", "eos_id", "pad_id"})

    Class attributes:
        label: what the object is called in messages (the argument's name in the
            statement): ``"`store.counters` may not be used in this problem"``.
        refused: names the solution may not read, besides every private name
            (``_x``): by default ``counters``, ``log``, ``with_framework``, ``__dict__``.
            Extend it for a test-only method: ``refused = Guarded.refused | {"peek"}``.
        allowed: if set, the *only* names the solution may read; ``refused`` is then
            not consulted.

    The solution may never set or delete an attribute, and nobody may copy the object.
    Kit and test code reads and sets everything; see the module docstring for the one
    exception (borrowed methods) and for what the guard cannot stop.

    A subclass can record a rule of its own with ``self.violation(message)``, which
    fails the case the same way. Python calls a special method (``len(store)``,
    ``store[i]``) without going through an attribute, so one that reads restricted
    names counts as a borrowed method and is refused while the solution runs. Only
    ``__repr__`` is exempt: ``print(store)`` is a learner debugging.
    """

    label: ClassVar[str] = "the kit object"
    refused: ClassVar[frozenset[str]] = frozenset({"counters", "log", "with_framework", "__dict__"})
    allowed: ClassVar[frozenset[str] | None] = None

    @staticmethod
    def violation(message: str) -> None:
        """Record ``message`` as a refused access and raise ``CheckFailure``."""
        _violation(message)

    def __getattribute__(self, name: str) -> Any:
        global _depth
        cls = type(self)
        if cls.allowed is not None:
            restricted = name not in cls.allowed
        else:
            restricted = name in cls.refused or (name[:1] == "_" and name[:2] != "__")
        if restricted and _solution_file is not None:
            who, frame = _asker(sys._getframe(1))
            if who == "solution":
                _violation(f"`{cls.label}.{name}` may not be used in this problem")
            if not _depth and frame is not None and _solution_is_running(frame.f_back):
                _violation(
                    f"`{cls.label}.{name}` may not be read outside the {cls.label}'s own methods"
                )
        state = _state(self)
        if name == "__dict__":
            return state
        if name in state:
            value = state[name]
        else:
            _depth += 1  # a property of the class runs here: a call through the object
            try:
                value = object.__getattribute__(self, name)
            finally:
                _depth -= 1
        if isinstance(value, (types.MethodType, types.FunctionType)):
            return _entering(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if _solution_file is not None and _asker(sys._getframe(1))[0] == "solution":
            _violation(f"`{type(self).label}.{name}` may not be set in this problem")
        if hasattr(getattr(type(self), name, None), "__set__"):
            object.__setattr__(self, name, value)  # a property with a setter
        else:
            _state(self)[name] = value

    def __delattr__(self, name: str) -> None:
        if _solution_file is not None and _asker(sys._getframe(1))[0] == "solution":
            _violation(f"`{type(self).label}.{name}` may not be set in this problem")
        try:
            del _state(self)[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self) -> str:
        # `print(store)` reaches the kit's `__repr__` without going through an
        # attribute of the object, and printing is not prying.
        return _entering(super().__repr__)()

    def _no_copy(self, *args: Any) -> Any:
        _violation(f"`{type(self).label}` may not be copied in this problem")

    __copy__ = __deepcopy__ = __reduce_ex__ = __reduce__ = _no_copy
