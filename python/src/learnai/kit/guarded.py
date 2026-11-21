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
   sets ``allowed`` lets the solution read those names only. The object's machinery
   (``__dict__``, ``__init__``, ``__getattribute__``, ``__setattr__``, ``__delattr__``)
   is never the solution's to use.
2. Kit and test code is not restricted, with one exception: while the solution is
   running, the restricted names can be read, and attributes set, only inside a call
   the solution made *through the object* (``store.gather(...)``). So the kit's own
   methods work, and borrowing one as a plain function (``CountingKV.reference(store)``,
   ``type(store).gather(store, ...)``, ``CountingKV.__init__(store, ...)``) does not.
   Nor does code the solution has such a call run for it: a callback, or the special
   method of an argument it passed in.
3. A guarded object cannot be copied.

Python calls a special method without asking for an attribute: ``lm(tokens)``,
``len(file)``, ``file[a:b]``, ``with file:``. The special methods a guarded class
inherits (the names in ``SPECIAL``) are therefore wrapped when the class is made, and
count as calls made through the object like any other method.

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

import functools
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
#: Names of the object's machinery: restricted in every guarded class, whatever its
#: ``refused``. Each one is a way around the rules: ``store.__init__(...)`` starts the
#: counters over, ``store.__getattribute__("counters")`` reads through the kit.
MACHINERY = frozenset({"__dict__", "__init__", "__getattribute__", "__setattr__", "__delattr__"})
#: Code objects whose frames mark a call made through a guarded object: the wrappers
#: around its methods and ``Guarded.__getattribute__`` (under which a property runs).
_THROUGH: set[types.CodeType] = set()

#: The special methods that count as calls made through the object. Python looks them
#: up on the class, past ``__getattribute__``, so ``Guarded`` wraps the ones a guarded
#: class inherits. ``__repr__`` is handled by ``Guarded`` itself.
SPECIAL = (
    "__call__",
    "__len__",
    "__length_hint__",
    "__bool__",
    "__contains__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
)


def guard(ctx: Any) -> None:
    """Arm the guards for the solution of this run and forget earlier violations.

    Call it in the test, right before the solution is called::

        guard(ctx)
        got = ctx.call("sparse_attend", q, store, idx)
        check_guard()

    The file that calls ``guard`` (the problem's ``tests.py``) is trusted like the kit.
    """
    global _solution_file, _test_file
    _solution_file = ctx.solution.__file__
    _test_file = sys._getframe(1).f_code.co_filename
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


def _asker(frame: types.FrameType | None) -> str:
    """Whose code touches a guarded object: ``"solution"``, ``"borrowed"``, or ``"trusted"``.

    Walking out from ``frame``, the first frame of the solution file or of a call made
    through a guarded object decides. Frames of the kit, the judge, and the test file
    are trusted, and frames of anything else (``copy``, ``operator``, a string given to
    ``eval``, a framework) act for whoever called them:

    - the solution file, with nothing trusted in between: ``"solution"``.
    - the solution file, behind trusted code that it called some other way than through
      the object (a kit method borrowed as a function, a callback): ``"borrowed"``.
    - a call made through the object, or no solution frame at all: ``"trusted"``.
    """
    trusted = False
    while frame is not None:
        code = frame.f_code
        if code.co_filename == _solution_file:
            return "borrowed" if trusted else "solution"
        if code in _THROUGH:
            return "trusted"
        if code.co_filename == _test_file or code.co_filename.startswith(_PACKAGE):
            trusted = True
        frame = frame.f_back
    return "trusted"


def _owner(label: str) -> str:
    return label + ("'" if label.endswith("s") else "'s")


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
        return function(*args, **kwargs)

    return call


def _special(function: Callable[..., Any]) -> Callable[..., Any]:
    """A special method of a guarded class: a call made through the object."""

    @functools.wraps(function)
    def method(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    method._through_the_object = True  # type: ignore[attr-defined]
    return method


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
            (``_x``) and the object's machinery (``MACHINERY``): by default
            ``counters``, ``log``, ``with_framework``. Extend it for a test-only method
            or a setting that would rebuild the object:
            ``refused = Guarded.refused | {"peek", "seed"}``.
        allowed: if set, the *only* names the solution may read; ``refused`` is then
            not consulted.

    The solution may never set or delete an attribute, and nobody may copy the object.
    Kit and test code reads and sets everything; see the module docstring for the one
    exception (borrowed methods) and for what the guard cannot stop.

    A subclass can record a rule of its own with ``self.violation(message)``, which
    fails the case the same way. The special methods in ``SPECIAL`` that the class
    defines or inherits (``lm(tokens)``, ``len(file)``, ``file[a:b]``, ``with file:``)
    are calls made through the object, like its other methods; so is ``__repr__``:
    ``print(store)`` is a learner debugging. Any other special method that reads
    restricted names counts as a borrowed method and is refused while the solution
    runs.
    """

    label: ClassVar[str] = "the kit object"
    refused: ClassVar[frozenset[str]] = frozenset({"counters", "log", "with_framework"})
    allowed: ClassVar[frozenset[str] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in SPECIAL:
            for base in cls.__mro__[:-1]:  # not `object`
                function = vars(base).get(name)
                if function is None:
                    continue
                wrapped = getattr(function, "_through_the_object", False)
                if isinstance(function, types.FunctionType) and not wrapped:
                    setattr(cls, name, _special(function))
                break

    @classmethod
    def of(cls, plain: Any) -> Any:
        """A guarded object with the attributes of ``plain``, for an object that a kit
        factory built (``Experts.of(moe.experts(1))``, ``LM.of(draft)``).

        The attributes are shared, not copied: the counters of the guarded object are
        the counters of ``plain``.
        """
        new = object.__new__(cls)
        _state(new).update(vars(plain))
        return new

    @staticmethod
    def violation(message: str) -> None:
        """Record ``message`` as a refused access and raise ``CheckFailure``."""
        _violation(message)

    def __getattribute__(self, name: str) -> Any:
        cls = type(self)
        if cls.allowed is not None:
            restricted = name not in cls.allowed
        else:
            restricted = (
                name in cls.refused or name in MACHINERY or (name[:1] == "_" and name[:2] != "__")
            )
        if restricted and _solution_file is not None:
            who = _asker(sys._getframe(1))
            if who == "solution":
                _violation(f"`{cls.label}.{name}` may not be used in this problem")
            if who == "borrowed":
                _violation(
                    f"`{cls.label}.{name}` may not be read outside the {_owner(cls.label)} "
                    "own methods"
                )
        state = _state(self)
        if name == "__dict__":
            return state
        if name in state:
            value = state[name]
        else:
            # A property of the class runs under this frame: a call through the object.
            value = object.__getattribute__(self, name)
        if isinstance(value, (types.MethodType, types.FunctionType)):
            return _entering(value)
        return value

    def _may_set(self, name: str, frame: types.FrameType | None) -> None:
        if _solution_file is None:
            return
        who = _asker(frame)
        label = type(self).label
        if who == "solution":
            _violation(f"`{label}.{name}` may not be set in this problem")
        if who == "borrowed":
            _violation(f"`{label}.{name}` may not be set outside the {_owner(label)} own methods")

    def __setattr__(self, name: str, value: Any) -> None:
        Guarded._may_set(self, name, sys._getframe(1))
        if hasattr(getattr(type(self), name, None), "__set__"):
            object.__setattr__(self, name, value)  # a property with a setter
        else:
            _state(self)[name] = value

    def __delattr__(self, name: str) -> None:
        Guarded._may_set(self, name, sys._getframe(1))
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


_THROUGH.update(
    (_entering(len).__code__, _special(len).__code__, Guarded.__getattribute__.__code__)
)
