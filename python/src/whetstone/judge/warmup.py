"""One-time library costs are kept off the clock, so a case's time is the learner's.

A few things torch does lazily take seconds the first time and nothing afterwards:

    import torch._dynamo     about 2.5 s, half of it sympy. Triggered by the first
                             operation on a tensor of ``torch.device("meta")``, by the
                             first operation under a ``TorchDispatchMode``
                             (``FakeCudaGraph`` capture), and by ``torch.compile``.
    the first torch.compile  about 0.7 s more, whatever is compiled.

Inside a case that time would count against the case's limit and show up in its
duration, although the learner's code had nothing to do with it. Paying it before
every run would be wrong too: most problems never need it, and a problem that names
the meta device may never run an operation on it. So the cost is paid when it is
first needed, wherever that is, with the clock stopped:

- the harness worker installs a clock that can be paused (``install``): pausing stops
  the case's soft time limit, tells the supervisor to move its hard deadline, and the
  paused time is subtracted from the case's reported duration;
- an import hook (``watch_imports``) pauses the clock while one of ``SLOW_IMPORTS`` is
  imported for the first time;
- kit code that knows about another one-time cost wraps it in ``once(name, fn)``.

Outside the harness (unit tests, a notebook) nothing is installed and all of this
does nothing but run the work.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

#: Modules whose first import is slow and is nobody's fault. Which one torch reaches
#: first depends on the operation: `x + 1` on a meta tensor imports torch._dynamo
#: (which imports the other two), `x.to("meta")` imports symbolic_shapes (and with it
#: sympy, half of the cost) before anything of Dynamo.
SLOW_IMPORTS = frozenset({"torch._dynamo", "torch.fx.experimental.symbolic_shapes", "sympy"})

_pause: Callable[[], AbstractContextManager[None]] | None = None
_done: set[str] = set()


def install(pause: Callable[[], AbstractContextManager[None]] | None) -> None:
    """Set (or with None, remove) the context manager that stops the judge's clock."""
    global _pause
    _pause = pause


@contextmanager
def off_the_clock() -> Iterator[None]:
    """Run the block without it counting toward the running case's time or limit."""
    if _pause is None:
        yield
        return
    with _pause():
        yield


def once(name: str, fn: Callable[[], Any]) -> None:
    """Run ``fn`` the first time ``name`` is seen in this process, off the clock.

    For a cost that is paid once per process (the first ``torch.compile``). An
    exception from ``fn`` is swallowed: a warm-up is an optimization, and whatever it
    was preparing will simply happen later, on the clock.
    """
    if name in _done:
        return
    _done.add(name)
    with off_the_clock():
        try:
            fn()
        except Exception:  # noqa: BLE001, S110 - see the docstring
            pass


class _SlowImportFinder:
    """A ``sys.meta_path`` entry that times nothing itself: it finds the module the
    usual way and wraps the execution of its body in ``off_the_clock``."""

    def __init__(self) -> None:
        self._looking = False

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name not in SLOW_IMPORTS or self._looking:
            return None
        self._looking = True
        try:
            spec = importlib.util.find_spec(name)
        except ImportError, ValueError:
            return None
        finally:
            self._looking = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        execute = spec.loader.exec_module

        def exec_module(module: Any) -> None:
            with off_the_clock():
                execute(module)

        try:
            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        except AttributeError:
            return None
        return spec


def watch_imports() -> None:
    """Install the import hook (idempotent)."""
    if not any(isinstance(finder, _SlowImportFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _SlowImportFinder())


def first_torch_compile() -> None:
    """The one-time setup inside the first ``torch.compile`` call."""
    import warnings

    import torch
    import torch._dynamo

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Dynamo's own entry point, not `torch.compile`: a test may have replaced that
        # name to record the solution's calls.
        torch._dynamo.optimize("eager")(lambda x: x + 1)(torch.ones(2))
    torch._dynamo.reset()
