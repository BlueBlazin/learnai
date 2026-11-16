"""Forbidden calls: the enforcement behind "do not call the built-in ``softmax``".

A problem lists dotted names in ``problem.toml``::

    forbidden = ["torch.softmax", "torch.nn.functional.softmax", "torch.Tensor.softmax"]

While a run is judged, each of those attributes is replaced by a guard. A guard looks at
the call stack: when a frame of the *solution file* is on it, the call is recorded as a
violation and ``ForbiddenCall`` is raised; otherwise (the oracle, the same-framework
reference, the tests, the framework's own internals) the original runs untouched. So
references need no special treatment, and a reference that itself used a forbidden call
would be caught when ``learnai validate`` judges it as a solution.

What a name can be:

* a function in a module (``torch.softmax``, ``jax.nn.softmax``, ``time.sleep``). Every
  alias of the same object in any loaded module is guarded too, so ``from torch import
  softmax`` and private re-exports do not get around it;
* a method on a class (``torch.Tensor.softmax``);
* a class (``torch.nn.RMSNorm``, ``mlx.nn.RoPE``): constructing it is the violation.
  The guard goes on the class's ``__init__`` (or on a ``__new__`` written in Python).
  A type implemented in C that constructs in ``__new__``
  (``json.scanner.make_scanner``) cannot be guarded without breaking it for every
  caller, so ``learnai validate`` refuses the name and points to ``forbidden_source``.

Because the stack is inspected, a forbidden function reached *through* another library
function still counts (``torch.nn.Softmax()(x)`` calls ``torch.nn.functional.softmax``).

A method on a type that cannot be modified (``mlx.core.array`` is immutable) cannot be
guarded at run time. For those names only, the solution's source is scanned instead:
any ``<expression>.<method>`` attribute access that is not on an imported module counts
(see ``Guard.scan_source``).

What cannot be a name: anything that is not called. A dtype object such as
``torch.bfloat16`` is only ever *mentioned*, so there is nothing to intercept. Those go
in the second list, ``forbidden_source``: plain identifiers that must not appear in the
solution's source at all (``Guard.scan_source``)::

    forbidden_source = ["bfloat16"]

rules out ``torch.bfloat16``, ``from torch import bfloat16``, ``x.bfloat16()``,
``dtype="bfloat16"``, and ``getattr(torch, "bfloat16")`` alike.

This is a guard against taking the shortcut, not a sandbox: the solution runs in this
process, and code that sets out to defeat the guard (another thread, ``getattr`` with a
computed name on an immutable type) can.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from learnai.judge.errors import CheckFailure

#: ``package.attribute`` at least; every part a Python identifier.
NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
_MARK = "__learnai_forbidden__"
_CHECK_PREFIX = "@@FORBIDDEN@@ "


class ForbiddenCall(CheckFailure):
    """The solution called something the problem's ``forbidden`` list rules out."""


class Unresolved(Exception):
    """A forbidden name does not resolve to an attribute that can be guarded."""


@dataclass(frozen=True)
class Violation:
    name: str
    file: str
    line: int
    function: str | None = None  # None for a finding of the source scan

    @property
    def message(self) -> str:
        where = f"{self.file} line {self.line}"
        if self.function is None:
            return f"`{self.name}` is not allowed in this problem (used at {where})"
        return (
            f"`{self.name}` is not allowed in this problem (called at {where}, in {self.function})"
        )


def resolve(name: str) -> tuple[Any, str, Any]:
    """Return ``(container, attribute, value)`` for a dotted name, importing as needed."""
    if not NAME_PATTERN.match(name):
        raise Unresolved("must be a dotted name such as `torch.softmax`")
    parts = name.split(".")
    container: Any = None
    cut = 0
    for cut in range(len(parts) - 1, 0, -1):
        try:
            container = importlib.import_module(".".join(parts[:cut]))
        except Exception:  # noqa: BLE001, S112 - not a module (or not installed): try shorter
            continue
        break
    if container is None:
        raise Unresolved(f"cannot import `{parts[0]}`")
    for part in parts[cut:-1]:
        try:
            container = getattr(container, part)
        except AttributeError:
            raise Unresolved(f"`{part}` does not exist") from None
    try:
        value = getattr(container, parts[-1])
    except AttributeError:
        raise Unresolved(f"`{parts[-1]}` does not exist") from None
    return container, parts[-1], value


class Guard:
    """Installs, enforces, and removes the guards for one run."""

    def __init__(
        self,
        names: Iterable[str],
        solution_file: str | os.PathLike[str],
        source_names: Iterable[str] = (),
    ) -> None:
        self.names = tuple(dict.fromkeys(names))
        #: Identifiers that must not appear in the solution's source (``forbidden_source``).
        self.source_names = tuple(dict.fromkeys(source_names))
        self.solution_file = str(solution_file)
        self.violations: list[Violation] = []
        #: name -> why it could not be guarded (reported by ``learnai validate``).
        self.unresolved: dict[str, str] = {}
        #: method name -> forbidden name, for methods on immutable types (source scan).
        self.static: dict[str, str] = {}
        self._pending = list(self.names)
        self._undo: list[Callable[[], None]] = []
        #: id(original) -> (original, guard) for module-level functions.
        self._aliases: dict[int, tuple[Any, Any]] = {}
        #: Names of the modules whose namespaces have been searched for aliases.
        self._searched: set[str] = set()

    # -- installing ---------------------------------------------------------------

    def install(self, *, import_roots: bool = False) -> None:
        """Guard every name whose top-level package is loaded. Safe to call again.

        A problem for three frameworks lists names from all of them, and a run loads
        only one or two, so names of packages that are not in ``sys.modules`` wait.
        Calling this again (the harness does, before every case) picks up packages the
        solution imported in the meantime. ``import_roots=True`` imports whatever is
        needed instead; ``learnai validate`` uses it to check every name.
        """
        for name in list(self._pending):
            if not import_roots and name.split(".")[0] not in sys.modules:
                continue
            self._pending.remove(name)
            try:
                self._install_one(name)
            except Unresolved as exc:
                self.unresolved[name] = str(exc)
        if self._aliases and len(sys.modules) != len(self._searched):
            self._guard_aliases()

    def _install_one(self, name: str) -> None:
        container, attr, value = resolve(name)
        if getattr(value, _MARK, None) is not None:
            return  # an alias of a name that is already guarded
        if isinstance(value, type):
            # Guard construction and keep the class object: isinstance checks and
            # subclasses inside the framework must keep working.
            self._guard_construction(name, value)
        elif isinstance(container, type):
            try:
                self._guard_class_attribute(name, container, attr)
            except Unresolved:
                self.static[attr] = name  # immutable type: enforced by the source scan
        elif isinstance(container, types.ModuleType):
            if not callable(value):
                raise Unresolved(
                    "is not callable, so there is no call to guard (a dtype or a constant "
                    "is only mentioned); list its identifier in `forbidden_source` instead"
                )
            self._aliases[id(value)] = (value, self._guarded(name, value))
            self._searched.clear()  # a new function: every module has to be searched again
        else:
            raise Unresolved(f"is an attribute of a {type(container).__name__}, not of a module")

    def _guard_construction(self, name: str, cls: type) -> None:
        """Guard the constructor of ``cls`` without changing how it constructs.

        Where the guard goes depends on where the class takes its arguments:

        * a class with an ``__init__`` (its own or a base's): on ``__init__``;
        * a class that constructs in a ``__new__`` written in Python: on ``__new__``;
        * a class with neither: on ``__init__``, which then takes no arguments.

        What is left is a type written in C that takes its arguments in ``__new__``
        (``json.scanner.make_scanner``). Neither attribute can be replaced on such a
        type without breaking it for every caller: an added ``__init__`` makes
        ``object.__init__`` reject the constructor arguments, and once ``__new__`` has
        been assigned, CPython refuses to call the original C constructor at all
        ("is not safe, use object.__new__()"), even after the assignment is undone.
        So it is refused, and ``learnai validate`` says to use ``forbidden_source``.
        """
        if cls.__init__ is not object.__init__ or cls.__new__ is object.__new__:
            self._guard_class_attribute(name, cls, "__init__")
            return
        raw = inspect.getattr_static(cls, "__new__")
        if isinstance(raw, staticmethod) and isinstance(raw.__func__, types.FunctionType):
            self._guard_class_attribute(name, cls, "__new__")
            return
        raise Unresolved(
            "is a type implemented in C that takes its arguments in `__new__`; a guard on "
            "it would break it for every caller. Rule out its identifier with "
            f'`forbidden_source = ["{name.rsplit(".", 1)[-1]}"]` instead'
        )

    def _guard_class_attribute(self, name: str, cls: type, attr: str) -> None:
        raw = inspect.getattr_static(cls, attr)
        if isinstance(raw, staticmethod):
            replacement: Any = staticmethod(self._guarded(name, raw.__func__))
        elif isinstance(raw, classmethod):
            replacement = classmethod(self._guarded(name, raw.__func__))
        elif callable(raw):
            replacement = self._guarded(name, raw)
        else:
            raise Unresolved("is not a method")
        own = attr in vars(cls)
        try:
            setattr(cls, attr, replacement)
        except TypeError as exc:
            raise Unresolved(f"cannot be guarded: {exc}") from None

        def undo() -> None:
            if own:
                setattr(cls, attr, raw)
            else:
                delattr(cls, attr)

        self._undo.append(undo)

    def _guard_aliases(self) -> None:
        """Point every module attribute that is a guarded function at its guard.

        Modules are searched once; later calls only look at modules loaded since (the
        frameworks import lazily, so new ones keep appearing during the first cases).
        """
        this = sys.modules.get(__name__)
        for module_name, module in list(sys.modules.items()):
            if module_name in self._searched:
                continue
            self._searched.add(module_name)
            if not isinstance(module, types.ModuleType) or module is this:
                continue
            namespace = vars(module)
            for key, value in list(namespace.items()):
                entry = self._aliases.get(id(value))
                if entry is None or entry[0] is not value:
                    continue
                namespace[key] = entry[1]
                self._undo.append(lambda ns=namespace, k=key, v=value: ns.__setitem__(k, v))

    def _guarded(self, name: str, original: Any) -> Callable[..., Any]:
        check = self._check

        def guard(*args: Any, **kwargs: Any) -> Any:
            check(name)
            return original(*args, **kwargs)

        for attribute in ("__name__", "__qualname__", "__doc__", "__module__"):
            try:
                setattr(guard, attribute, getattr(original, attribute))
            except AttributeError, TypeError:
                pass
        setattr(guard, _MARK, name)
        return guard

    def uninstall(self) -> None:
        """Restore everything (the judge's own tests share one process)."""
        while self._undo:
            self._undo.pop()()
        self._aliases.clear()
        self._searched.clear()
        self._pending = list(self.names)
        self.static.clear()

    # -- enforcing ----------------------------------------------------------------

    def _check(self, name: str) -> None:
        frame = sys._getframe(2)  # the caller of the guard
        while frame is not None:
            code = frame.f_code
            if code.co_filename == self.solution_file:
                violation = Violation(
                    name, os.path.basename(self.solution_file), frame.f_lineno, code.co_name
                )
                self.violations.append(violation)
                raise ForbiddenCall(violation.message, self.rule_text())
            frame = frame.f_back

    def rule_text(self) -> str:
        text = ""
        if self.names:
            text += "The statement rules these out for this problem:\n"
            text += "".join(f"  {name}\n" for name in self.names)
        if self.source_names:
            text += "These identifiers may not appear in the solution at all:\n"
            text += "".join(f"  {name}\n" for name in self.source_names)
        return text

    def take(self) -> list[Violation]:
        """Violations recorded since the last call (also those the solution swallowed)."""
        taken, self.violations = self.violations, []
        return taken

    def scan_source(self) -> Violation | None:
        """Find what only the solution's source can show. Returns the first finding.

        Two rules. Every identifier in ``self.source_names`` (``forbidden_source``):
        see ``find_identifiers``. And every method in ``self.static`` (a forbidden
        method of an immutable type): the first ``<expr>.<method>`` whose expression is
        not a chain of names starting at an imported module (those are module
        functions, which the run-time guards cover).
        """
        if not self.static and not self.source_names:
            return None
        try:
            with open(self.solution_file, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except OSError, SyntaxError, ValueError:
            return None  # importing the solution reports this properly
        base_name = os.path.basename(self.solution_file)
        named = find_identifiers(tree, self.source_names)
        if named:
            line, name = min(named)
            return Violation(name, base_name, line)
        if not self.static:
            return None
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imported.add(alias.asname or alias.name.split(".")[0])
        found: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in self.static:
                continue
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in imported:
                continue
            found.append((node.lineno, self.static[node.attr]))
        if not found:
            return None
        line, name = min(found)
        return Violation(name, base_name, line)


def find_identifiers(tree: ast.AST, identifiers: Iterable[str]) -> list[tuple[int, str]]:
    """Where the given identifiers appear in a parsed module: ``(line, identifier)``.

    An identifier counts wherever the source could be naming the thing: as a bare name
    (``bfloat16``), as an attribute (``torch.bfloat16``, ``x.bfloat16()``), as any part
    of an imported name (``from torch import bfloat16``, ``import safetensors.torch``
    for ``safetensors``), as a keyword argument, and as a string that is exactly the
    identifier (``getattr(torch, "bfloat16")``, ``dtype="bfloat16"``). Comments and
    longer strings (docstrings) do not count.
    """
    wanted = set(identifiers)
    found: list[tuple[int, str]] = []
    if not wanted:
        return found
    for node in ast.walk(tree):
        hits: Iterable[str] = ()
        if isinstance(node, ast.Name):
            hits = (node.id,)
        elif isinstance(node, ast.Attribute):
            hits = (node.attr,)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            hits = (node.value,)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            hits = (node.arg,)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            parts = [part for alias in node.names for part in alias.name.split(".")]
            parts += [alias.asname for alias in node.names if alias.asname]
            if isinstance(node, ast.ImportFrom) and node.module:
                parts += node.module.split(".")
            hits = parts
        line = getattr(node, "lineno", None)
        if line is None:
            continue
        found.extend((line, hit) for hit in hits if hit in wanted)
    return found


def check_names(names: Iterable[str]) -> dict[str, str]:
    """Try to guard every name for real. Returns ``{name: problem}`` for the failures."""
    guard = Guard(names, "<no solution>")
    try:
        guard.install(import_roots=True)
        return dict(guard.unresolved)
    finally:
        guard.uninstall()


def parse_check_output(text: str) -> dict[str, str] | None:
    """The result printed by ``python -m learnai.judge.forbidden`` (None if absent)."""
    for line in reversed(text.splitlines()):
        if line.startswith(_CHECK_PREFIX):
            try:
                result = json.loads(line[len(_CHECK_PREFIX) :])
            except ValueError:
                return None
            return result if isinstance(result, dict) else None
    return None


def main(argv: list[str] | None = None) -> int:
    """``python -m learnai.judge.forbidden NAME...``: used by ``learnai validate``."""
    names = list(sys.argv[1:] if argv is None else argv)
    print(_CHECK_PREFIX + json.dumps(check_names(names)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
