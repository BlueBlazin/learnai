"""Runtime documentation for names the language server has no docstring for.

The language server reads docstrings from source and stub files, so names that only
get their docstring at import time (most of ``torch.nn.functional``, all of
``torch.linalg``) come back bare. This module answers "what is the signature and
docstring of this dotted name" by importing the library and asking ``inspect``.

Core starts it as a long-lived process (``python -I -m learnai.editor.docs``) and
talks to it in line-delimited JSON (ARCHITECTURE section 4.4): one request object per
line on stdin, one response object per line on stdout, matched by ``id``.

Safety rules, which the tests pin:

- The learner's file is never imported or executed. Only its ``import`` and ``from``
  statements are read, with ``ast``, to learn what each local name refers to.
- Only packages on the allowlist are ever imported, lazily, on first use. A name that
  resolves anywhere else is refused before anything is imported.
- Nothing is imported from the working directory (the process runs isolated), so a
  solution file named ``torch.py`` can never shadow the real package.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import json
import keyword
import os
import re
import sys
from types import ModuleType
from typing import IO, Any

#: Third-party packages a solution may use, by top-level name.
_PACKAGES = frozenset({"torch", "jax", "mlx", "numpy"})

#: Standard library modules that are safe to import and plausible in a solution.
_STDLIB = frozenset(
    {
        "abc",
        "array",
        "asyncio",
        "bisect",
        "cmath",
        "collections",
        "concurrent",
        "contextlib",
        "copy",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "queue",
        "random",
        "re",
        "statistics",
        "string",
        "struct",
        "threading",
        "time",
        "typing",
        "weakref",
    }
)

#: The one part of this package that solutions build on.
_KIT = "learnai.kit"

#: Longest docstring returned, in characters; longer ones are cut at a line break.
MAX_DOC_CHARS = 16000
#: Longest signature returned, in characters.
MAX_SIGNATURE_CHARS = 2000
#: Longest expression accepted: characters, and dotted parts.
MAX_EXPR_CHARS = 200
MAX_EXPR_PARTS = 12

#: How many lines a parenthesized import may span when the file does not parse.
_MAX_IMPORT_LINES = 50
#: Line starts that end an unfinished import when the file does not parse.
_STATEMENT_STARTS = ("import ", "from ", "def ", "class ", "@", "async ")


def allowed(dotted: str) -> bool:
    """Whether ``dotted`` names something inside an allowlisted package."""
    root = dotted.split(".", 1)[0]
    if root in _PACKAGES or root in _STDLIB:
        return True
    return dotted == _KIT or dotted.startswith(_KIT + ".")


def _bindings(node: ast.stmt) -> list[tuple[str, str]]:
    """The ``(local name, dotted target)`` pairs one import statement binds."""
    if isinstance(node, ast.Import):
        # `import a.b` binds `a`; `import a.b as c` binds `c` to `a.b`.
        return [
            (alias.asname, alias.name) if alias.asname else (alias.name.split(".", 1)[0],) * 2
            for alias in node.names
        ]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [
            (alias.asname or alias.name, f"{node.module}.{alias.name}")
            for alias in node.names
            if alias.name != "*"
        ]
    return []


def _import_statements(source: str) -> list[ast.stmt]:
    """Import statements recovered line by line from a file that does not parse."""
    found: list[ast.stmt] = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        statement = lines[index].strip()
        index += 1
        if not statement.startswith(("import ", "from ")):
            continue
        # A parenthesized or backslash-continued import spans several lines.
        taken = 1
        while (
            statement.count("(") > statement.count(")") or statement.endswith("\\")
        ) and index < len(lines):
            following = lines[index].strip()
            # An import left open while typing must not swallow the statements after it.
            if taken >= _MAX_IMPORT_LINES or following.startswith(_STATEMENT_STARTS):
                break
            statement = statement.removesuffix("\\") + " " + following
            index += 1
            taken += 1
        try:
            found.extend(ast.parse(statement).body)
        except SyntaxError, ValueError, RecursionError, MemoryError:
            continue
    return found


@functools.lru_cache(maxsize=16)
def import_aliases(source: str) -> dict[str, str]:
    """What each imported name in ``source`` refers to, as a dotted path.

    Reads only ``import`` and ``from`` statements, wherever they are in the file, and
    executes nothing. A file that does not parse (the learner is in the middle of
    typing) still yields the imports that can be read on their own. Relative and star
    imports are ignored. A later binding of a name replaces an earlier one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError, ValueError, RecursionError, MemoryError:
        nodes = _import_statements(source)
    else:
        nodes = sorted(
            (n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
            key=lambda n: (n.lineno, n.col_offset),
        )
    aliases: dict[str, str] = {}
    for node in nodes:
        aliases.update(_bindings(node))
    return aliases


def _parts(expr: str) -> list[str] | None:
    """The dotted parts of ``expr``, or ``None`` when it is not a plain dotted name."""
    if not expr or len(expr) > MAX_EXPR_CHARS:
        return None
    parts = [part.strip() for part in expr.split(".")]
    if len(parts) > MAX_EXPR_PARTS:
        return None
    if not all(part.isidentifier() and not keyword.iskeyword(part) for part in parts):
        return None
    return parts


class _Refusal(Exception):
    """The lookup ends without a result; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


#: Packages whose import failed, with the reason, so they are not retried every time.
_unavailable: dict[str, str] = {}


def _import(name: str) -> ModuleType:
    if name in _unavailable:
        raise _Refusal(_unavailable[name])
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        # The package itself is absent (MLX on Linux, JAX not installed), as opposed
        # to the package being present and failing on one of its own imports.
        missing = error.name is not None and (
            error.name == name or name.startswith(error.name + ".")
        )
        reason = "not_installed" if missing else "import_failed"
    except Exception:  # noqa: BLE001 - a broken install must not take the helper down
        reason = "import_failed"
    if "." not in name:
        _unavailable[name] = reason
    raise _Refusal(reason)


def _resolve(dotted: list[str]) -> Any:
    """Imports the root package of ``dotted`` and walks the rest with ``getattr``.

    The caller has checked that the whole path is on the allowlist.
    """
    obj: Any = _import(dotted[0])
    for index, part in enumerate(dotted[1:], start=1):
        try:
            obj = getattr(obj, part)
            continue
        except AttributeError:
            pass
        except Exception as error:  # a lazy attribute that fails to load
            raise _Refusal("import_failed") from error
        # A submodule nothing has imported yet (`mlx.nn`, `concurrent.futures`).
        if not (inspect.ismodule(obj) and hasattr(obj, "__path__")):
            raise _Refusal("no_attribute")
        try:
            obj = _import(".".join(dotted[: index + 1]))
        except _Refusal as refusal:
            if refusal.reason == "not_installed":
                raise _Refusal("no_attribute") from None
            raise
    return obj


def _kind(obj: Any) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if inspect.isroutine(obj) or callable(obj):
        return "function"
    if isinstance(obj, property) or inspect.isdatadescriptor(obj):
        return "property"
    return "value"


_ROLE = re.compile(r":[\w:+-]+:`(~?)([^`<]+?)(?:\s*<[^`>]*>)?`")
_LITERAL = re.compile(r"``([^`\n]+)``")
_BLANK_RUN = re.compile(r"\n{3,}")


def clean_doc(doc: str) -> tuple[str, bool]:
    """Tidies a docstring for display and caps its size.

    Light touch only: trailing spaces and runs of blank lines go, reStructuredText
    roles and double-backtick literals become single-backtick code. Everything else
    (sections, directives, tables) is left as the author wrote it. Returns the text
    and whether it was cut.
    """
    text = "\n".join(line.rstrip() for line in doc.strip().splitlines())
    # :func:`~torch.nn.functional.softmax` displays as `softmax`, as Sphinx renders it.
    text = _ROLE.sub(lambda m: f"`{m[2].rsplit('.', 1)[-1] if m[1] else m[2]}`", text)
    text = _LITERAL.sub(r"`\1`", text)
    text = _BLANK_RUN.sub("\n\n", text)
    if len(text) <= MAX_DOC_CHARS:
        return text, False
    cut = text.rfind("\n", 0, MAX_DOC_CHARS)
    return text[: cut if cut > 0 else MAX_DOC_CHARS].rstrip(), True


def _leading_signature(doc: str, name: str) -> tuple[str, str] | None:
    """Splits ``name(...) -> T`` off the top of a docstring, as builtins write it.

    Functions implemented in C have no introspectable signature; by convention their
    docstring starts with one, sometimes wrapped over several lines and sometimes
    with part of the module path in front (``linalg.svd(A, ...)``).
    """
    if not re.match(rf"(?:\w+\.)*{re.escape(name)}\(", doc):
        return None
    depth = 0
    for index, char in enumerate(doc):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                break
    else:
        return None
    end = doc.find("\n", index)
    end = len(doc) if end == -1 else end
    signature = " ".join(doc[:end].split()).rstrip(":").rstrip()
    return signature, doc[end:].lstrip("\n")


def _describe(obj: Any, dotted: str) -> dict[str, Any]:
    kind = _kind(obj)
    short = dotted.rsplit(".", 1)[-1]
    # A plain value (a number, a dtype instance) would only report its type's docs.
    doc = inspect.getdoc(obj) if kind != "value" else None

    signature = None
    if kind in ("class", "function"):
        try:
            signature = f"{short}{inspect.signature(obj)}"
        except TypeError, ValueError:
            signature = None
        if doc:
            for candidate in {short, getattr(obj, "__name__", short)}:
                split = _leading_signature(doc, candidate)
                if split is not None:
                    # The written signature is the better one when introspection has
                    # nothing or only `(*args, **kwargs)`.
                    if signature is None or signature.endswith("(*args, **kwargs)"):
                        signature = split[0]
                    doc = split[1]
                    break
    if signature is not None and len(signature) > MAX_SIGNATURE_CHARS:
        signature = signature[: MAX_SIGNATURE_CHARS - 1] + "…"

    truncated = False
    if doc:
        doc, truncated = clean_doc(doc)
    return {
        "found": True,
        # The public path the name was reached by, not where the object is implemented
        # (`torch.nn.functional.softmax`, not `torch._C._nn.softmax`).
        "name": dotted,
        "kind": kind,
        "signature": signature,
        "doc": doc or None,
        "truncated": truncated,
        "reason": None,
    }


def _not_found(reason: str) -> dict[str, Any]:
    return {
        "found": False,
        "name": None,
        "kind": None,
        "signature": None,
        "doc": None,
        "truncated": False,
        "reason": reason,
    }


def lookup(source: str, expr: str, member: str | None = None) -> dict[str, Any]:
    """Documentation for ``expr`` as written in a file whose text is ``source``.

    ``expr`` is a dotted name: either as the file spells it (``F.softmax``, resolved
    through the file's imports) or fully qualified (``torch.linalg.svd``). With
    ``member``, the lookup is for that attribute of ``expr``, which is how a
    completion item is described: the expression before the dot, plus the item's
    label. An empty ``expr`` with a ``member`` looks the member up as a bare name.

    Returns the response object of the wire protocol. ``found`` is false, with a
    ``reason``, when the name is not a dotted name (``not_a_name``), does not start
    at an import of the file or an allowlisted package (``unknown_name``), resolves
    outside the allowlist (``not_allowed``), needs a package that is not installed
    (``not_installed``) or fails to import (``import_failed``), or has no such
    attribute (``no_attribute``).
    """
    if member is not None:
        expr = f"{expr}.{member}" if expr.strip() else member
    parts = _parts(expr)
    if parts is None:
        return _not_found("not_a_name")
    target = import_aliases(source).get(parts[0])
    if target is not None:
        dotted = [*target.split("."), *parts[1:]]
    elif allowed(expr):
        dotted = parts
    else:
        # A local variable, a builtin, or a name this helper has no business guessing.
        return _not_found("unknown_name")
    path = ".".join(dotted)
    if not allowed(path):
        return _not_found("not_allowed")
    try:
        obj = _resolve(dotted)
    except _Refusal as refusal:
        return _not_found(refusal.reason)
    return _describe(obj, path)


def answer(line: str) -> dict[str, Any]:
    """The response to one request line. Never raises."""
    request_id = None
    try:
        request = json.loads(line)
        request_id = request.get("id")
        source, expr, member = request.get("source", ""), request["expr"], request.get("member")
        if not (
            isinstance(source, str) and isinstance(expr, str) and isinstance(member, str | None)
        ):
            raise TypeError("source, expr, and member must be strings")
        response = lookup(source, expr, member)
    except Exception as error:  # noqa: BLE001 - one bad request must not end the process
        response = _not_found("bad_request")
        response["message"] = f"{type(error).__name__}: {error}"
    response["id"] = request_id
    return response


def serve(requests: IO[str], responses: IO[str]) -> None:
    """Answers requests until ``requests`` reaches end of input."""
    for line in requests:
        if not line.strip():
            continue
        responses.write(json.dumps(answer(line)) + "\n")
        responses.flush()


def main() -> None:
    """Entry point of ``python -m learnai.editor.docs``."""
    # The working directory must never be importable, whatever started the process.
    here = os.getcwd()
    sys.path[:] = [entry for entry in sys.path if entry and os.path.abspath(entry) != here]
    # Libraries print on import now and then. Keep the protocol on a private copy of
    # stdout and send everything else to stderr.
    responses = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    serve(sys.stdin, responses)


if __name__ == "__main__":
    main()
