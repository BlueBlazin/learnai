"""The runtime documentation helper behind editor intelligence (ARCHITECTURE 4.4)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from learnai.editor import docs

TORCH_FILE = "import torch\nimport torch.nn.functional as F\nfrom torch import nn, linalg as LA\n"


def test_import_aliases_cover_every_import_form() -> None:
    source = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "import os.path\n"
        "from torch import nn, linalg as LA\n"
        "from . import sibling\n"
        "from .pkg import thing\n"
        "from math import *\n"
        "def solve():\n"
        "    import numpy as np\n"
        "    return np\n"
    )
    assert docs.import_aliases(source) == {
        "torch": "torch",
        "F": "torch.nn.functional",
        "os": "os",
        "nn": "torch.nn",
        "LA": "torch.linalg",
        "np": "numpy",
    }


def test_a_later_import_replaces_an_earlier_binding() -> None:
    source = "import numpy as xp\ntry:\n    import jax.numpy as xp\nexcept ImportError:\n    pass\n"
    assert docs.import_aliases(source) == {"xp": "jax.numpy"}


def test_imports_survive_a_file_that_does_not_parse() -> None:
    source = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from torch.nn import (\n"
        "    Linear,\n"
        "    Module as Base,\n"
        ")\n"
        "from torch import \\\n"
        "    linalg\n"
        "import\n"
        "from broken import (\n"
        "def softmax(x):\n"
        "    return F.softmax(x,\n"
        "    import numpy as np\n"
    )
    with pytest.raises(SyntaxError):
        compile(source, "<solution>", "exec")
    assert docs.import_aliases(source) == {
        "torch": "torch",
        "F": "torch.nn.functional",
        "Linear": "torch.nn.Linear",
        "Base": "torch.nn.Module",
        "linalg": "torch.linalg",
        "np": "numpy",
    }


def test_runtime_only_docstring_is_found_through_an_alias() -> None:
    result = docs.lookup(TORCH_FILE, "F.scaled_dot_product_attention")
    assert result["found"] is True
    assert result["name"] == "torch.nn.functional.scaled_dot_product_attention"
    assert result["kind"] == "function"
    assert result["signature"].startswith("scaled_dot_product_attention(query, key, value")
    assert result["signature"].endswith("-> Tensor")
    assert "\n" not in result["signature"]
    assert result["doc"].startswith("Computes scaled dot product attention")
    assert result["reason"] is None


def test_fully_qualified_names_need_no_import_in_the_file() -> None:
    result = docs.lookup("", "torch.linalg.svd")
    assert result["found"] is True
    assert result["name"] == "torch.linalg.svd"
    assert result["signature"].startswith("linalg.svd(A, full_matrices=True")
    assert result["doc"].startswith("Computes the singular value decomposition")


def test_from_import_aliases_resolve() -> None:
    result = docs.lookup(TORCH_FILE, "LA.svd")
    assert result["name"] == "torch.linalg.svd"
    result = docs.lookup(TORCH_FILE, "nn.Linear")
    assert (result["name"], result["kind"]) == ("torch.nn.Linear", "class")
    assert result["signature"].startswith("Linear(in_features: int, out_features: int")
    assert "affine linear transformation" in result["doc"]


def test_member_form_describes_a_completion_item() -> None:
    by_member = docs.lookup(TORCH_FILE, "F", "scaled_dot_product_attention")
    assert by_member == docs.lookup(TORCH_FILE, "F.scaled_dot_product_attention")
    nested = docs.lookup(TORCH_FILE, "torch.linalg", "svd")
    assert nested["name"] == "torch.linalg.svd"
    # A bare name being completed has nothing before the dot.
    bare = docs.lookup(TORCH_FILE, "", "F")
    assert (bare["name"], bare["kind"]) == ("torch.nn.functional", "module")
    assert docs.lookup(TORCH_FILE, "F", "no_such_function")["reason"] == "no_attribute"
    assert docs.lookup(TORCH_FILE, "F", "soft max")["reason"] == "not_a_name"


def test_introspected_signature_is_used_when_there_is_one() -> None:
    result = docs.lookup("import numpy as np", "np.linalg.norm")
    assert result["signature"].startswith("norm(x, ord=None, axis=None")
    result = docs.lookup("import math", "math.sqrt")
    assert result["found"] and result["signature"] == "sqrt(x, /)"


def test_plain_values_do_not_borrow_their_types_docstring() -> None:
    result = docs.lookup("import math", "math.pi")
    assert result == {
        "found": True,
        "name": "math.pi",
        "kind": "value",
        "signature": None,
        "doc": None,
        "truncated": False,
        "reason": None,
    }


def test_the_kit_is_allowed_and_the_rest_of_the_package_is_not() -> None:
    result = docs.lookup("from learnai.kit import vocab", "vocab")
    assert (result["found"], result["name"], result["kind"]) == (
        True,
        "learnai.kit.vocab",
        "module",
    )
    assert docs.lookup("import learnai.kit", "learnai.kit.vocab")["found"] is True
    for source, expr in [
        ("import learnai", "learnai"),
        ("from learnai import harness", "harness"),
        ("from learnai.judge import registry", "registry.Case"),
    ]:
        assert docs.lookup(source, expr)["reason"] == "not_allowed", (source, expr)


@pytest.mark.parametrize(
    ("source", "expr"),
    [
        ("import os", "os.system"),
        ("import subprocess as sp", "sp.run"),
        ("from pathlib import Path", "Path.unlink"),
        ("import antigravity", "antigravity"),
        ("import solution_helpers as h", "h.run"),
    ],
)
def test_names_outside_the_allowlist_are_refused_without_importing(
    source: str, expr: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))
    result = docs.lookup(source, expr)
    assert (result["found"], result["reason"]) == (False, "not_allowed")
    assert imported == []


def test_fully_qualified_names_outside_the_allowlist_are_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))
    # No import in the file binds these, so they are locals or builtins to the helper.
    for expr in ["os.system", "x.sum", "print", "self.weight"]:
        assert docs.lookup("import torch\nx = torch.zeros(3)\n", expr)["reason"] == "unknown_name"
    assert imported == []


@pytest.mark.parametrize(
    "expr", ["", "F.", ".softmax", "F.softmax(x)", "a b", "F.class", "x" * 300]
)
def test_only_plain_dotted_names_are_looked_up(expr: str) -> None:
    assert docs.lookup(TORCH_FILE, expr)["reason"] == "not_a_name"


def test_a_missing_package_is_reported_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []

    def missing(name: str) -> None:
        attempts.append(name)
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    monkeypatch.setattr(docs, "_unavailable", {})
    source = "import mlx.core as mx\nimport jax\n"
    assert docs.lookup(source, "mx.exp")["reason"] == "not_installed"
    assert docs.lookup(source, "mx.softmax")["reason"] == "not_installed"
    assert docs.lookup(source, "jax.jit")["reason"] == "not_installed"
    assert attempts == ["mlx", "jax"]


def test_a_package_that_fails_to_import_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(name: str) -> None:
        raise RuntimeError("the native library could not be loaded")

    monkeypatch.setattr(importlib, "import_module", broken)
    monkeypatch.setattr(docs, "_unavailable", {})
    assert docs.lookup("import jax", "jax.jit")["reason"] == "import_failed"


def test_docstrings_are_tidied_lightly_and_capped() -> None:
    text, truncated = docs.clean_doc(
        "  Applies :func:`~torch.nn.functional.softmax` to :attr:`input`.   \n\n\n\n"
        "See :class:`torch.Tensor` and :ref:`the notes <broadcasting-semantics>`.\n"
        "Use ``dim=-1`` for the last axis.\n\n.. note::\n    Kept as written.\n"
    )
    assert not truncated
    assert text == (
        "Applies `softmax` to `input`.\n\n"
        "See `torch.Tensor` and `the notes`.\n"
        "Use `dim=-1` for the last axis.\n\n.. note::\n    Kept as written."
    )

    long = "\n".join(f"line {index} " + "x" * 60 for index in range(1000))
    text, truncated = docs.clean_doc(long)
    assert truncated
    assert len(text) <= docs.MAX_DOC_CHARS
    assert text.splitlines()[-1].endswith("x" * 60), "cut at a line break"


def test_bad_requests_get_an_answer() -> None:
    assert docs.answer("not json") == {**docs._not_found("bad_request"), "id": None} | {
        "message": docs.answer("not json")["message"]
    }
    response = docs.answer(json.dumps({"id": 7, "source": 3, "expr": "math.pi"}))
    assert (response["id"], response["reason"]) == (7, "bad_request")
    response = docs.answer(json.dumps({"id": 8}))
    assert (response["id"], response["reason"]) == (8, "bad_request")


def run_helper(cwd: Path, requests: list[dict]) -> list[dict]:
    """Runs the helper the way core does and returns one response per request."""
    done = subprocess.run(
        [sys.executable, "-I", "-m", "learnai.editor.docs"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
        check=True,
    )
    return [json.loads(line) for line in done.stdout.splitlines()]


def test_the_process_never_imports_the_learners_file(tmp_path: Path) -> None:
    # The learner's solution is literally named after its framework, and would run if
    # anything imported `torch` (or `numpy`) from the working directory.
    marker = tmp_path / "executed"
    booby_trap = f"open({str(marker)!r}, 'w').close()\nraise SystemExit(3)\n"
    for name in ["torch.py", "numpy.py", "math.py", "sitecustomize.py"]:
        (tmp_path / name).write_text(booby_trap)

    source = "import torch\nimport torch.nn.functional as F\nimport math\nx = F.softmax(\n"
    responses = run_helper(
        tmp_path,
        [
            {"id": 1, "source": source, "expr": "F.scaled_dot_product_attention"},
            {"id": 2, "source": source, "expr": "math", "member": "sqrt"},
            {"id": 3, "source": source, "expr": "os.system"},
            {"id": 4, "source": "import torch", "expr": "torch.linalg.svd"},
        ],
    )
    assert not marker.exists()
    assert [response["id"] for response in responses] == [1, 2, 3, 4]
    assert responses[0]["found"] and responses[0]["doc"].startswith("Computes scaled dot product")
    assert responses[1]["signature"] == "sqrt(x, /)"
    assert responses[2]["reason"] == "unknown_name"
    assert responses[3]["doc"].startswith("Computes the singular value decomposition")
