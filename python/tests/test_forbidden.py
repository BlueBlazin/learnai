"""Forbidden-call enforcement: the guard itself, then the harness and the gate around it."""

import math
import sys
import textwrap
import types

import pytest
from conftest import SOLUTIONS, fixture_problem, run_harness

from learnai.judge import forbidden
from learnai.judge.forbidden import ForbiddenCall, Guard


@pytest.fixture
def lib(monkeypatch):
    """A stand-in framework: a module, an alias of its function elsewhere, and a class."""
    module = types.ModuleType("ws_fake_lib")
    exec(  # noqa: S102 - builds the fixture module
        textwrap.dedent(
            """
            def builtin(x):
                return x * 2

            def uses_builtin(x):      # a library function built on the forbidden one
                return builtin(x) + 1

            constant = 3

            class Tensor:
                def __init__(self, value=0):
                    self.value = value
                def double(self):
                    return self.value * 2
                @staticmethod
                def make(value):
                    return Tensor(value)
                @classmethod
                def zero(cls):
                    return cls(0)

            class Layer:
                def __init__(self, scale):
                    self.scale = scale
            """
        ),
        module.__dict__,
    )
    alias = types.ModuleType("ws_fake_lib_private")
    alias.reexported = module.builtin
    monkeypatch.setitem(sys.modules, "ws_fake_lib", module)
    monkeypatch.setitem(sys.modules, "ws_fake_lib_private", alias)
    return module


def load(tmp_path, source: str, name: str = "solution.py"):
    """Compile ``source`` as if it were the solution file at ``tmp_path / name``."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    return path, namespace


@pytest.fixture
def guarded(lib, tmp_path):
    path, solution = load(
        tmp_path,
        """
        import ws_fake_lib
        import ws_fake_lib_private
        from ws_fake_lib import builtin as renamed      # bound before the guard is installed

        def direct(x): return ws_fake_lib.builtin(x)
        def from_import(x): return renamed(x)
        def private_alias(x): return ws_fake_lib_private.reexported(x)
        def through_library(x): return ws_fake_lib.uses_builtin(x)
        def method(x): return ws_fake_lib.Tensor(x).double()
        def static(x): return ws_fake_lib.Tensor.make(x)
        def klass(): return ws_fake_lib.Tensor.zero()
        def layer(): return ws_fake_lib.Layer(2)
        def allowed(x): return ws_fake_lib.Tensor(x).value
        def swallows(x):
            try:
                return ws_fake_lib.builtin(x)
            except Exception:
                return x * 2
        """,
    )
    names = [
        "ws_fake_lib.builtin",
        "ws_fake_lib.Tensor.double",
        "ws_fake_lib.Tensor.make",
        "ws_fake_lib.Tensor.zero",
        "ws_fake_lib.Layer",
    ]
    guard = Guard(names, path)
    guard.install()
    # What the worker does: the solution is (re)imported after the guard is in place.
    _, solution = load(tmp_path, path.read_text())
    yield guard, solution, lib
    guard.uninstall()


@pytest.mark.parametrize(
    ("call", "name"),
    [
        ("direct", "ws_fake_lib.builtin"),
        ("from_import", "ws_fake_lib.builtin"),
        ("private_alias", "ws_fake_lib.builtin"),
        ("through_library", "ws_fake_lib.builtin"),
        ("method", "ws_fake_lib.Tensor.double"),
        ("static", "ws_fake_lib.Tensor.make"),
        ("klass", "ws_fake_lib.Tensor.zero"),
        ("layer", "ws_fake_lib.Layer"),
    ],
)
def test_the_solution_cannot_reach_a_forbidden_name(guarded, call, name):
    guard, solution, _ = guarded
    args = () if call in ("klass", "layer") else (4,)
    with pytest.raises(ForbiddenCall) as info:
        solution[call](*args)
    message = info.value.message
    assert message.startswith(f"`{name}` is not allowed in this problem (called at solution.py ")
    assert message.endswith(f", in {call})")
    (violation,) = guard.take()
    assert violation.name == name and violation.function == call
    assert guard.take() == []
    assert "ws_fake_lib.Tensor.double" in info.value.detail  # the whole rule is listed


def test_everyone_else_still_can(guarded):
    # The oracle, the references, the tests, and the library itself are not the
    # solution file, so for them the originals run.
    guard, solution, lib = guarded
    assert lib.builtin(4) == 8
    assert lib.uses_builtin(4) == 9
    assert sys.modules["ws_fake_lib_private"].reexported(4) == 8
    assert lib.Tensor(5).double() == 10
    assert lib.Tensor.make(5).value == 5 and lib.Tensor.zero().value == 0
    assert isinstance(lib.Layer(2), lib.Layer)
    assert solution["allowed"](7) == 7  # the solution may use everything else
    assert guard.take() == []


def test_a_swallowed_violation_is_still_recorded(guarded):
    guard, solution, _ = guarded
    assert solution["swallows"](4) == 8
    (violation,) = guard.take()
    assert violation.name == "ws_fake_lib.builtin"


def test_uninstall_restores_every_attribute(lib, tmp_path):
    before = (lib.builtin, lib.Tensor.__dict__["double"], lib.Layer.__init__)
    private = sys.modules["ws_fake_lib_private"]
    guard = Guard(["ws_fake_lib.builtin", "ws_fake_lib.Tensor.double", "ws_fake_lib.Layer"], "x")
    guard.install()
    assert lib.builtin is not before[0] and private.reexported is lib.builtin
    guard.uninstall()
    assert (lib.builtin, lib.Tensor.__dict__["double"], lib.Layer.__init__) == before
    assert private.reexported is before[0]


def test_names_wait_until_their_package_is_loaded(lib, tmp_path, monkeypatch):
    late = types.ModuleType("ws_late_lib")
    late.shortcut = lambda x: x
    original = late.shortcut
    guard = Guard(["ws_late_lib.shortcut", "ws_fake_lib.builtin"], "x")
    try:
        guard.install()
        assert late.shortcut is original  # a framework that is not loaded costs nothing
        monkeypatch.setitem(sys.modules, "ws_late_lib", late)  # the solution imported it
        guard.install()
        assert late.shortcut is not original
        assert guard.unresolved == {}
    finally:
        guard.uninstall()


def test_names_that_cannot_be_guarded_are_reported(lib):
    problems = forbidden.check_names(
        [
            "ws_fake_lib.builtin",
            "ws_fake_lib.missing",
            "ws_fake_lib.Tensor.missing",
            "ws_fake_lib.constant",
            "ws_no_such_package.thing",
            "notdotted",
            "builtins.int.bit_length",
        ]
    )
    assert problems == {
        "ws_fake_lib.missing": "`missing` does not exist",
        "ws_fake_lib.Tensor.missing": "`missing` does not exist",
        "ws_fake_lib.constant": (
            "is not callable, so there is no call to guard (a dtype or a constant is only "
            "mentioned); list its identifier in `forbidden_source` instead"
        ),
        "ws_no_such_package.thing": "cannot import `ws_no_such_package`",
        "notdotted": "must be a dotted name such as `torch.softmax`",
    }
    assert lib.builtin(2) == 4 and not hasattr(lib.builtin, "__learnai_forbidden__")


def test_methods_of_immutable_types_are_found_in_the_source(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text(
        textwrap.dedent(
            """
            import math

            def f(n, values):
                a = math.bit_length if hasattr(math, "bit_length") else None   # a module: fine
                size = len(values)
                return (n + size).bit_length()
            """
        )
    )
    guard = Guard(["builtins.int.bit_length"], path)
    try:
        guard.install()
        assert guard.static == {"bit_length": "builtins.int.bit_length"} and not guard.unresolved
        violation = guard.scan_source()
        assert violation.message == (
            "`builtins.int.bit_length` is not allowed in this problem (used at solution.py line 7)"
        )
        path.write_text("import math\n\ndef f(n):\n    return math.floor(math.log2(n)) + 1\n")
        assert guard.scan_source() is None
        path.write_text("def f(:\n")  # not parseable: importing it reports that
        assert guard.scan_source() is None
    finally:
        guard.uninstall()
    assert (5).bit_length() == 3 and math.floor(2.5) == 2


def test_forbidding_a_kit_operation_guards_its_backends(tmp_path):
    # `learnai.kit.ops.softmax` only routes the call; a reviewer's solution passed a
    # problem by calling `learnai.kit.ops._torch.<name>` directly.
    assert forbidden.backend_names("learnai.kit.ops.softmax") == [
        f"learnai.kit.ops.{backend}.softmax" for backend in ("_torch", "_jax", "_mlx", "_numpy")
    ]
    assert forbidden.backend_names("learnai.kit.ops._numpy.softmax") == []
    assert forbidden.backend_names("learnai.kit.gdn.gdn_step_ref") == []
    assert forbidden.backend_names("torch.softmax") == []

    import numpy as np

    from learnai.kit import ops
    from learnai.kit.ops import _numpy

    path, _ = load(tmp_path, "x = 1")
    guard = Guard(["learnai.kit.ops.softmax"], path)
    original = _numpy.softmax
    try:
        guard.install()
        assert not guard.unresolved
        _, solution = load(
            tmp_path,
            """
            import importlib
            from learnai.kit import ops
            from learnai.kit.ops import _numpy

            def dispatcher(x): return ops.softmax(x)
            def backend(x): return _numpy.softmax(x)
            def late(x):
                module = importlib.import_module("learnai.kit.ops." + "_numpy")
                return getattr(module, "soft" + "max")(x)
            def other(x): return ops.log_softmax(x)
            """,
        )
        x = np.array([0.0, 1.0], np.float32)
        for call in ("dispatcher", "backend", "late"):
            with pytest.raises(ForbiddenCall) as info:
                solution[call](x)
            # The message names the operation the problem listed, not the backend.
            assert info.value.message.startswith(
                "`learnai.kit.ops.softmax` is not allowed in this problem (called at solution.py"
            ), call
            (violation,) = guard.take()
            assert violation.name == "learnai.kit.ops.softmax" and violation.function == call
        assert "_numpy" not in guard.rule_text()  # the rule shown is the list as written
        # Everyone else, and the solution's use of another operation, are untouched.
        np.testing.assert_allclose(ops.softmax(x).sum(), 1.0, rtol=1e-6)
        np.testing.assert_allclose(_numpy.softmax(x).sum(), 1.0, rtol=1e-6)
        np.testing.assert_allclose(np.exp(solution["other"](x)).sum(), 1.0, rtol=1e-6)
        assert guard.take() == []
    finally:
        guard.uninstall()
    assert _numpy.softmax is original


def test_a_backend_waits_for_its_framework_and_a_missing_one_is_not_an_error(monkeypatch):
    # Guarding `learnai.kit.ops._jax.softmax` imports JAX, so it waits until the run has
    # loaded JAX, like any name of a package that is not loaded.
    assert forbidden._needs("learnai.kit.ops._jax.softmax") == "jax"
    assert forbidden._needs("learnai.kit.ops._numpy.softmax") == "numpy"
    assert forbidden._needs("learnai.kit.ops.softmax") == "learnai"
    monkeypatch.setitem(forbidden._OPS_BACKENDS, "_ws_absent", "ws_absent_framework")
    guard = Guard(["learnai.kit.ops.softmax"], "x")
    try:
        guard.install()
        assert "learnai.kit.ops._ws_absent.softmax" in guard._pending  # its package is not loaded
        guard.install(import_roots=True)  # what `learnai validate` does
        assert guard._pending == [] and guard.unresolved == {}
    finally:
        guard.uninstall()
    # A listed name that does not exist is still reported, once, under its own name.
    assert forbidden.check_names(["learnai.kit.ops.ws_no_such_op"]) == {
        "learnai.kit.ops.ws_no_such_op": "`ws_no_such_op` does not exist"
    }


# -- through the harness -----------------------------------------------------------------


def solve(tmp_path, framework: str, source: str, **options):
    path = tmp_path / f"{framework}.py"
    path.write_text(textwrap.dedent(source))
    return run_harness(fixture_problem("unit-normalize"), path, framework, **options)


TORCH_SHORTCUTS = {
    "function": """
        import torch
        def normalize(x, axis=-1):
            return torch.nn.functional.normalize(x, dim=axis, eps=0.0)
        """,
    "from-import alias": """
        from torch.nn.functional import normalize as unit
        def normalize(x, axis=-1):
            return unit(x, dim=axis, eps=0.0)
        """,
    "tensor method": """
        def normalize(x, axis=-1):
            return x / x.norm(dim=axis, keepdim=True)
        """,
    "another function of the list": """
        import torch
        def normalize(x, axis=-1):
            return x / torch.linalg.norm(x, dim=axis, keepdim=True)
        """,
    "swallowed, with a correct fallback": """
        import torch
        def normalize(x, axis=-1):
            try:
                return torch.nn.functional.normalize(x, dim=axis, eps=0.0)
            except Exception:
                return x / torch.sqrt((x * x).sum(dim=axis, keepdim=True))
        """,
}


@pytest.mark.parametrize("how", list(TORCH_SHORTCUTS))
def test_a_torch_solution_that_takes_the_shortcut_fails_every_case(tmp_path, how):
    run = solve(tmp_path, "torch", TORCH_SHORTCUTS[how])
    assert run.end["status"] == "wrong_answer"
    assert run.statuses == {"worked_example": "fail", "hidden-1": "fail"}
    public, hidden = run.case("worked_example"), run.case("hidden-1")
    assert " is not allowed in this problem (called at torch.py line " in public["message"]
    assert "in normalize)" in public["message"]
    assert "torch.nn.functional.normalize" in public["detail"]
    # The hidden case names the call and the learner's line too, and has no detail.
    assert hidden["message"] == public["message"] and hidden["detail"] is None


def test_the_references_pass_under_the_same_guards():
    problem = fixture_problem("unit-normalize")
    for framework in ("torch", "jax", "mlx"):
        run = run_harness(problem, problem / "reference" / f"{framework}.py", framework)
        assert run.end["status"] == "accepted", (framework, run.end)


def test_jax_and_mlx_functions_are_guarded(tmp_path):
    run = solve(
        tmp_path, "jax",
        """
        import jax.numpy as jnp
        def normalize(x, axis=-1):
            return x / jnp.linalg.norm(x, axis=axis, keepdims=True)
        """,
    )  # fmt: skip
    assert run.end["status"] == "wrong_answer"
    assert run.case("worked_example")["message"].startswith(
        "`jax.numpy.linalg.norm` is not allowed in this problem (called at jax.py line 4"
    )
    run = solve(
        tmp_path, "mlx",
        """
        import mlx.core as mx
        def normalize(x, axis=-1):
            return x / mx.linalg.norm(x, axis=axis, keepdims=True)
        """,
    )  # fmt: skip
    assert run.end["status"] == "wrong_answer"
    assert run.case("hidden-1")["message"].startswith("`mlx.core.linalg.norm` is not allowed")


def test_borrowing_another_frameworks_builtin_is_caught(tmp_path):
    # A JAX solution that computes the answer with torch: torch names are guarded too.
    run = solve(
        tmp_path, "jax",
        """
        import jax.numpy as jnp
        import numpy as np
        import torch
        def normalize(x, axis=-1):
            y = torch.nn.functional.normalize(torch.from_numpy(np.asarray(x)), dim=axis, eps=0.0)
            return jnp.asarray(y.numpy())
        """,
    )  # fmt: skip
    assert run.end["status"] == "wrong_answer"
    assert "`torch.nn.functional.normalize` is not allowed" in run.case("hidden-1")["message"]


def test_a_method_of_an_immutable_type_is_found_statically(tmp_path):
    # mlx.core.array cannot be patched, so `.rsqrt` is looked for in the source.
    run = solve(
        tmp_path, "mlx",
        """
        import mlx.core as mx
        def normalize(x, axis=-1):
            return x * mx.sum(x * x, axis=axis, keepdims=True).rsqrt()
        """,
    )  # fmt: skip
    assert run.end["status"] == "wrong_answer"
    assert [e["event"] for e in run.events] == [
        "start", "case_start", "case_end", "case_start", "case_end", "end",
    ]  # fmt: skip
    for case in run.of("case_end"):
        assert case["status"] == "fail" and case["duration_ms"] == 0
        assert case["message"] == (
            "`mlx.core.array.rsqrt` is not allowed in this problem (used at mlx.py line 4)"
        )
    assert run.case("hidden-1")["detail"] is None
    # The module function of the same name is not on the list, and is not flagged.
    run = solve(
        tmp_path, "mlx",
        """
        import mlx.core as mx
        def normalize(x, axis=-1):
            return x * mx.rsqrt(mx.sum(x * x, axis=axis, keepdims=True))
        """,
    )  # fmt: skip
    assert run.end["status"] == "accepted", run.end


def test_a_forbidden_call_at_import_time_is_an_import_error(tmp_path):
    run = solve(
        tmp_path, "torch",
        """
        import torch
        UNIT = torch.nn.functional.normalize(torch.ones(2), dim=0)
        def normalize(x, axis=-1):
            return x
        """,
    )  # fmt: skip
    assert run.end["status"] == "import_error"
    assert run.end["message"] == (
        "the solution failed to import: `torch.nn.functional.normalize` is not allowed in "
        "this problem (called at torch.py line 3, in <module>)"
    )


def test_problems_without_a_list_are_untouched(sum_list):
    run = run_harness(sum_list, SOLUTIONS / "good.py")
    assert run.end["status"] == "accepted"
