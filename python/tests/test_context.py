"""The Context object, exercised in process against the fixture problems."""

import types

import numpy as np
import pytest
from conftest import fixture_problem

from whetstone.judge import CheckFailure, frameworks
from whetstone.judge.context import make_context
from whetstone.judge.registry import Case, load_cases
from whetstone.kit import TinyLM
from whetstone.problem import load_problem


def context_for(problem_name, framework, solution=None, case_id="unit"):
    problem = load_problem(fixture_problem(problem_name))
    try:
        frameworks.get(framework)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
    case = Case(id=case_id, name=case_id, public=True, fn=lambda ctx: None)
    return make_context(problem, solution, framework, case)


def module(**functions):
    return types.SimpleNamespace(**functions)


@pytest.mark.parametrize("framework", ["torch", "jax", "mlx"])
def test_check_compares_any_framework_with_the_torch_oracle(framework):
    ctx = context_for("unit-normalize", framework)
    ctx.solution = ctx.reference  # the same-framework reference plays the solution
    x = ctx.randn(3, 5)
    got = ctx.check("normalize", x, axis=-1)
    assert isinstance(got, np.ndarray) and got.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(ctx.baseline("normalize", x), ctx.expected("normalize", x), 1e-5)


def test_check_failure_names_the_call_and_its_inputs():
    ctx = context_for("unit-normalize", "torch", module(normalize=lambda x, axis=-1: x * 2))
    with pytest.raises(CheckFailure) as failure:
        ctx.check("normalize", np.array([[3.0, 4.0]], dtype=np.float32), axis=-1)
    assert failure.value.message.startswith("normalize: 2 of 2 values differ")
    detail = failure.value.detail.splitlines()
    assert detail[0] == "call: normalize(x, axis)"
    assert detail[1] == "  x = float32 array, shape (1, 2): [[3., 4.]]"
    assert detail[2] == "  axis = -1"
    assert "expected = [[0.6, 0.8]]" in detail


def test_check_enforces_tensor_type_and_dtype():
    ctx = context_for("unit-normalize", "torch", module(normalize=lambda x, axis=-1: x.numpy()))
    with pytest.raises(CheckFailure) as failure:
        ctx.check("normalize", ctx.randn(2, 2))
    assert failure.value.message == (
        "`normalize` returned a numpy.ndarray, expected a torch.Tensor"
    )
    ctx = context_for("unit-normalize", "torch")
    ctx.solution = module(normalize=lambda x, axis=-1: ctx.reference.normalize(x, axis).double())
    with pytest.raises(CheckFailure, match="got dtype float64, which is not the expected dtype"):
        ctx.check("normalize", ctx.randn(2, 2))
    ctx.check("normalize", ctx.randn(2, 2), dtype=False)
    ctx.solution = module(normalize=lambda x, axis=-1: None)
    with pytest.raises(CheckFailure, match="got None"):
        ctx.check("normalize", ctx.randn(2, 2))


def test_seeds_are_per_case_and_identical_across_frameworks():
    a = context_for("unit-normalize", "torch", case_id="one")
    b = context_for("unit-normalize", "jax", case_id="one")
    c = context_for("unit-normalize", "torch", case_id="two")
    assert a.seed == b.seed != c.seed
    np.testing.assert_array_equal(a.randn(4, 4), b.randn(4, 4))
    assert not np.array_equal(a.randn(4, 4), c.randn(4, 4))
    np.testing.assert_array_equal(a.child_rng(3).random(5), b.child_rng(3).random(5))


def test_input_helpers_produce_house_dtypes():
    ctx = context_for("sum-list", "python")
    x = ctx.randn(2, 3, scale=5.0, loc=1.0)
    assert x.dtype == np.float32 and x.shape == (2, 3)
    u = ctx.uniform(1000, low=2.0, high=3.0)
    assert u.dtype == np.float32 and 2.0 <= u.min() and u.max() < 3.0
    ids = ctx.randint(0, 10, (4, 5))
    assert ids.dtype == np.int64 and ids.shape == (4, 5) and 0 <= ids.min() and ids.max() < 10


def test_python_framework_passes_lists():
    seen = {}

    def total(xs):
        seen["type"] = type(xs)
        return sum(xs)

    ctx = context_for("sum-list", "python", module(total=total))
    assert ctx.call("total", np.array([1, 2, 3])) == 6
    assert seen["type"] is list
    assert ctx.expected("total", [4, 5]) == 9


def test_reference_calls_get_their_own_kit_counters():
    ctx = context_for("greedy-decode", "torch")
    ctx.solution = ctx.reference
    lm = TinyLM(ctx.fw, seed=3)
    got = ctx.call("greedy_decode", lm, [0, 17], 5)
    used = lm.counters.snapshot()
    assert used == {"forward_calls": 5, "tokens_processed": 6}
    assert ctx.expected("greedy_decode", lm, [0, 17], 5) == got
    assert ctx.baseline("greedy_decode", lm, [0, 17], 5) == got
    assert lm.counters.snapshot() == used  # the oracle ran on a twin


def test_metrics_replace_by_name_and_drop_non_finite_values():
    ctx = context_for("sum-list", "python")
    ctx.metric("tokens", 10, "tokens", 12)
    ctx.metric("tokens", 11, "tokens", 12)
    ctx.metric("ratio", float("nan"))
    assert ctx.metrics == [{"name": "tokens", "value": 11, "unit": "tokens", "limit": 12}]


def test_check_speed_reports_the_slowdown():
    ctx = context_for("pairwise-distances", "torch")
    ctx.solution = ctx.reference
    ratio = ctx.check_speed("pairwise_sq_dists", ctx.randn(32, 8), ctx.randn(32, 8), max_ratio=20)
    assert 0.2 < ratio < 5.0
    assert ctx.metrics[0]["name"] == "slowdown" and ctx.metrics[0]["limit"] == 20


def test_deferred_cleanups_run_in_reverse_and_never_raise():
    ctx = context_for("sum-list", "python")
    order = []
    ctx.defer(lambda: order.append("first"))
    ctx.defer(lambda: 1 / 0)
    ctx.defer(lambda: order.append("last"))
    sim = ctx.simclock()
    ctx.close()
    assert order == ["last", "first"]
    with pytest.raises(RuntimeError):
        sim.run()


def test_case_registration_rules(tmp_path):
    tests = tmp_path / "tests.py"
    tests.write_text(
        "from whetstone.judge import case\n"
        "@case('first', public=True)\n"
        "def b(ctx): pass\n"
        "@case('second', id='custom-id', time_limit_s=2.5, frameworks=['torch'])\n"
        "def a(ctx): pass\n"
    )
    first, second = load_cases(tests)
    assert (first.id, first.name, first.public, first.time_limit_s) == ("b", "first", True, None)
    assert (second.id, second.public, second.time_limit_s) == ("custom-id", False, 2.5)
    assert second.applies_to("torch") and not second.applies_to("mlx")
    tests.write_text(
        "from whetstone.judge import case\n"
        "@case('x')\ndef same(ctx): pass\n"
        "@case('y', id='same')\ndef other(ctx): pass\n"
    )
    with pytest.raises(ValueError, match="duplicate case id 'same'"):
        load_cases(tests)
    tests.write_text("x = 1\n")
    with pytest.raises(ValueError, match="defines no @case"):
        load_cases(tests)
