import math

import numpy as np
import pytest

from learnai.judge import CheckFailure, check

F32 = np.float32


def fails(fn, *args, **kwargs) -> CheckFailure:
    with pytest.raises(CheckFailure) as info:
        fn(*args, **kwargs)
    return info.value


# -- numeric ------------------------------------------------------------------------


def test_allclose_accepts_float32_noise_and_rejects_real_errors():
    expected = np.linspace(0.1, 5.0, 40, dtype=F32)
    check.allclose(expected * F32(1 + 2e-6), expected)
    failure = fails(check.allclose, expected * F32(1.01), expected, label="softmax")
    assert failure.message.startswith("softmax: 40 of 40 values differ from the expected result")
    assert "largest error on the order of 1e-02" in failure.message


def test_shape_mismatch_is_stated_first_and_alone():
    failure = fails(check.allclose, np.zeros((4,)), np.zeros((4, 8)), label="softmax")
    assert failure.message == "softmax: got shape (4,), which is not the expected shape"
    assert failure.detail == "result\n  expected shape    (4, 8)\n  got shape         (4,)\n"


def test_detail_has_errors_worst_index_and_excerpts():
    expected = np.arange(60, dtype=F32).reshape(3, 20)
    got = expected.copy()
    got[1, 13] += 0.5
    got[0, 1] += 0.25
    failure = fails(check.allclose, got, expected, label="y", context="call: f(x)")
    assert failure.message == (
        "y: 2 of 60 values differ from the expected result "
        "(largest error on the order of 1e-01, at index [1, 13])"
    )
    assert failure.detail == (
        "call: f(x)\n"  # a test's own context is passed through as it is
        "\n"
        "result\n"
        "  shape            (3, 20), dtype float32\n"
        "  mismatched       2 of 60 values\n"
        "  max abs error    0.5 at index [1, 13]   (expected 33, got 33.5)\n"
        "  max rel error    0.25 at index [0, 1]   (expected 1, got 1.25)\n"
        "  tolerance        |got - expected| <= 1e-06 + 0.0001 * |expected|\n"
        "\n"
        "expected[1, 9:17]\n"
        "  [29., 30., 31., 32., 33., 34., 35., 36.]\n"
        "\n"
        "got[1, 9:17]\n"
        "  [29. , 30. , 31. , 32. , 33.5, 34. , 35. , 36. ]\n"
    )


def test_message_never_contains_expected_values():
    failure = fails(check.allclose, np.array([1.0, 2.0]), np.array([1.0, 7.125]))
    assert "7.125" not in failure.message
    assert "7.125" in failure.detail


def test_nan_and_infinity_handling():
    expected = np.array([0.5, -np.inf, np.nan], dtype=F32)
    check.allclose(np.array([0.5, -np.inf, np.nan], dtype=F32), expected)
    failure = fails(check.allclose, np.array([np.nan, -np.inf, np.nan], dtype=F32), expected)
    assert failure.message == (
        "output: got NaN where a number is expected (1 of 3 values, first at index [0])"
    )
    failure = fails(check.allclose, np.array([0.5, np.inf, np.nan], dtype=F32), expected)
    assert "1 of 3 values differ" in failure.message
    failure = fails(check.allclose, np.array([np.inf, 1.0]), np.array([3.0, 1.0]))
    assert failure.message.startswith("output: got infinity where a finite number is expected")
    failure = fails(check.allclose, np.array([-50.0]), np.array([-np.inf]))
    assert "1 of 1 values differ" in failure.message


def test_none_and_non_numeric_results():
    assert "got None" in fails(check.allclose, None, np.zeros(2), label="softmax").message
    assert "non-numeric" in fails(check.allclose, ["a", "b"], np.zeros(2)).message


def test_dtype_check_is_opt_in():
    check.allclose(np.ones(2, dtype=np.float64), np.ones(2, dtype=F32))
    failure = fails(check.allclose, np.ones(2, dtype=np.float64), np.ones(2, dtype=F32), dtype=True)
    assert failure.message == "output: got dtype float64, which is not the expected dtype"
    assert failure.detail == (
        "result\n"
        "  expected dtype    float32\n"
        "  got dtype         float64\n"
        "\n"
        "The result must have the same dtype as the reference's result.\n"
    )


def test_equal_on_arrays_ignores_integer_width():
    check.equal(np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int64))
    failure = fails(check.equal, np.array([1, 9, 3]), np.array([1, 2, 3]), label="tokens")
    assert failure.message == (
        "tokens: 1 of 3 values differ from the expected result (first at index [1])"
    )
    assert failure.detail == (
        "result\n"
        "  first difference    at index [1]   (expected 2, got 9)\n"
        "\n"
        "expected\n"
        "  [1, 2, 3]\n"
        "\n"
        "got\n"
        "  [1, 9, 3]\n"
    )
    assert "not the expected shape" in (
        fails(check.equal, np.array([1, 2]), np.array([1, 2, 3])).message
    )


def test_equal_on_python_values():
    check.equal([1, 2, 3], [1, 2, 3])
    check.equal("abc", "abc")
    failure = fails(check.equal, [5, 6, 7, 8], [5, 6, 9, 8], label="tokens")
    assert failure.message == (
        "tokens: differs from the expected result (first difference at position 2)"
    )
    assert failure.detail == (
        "result\n"
        "  first difference    position 2   (expected 9, got 7)\n"
        "\n"
        "expected[0:10]\n"
        "  [5, 6, 9, 8]\n"
        "\n"
        "got[0:10]\n"
        "  [5, 6, 7, 8]\n"
    )
    failure = fails(check.equal, [1], [1, 2])
    assert failure.message == "output: got 1 item, which is not the expected number"
    assert failure.detail == (
        "result\n  expected items    2\n  got items         1\n\nexpected\n  [1, 2]\n\ngot\n  [1]\n"
    )
    assert "got None" in fails(check.equal, None, [1]).message
    assert fails(check.equal, {"a": 2}, {"a": 1}).detail == (
        "expected\n  {'a': 1}\n\ngot\n  {'a': 2}\n"
    )
    assert fails(check.equal, "ab", "abc", context="why").detail == (
        "why\n\nexpected\n  'abc'\n\ngot\n  'ab'\n"
    )


def test_equal_treats_a_list_of_0d_arrays_as_a_list_of_scalars():
    # What a solution returns when it collects scalar tensors: [logits.argmax(), ...].
    tokens = [np.array(5, dtype=np.int32), np.array(6), np.int64(7)]
    check.equal(tokens, [5, 6, 7])
    check.equal([5, 6, 7], tokens)
    check.equal(tokens, np.array([5, 6, 7]))
    check.equal((np.array(True), np.array(False)), [True, False])
    check.equal([[np.array(1), np.array(2)], [np.array(3)]], [[1, 2], [3]])
    failure = fails(check.equal, tokens, [5, 9, 7], label="tokens")
    assert failure.message == (
        "tokens: differs from the expected result (first difference at position 1)"
    )
    # The detail shows numbers, not `array(6)`.
    assert "position 1   (expected 9, got 6)" in failure.detail
    assert "got[0:10]\n  [5, 6, 7]\n" in failure.detail and "array(" not in failure.detail
    assert fails(check.equal, tokens, [5, 6]).message == (
        "output: got 3 items, which is not the expected number"
    )


def test_equal_never_squeezes_an_array_inside_a_list_into_a_scalar():
    # A one-element tensor is not a scalar: `bool(array([5]) == 5)` used to let it pass.
    failure = fails(check.equal, [np.array([5]), np.array(6)], [5, 6], label="tokens")
    assert failure.message == "tokens[0]: got shape (1,), which is not the expected shape"
    failure = fails(check.equal, [np.array([1, 2]), np.array([3, 9])], [[1, 2], [3, 4]], label="t")
    assert failure.message == (
        "t[1]: 1 of 2 values differ from the expected result (first at index [1])"
    )
    check.equal([np.array([1, 2]), np.array([3, 4])], [np.array([1, 2]), [3, 4]])


def test_match_compares_a_list_of_scalar_tensors_as_one_sequence():
    oracle = [np.array(3, dtype=np.int64), np.array(1, dtype=np.int64)]
    check.match([np.array(3, dtype=np.int32), np.array(1, dtype=np.int32)], oracle)
    check.match([3, 1], oracle)
    failure = fails(check.match, [np.array(3), np.array(2)], oracle, label="tokens")
    assert failure.message == (
        "tokens: differs from the expected result (first difference at position 1)"
    )
    assert "got 1 item, which is not the expected number" in (
        fails(check.match, [np.array(3)], oracle).message
    )
    failure = fails(check.match, np.array([3, 1]), oracle)
    assert failure.message == "output: expected a list of scalars, got a ndarray"
    assert "expected a list of 2 scalars" in failure.detail
    floats = [np.array(0.5, dtype=F32), np.array(0.25, dtype=F32)]
    check.match([np.array(0.5000001, dtype=F32), 0.25], floats)
    assert "1 of 2 values differ" in fails(check.match, [0.5, 0.75], floats).message


def test_match_dispatches_on_structure():
    expected = (np.array([0.5, 0.25], dtype=F32), np.array([3, 1]), {"n": 2, "tokens": [1, 2]})
    check.match((np.array([0.5, 0.25], dtype=F32), np.array([3, 1]), {"n": 2, "tokens": [1, 2]}),
                expected)  # fmt: skip
    failure = fails(
        check.match,
        (np.array([0.5, 0.75], dtype=F32), np.array([3, 1]), {"n": 2, "tokens": [1, 2]}),
        expected,
        label="out",
    )
    assert failure.message.startswith("out[0]: 1 of 2 values differ")
    failure = fails(
        check.match, (expected[0], np.array([3, 2]), expected[2]), expected, label="out"
    )
    assert failure.message.startswith("out[1]: 1 of 2 values differ")
    failure = fails(check.match, (expected[0], expected[1], {"n": 2, "tokens": [1, 3]}), expected)
    assert failure.message.startswith("output[2]['tokens']: differs")
    failure = fails(check.match, expected[:2], expected)
    assert failure.message == "output: got 2 values, which is not the expected number"
    assert "expected 3 values" in failure.detail
    assert "expected a tensor, got a list" in fails(check.match, [0.5, 0.25], expected[0]).message
    failure = fails(check.match, np.array([3.0, 1.0]), np.array([3, 1]), dtype=True)
    assert "not the expected dtype" in failure.message


def test_shape_finite_that_and_raises():
    check.shape(np.zeros((2, 3)), (2, 3))
    failure = fails(check.shape, np.zeros((3, 2)), (2, 3))
    assert failure.message == "output: got shape (3, 2), which is not the expected shape"
    assert failure.detail == "result\n  expected shape    (2, 3)\n  got shape         (3, 2)\n"
    check.finite(np.ones(3))
    failure = fails(check.finite, np.array([1.0, np.inf]))
    assert "non-finite" in failure.message
    assert failure.detail == "value at [1]: inf\n"  # one line stays one line
    check.that(True, "never shown")
    assert fails(check.that, 0, "broken invariant", "why").detail == "why"
    check.raises(ValueError, int, "x")
    failure = fails(check.raises, ValueError, int, "3")
    assert failure.message == "call: returned normally where the statement specifies an exception"
    assert failure.detail == "expected ValueError, but nothing was raised\n"
    failure = fails(check.raises, ValueError, {}.__getitem__, "k")
    assert failure.message.startswith("call: raised KeyError, which is not the exception")
    assert "ValueError" not in failure.message
    assert failure.detail == "result\n  expected    ValueError\n  raised      KeyError: 'k'\n"


# -- property -------------------------------------------------------------------------


def test_forall_reports_the_first_failing_trial_and_how_to_replay_it(ctx):
    seen = []

    def prop(rng):
        value = float(rng.random())
        seen.append(value)
        check.that(len(seen) < 4, "invariant broken", "the detail")

    failure = fails(check.forall, prop, trials=10)
    assert failure.message == "invariant broken (property failed on trial 4 of 10)"
    assert failure.detail == (
        "property failed on trial 4 of 10; "
        f"replay with np.random.default_rng([{ctx.seed}, 3])\n"
        "\n"
        "the detail\n"
    )
    assert len(seen) == 4
    # Replaying the reported seed reproduces the failing input exactly.
    assert float(np.random.default_rng([ctx.seed, 3]).random()) == seen[3]


def test_forall_points_ctx_helpers_at_the_trial_generator(ctx):
    outer = ctx.rng
    draws = []
    check.forall(lambda rng: draws.append((ctx.rng is rng, float(ctx.uniform(1)[0]))), trials=3)
    assert all(same for same, _ in draws)
    assert len({value for _, value in draws}) == 3
    assert ctx.rng is outer
    again = []
    check.forall(lambda rng: again.append(float(ctx.uniform(1)[0])), trials=3)
    assert again == [value for _, value in draws]  # seeded: identical on every run


# -- statistical ------------------------------------------------------------------------


def test_distribution_accepts_correct_samplers_for_many_seeds():
    probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    for seed in range(200):
        rng = np.random.default_rng(seed)
        check.distribution(rng.choice(5, size=5000, p=probs), probs)


def test_distribution_rejects_a_shifted_distribution():
    probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    wrong = np.array([0.44, 0.36, 0.15, 0.04, 0.01])
    samples = np.random.default_rng(0).choice(5, size=20_000, p=wrong)
    failure = fails(check.distribution, samples, probs, label="tokens")
    assert failure.message.startswith(
        "tokens: the empirical distribution does not match the target: outcome"
    )
    assert "20000" not in failure.message
    assert "0.5" not in failure.message  # target probabilities stay in the detail
    facts, table = failure.detail.split("\n\n")
    assert facts == (
        "result\n"
        "  samples                20000\n"
        "  outcomes               5\n"
        "  allowed frequencies    false-failure probability below 1e-10"
    )
    rows = table.splitlines()
    assert rows[:2] == [
        "frequencies",
        "  outcome   target  allowed frequency    observed   count",
    ]
    assert len(rows) == 7 and all(row.startswith("  ") for row in rows[1:])
    assert rows[2].startswith("        0   0.5000  [") and rows[2].endswith("  <-- outside")
    assert not rows[4].endswith("<-- outside")


def test_distribution_threshold_is_the_documented_chernoff_bound():
    # One outcome of K=2 with p = 0.5, N = 1000: fail iff N * KL(f || p) >= ln(2K / delta).
    n, delta = 1000, 1e-10
    threshold = math.log(4 / delta)

    def statistic(count):
        f = count / n
        return n * (f * math.log(f / 0.5) + (1 - f) * math.log((1 - f) / 0.5))

    boundary = next(c for c in range(500, 1000) if statistic(c) >= threshold)
    check.distribution(counts=[boundary - 1, n - boundary + 1], probs=[0.5, 0.5], delta=delta)
    fails(check.distribution, counts=[boundary, n - boundary], probs=[0.5, 0.5], delta=delta)
    # The acceptance interval reported to the learner is that same boundary.
    low, high = check._kl_interval(0.5, threshold / n)
    assert (boundary - 1) / n <= high < boundary / n
    assert math.isclose(low, 1 - high, abs_tol=1e-9)


def test_distribution_impossible_outcomes_and_bad_samples():
    fails(check.distribution, [0, 0, 1, 2], [0.5, 0.5, 0.0])  # outcome 2 has probability 0
    failure = fails(check.distribution, [0, 1, 7], [0.5, 0.5])
    assert failure.message == "samples: sample 2 is 7, which is not a valid outcome"
    assert "range(2)" in failure.detail
    with pytest.raises(ValueError):
        check.distribution([0, 1], [0.5, 0.6])


def test_mean_uses_a_hoeffding_bound():
    rng = np.random.default_rng(1)
    accepted = rng.random(10_000) < 0.3
    check.mean(accepted, 0.3, low=0, high=1)
    failure = fails(check.mean, accepted, 0.4, low=0, high=1, label="acceptance rate")
    assert failure.message == (
        "acceptance rate: the average over the samples is not consistent with the target"
    )
    eps = math.sqrt(math.log(2 / 1e-10) / (2 * 10_000))
    assert failure.detail == (
        "result\n"
        "  samples     10000\n"
        f"  expected    0.4 +/- {eps:.3g}   (false-failure probability below 1e-10)\n"
        f"  observed    {accepted.mean():.6g}\n"
    )
    assert "outside the possible range" in fails(check.mean, [0.5, 2.0], 0.5, low=0, high=1).message


# -- cost --------------------------------------------------------------------------------


def test_cost_reports_a_metric_and_enforces_the_bound(ctx):
    counters = {"tokens_processed": 37, "forward_calls": 12}
    check.cost(counters, "tokens_processed", at_most=37, unit="tokens")
    assert ctx.metrics == [{"name": "tokens_processed", "value": 37, "unit": "tokens", "limit": 37}]
    failure = fails(check.cost, counters, "tokens_processed", at_most=20, unit="tokens")
    assert failure.message == "tokens_processed: 37 tokens exceeds the limit of 20 tokens"
    assert ctx.metrics[0]["limit"] == 20  # reported even when the check fails
    check.cost(counters, "forward_calls", exactly=12)
    assert "expected exactly 11" in fails(check.cost, counters, "forward_calls", exactly=11).message
    assert "below the required" in fails(check.cost, counters, "forward_calls", at_least=13).message


def test_at_most(ctx):
    check.at_most("p99_latency", 12.5, 50, unit="ms")
    assert ctx.metrics == [{"name": "p99_latency", "value": 12.5, "unit": "ms", "limit": 50}]
    failure = fails(check.at_most, "p99_latency", 80.0, 50, unit="ms")
    assert failure.message == "p99_latency: 80 ms exceeds the limit of 50 ms"
