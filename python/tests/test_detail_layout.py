"""The layout of failure details: groups a blank line apart, facts in two columns.

One test per kind of detail. Each pins the whole text, so a change of layout (or a
fact that goes missing) fails here. The details of sequences, records, shapes, dtypes,
exceptions, properties, and the statistical checks are pinned in test_check.py, the
``call`` group in test_context.py, and the speed judge's in test_speed.py.
"""

import numpy as np
import pytest

from learnai.judge import CheckFailure, check, layout

F32 = np.float32
INF = np.inf


def detail(fn, *args, **kwargs) -> str:
    with pytest.raises(CheckFailure) as info:
        fn(*args, **kwargs)
    return info.value.detail


def test_allclose_on_a_small_array_shows_both_arrays_whole():
    expected = np.array([[-INF, 3, -INF, 2, -INF], [-INF, 2, -INF, 2, -INF]], dtype=F32)
    got = expected.copy()
    got[1, 4] = 2
    assert detail(check.allclose, got, expected) == (
        "result\n"
        "  shape            (2, 5), dtype float32\n"
        "  mismatched       1 of 10 values\n"
        "  max abs error    inf at index [1, 4]   (expected -inf, got 2)\n"
        "  tolerance        |got - expected| <= 1e-06 + 0.0001 * |expected|\n"
        "\n"
        "expected\n"
        "  [[-inf,   3., -inf,   2., -inf],\n"
        "   [-inf,   2., -inf,   2., -inf]]\n"
        "\n"
        "got\n"
        "  [[-inf,   3., -inf,   2., -inf],\n"
        "   [-inf,   2., -inf,   2.,   2.]]\n"
    )


def test_allclose_on_a_large_array_shows_a_slice_and_both_errors():
    expected = np.ones((2, 1, 4, 4), dtype=F32)
    got = expected.copy()
    got[0, 0, 0, 1] = 3
    got[1, 0, 2, 2] = 1.5
    assert detail(check.allclose, got, expected, rtol=1e-3, atol=0) == (
        "result\n"
        "  shape            (2, 1, 4, 4), dtype float32\n"
        "  mismatched       2 of 32 values\n"
        "  max abs error    2 at index [0, 0, 0, 1]   (expected 1, got 3)\n"
        "  max rel error    2 at index [0, 0, 0, 1]   (expected 1, got 3)\n"
        "  tolerance        |got - expected| <= 0 + 0.001 * |expected|\n"
        "\n"
        "expected[0, 0, 0, 0:4]\n"
        "  [1., 1., 1., 1.]\n"
        "\n"
        "got[0, 0, 0, 0:4]\n"
        "  [1., 3., 1., 1.]\n"
    )


def test_an_excerpt_that_numpy_wraps_keeps_numpys_lines():
    row = [1.482301, -1.385029e-3, -0.3153042, 1.350284e-3, -1.018483, 1.36616, -3.201272, -0.82]
    expected = np.array([row, row, row], dtype=F32) * np.array([[1], [2], [3]], dtype=F32)
    got = expected * F32(2)
    lines = detail(check.allclose, got, expected).split("\n\n")[1].splitlines()
    text = np.array2string(
        expected[2], precision=6, separator=", ", max_line_width=96, threshold=64
    )
    assert "\n" in text  # numpy needs more than one line for this row
    assert lines == ["expected[2, 0:8]", *("  " + line for line in text.splitlines())]


def test_allclose_on_a_scalar():
    assert detail(check.allclose, 2.5, 1.5) == (
        "result\n"
        "  shape            (), dtype float64\n"
        "  mismatched       1 of 1 values\n"
        "  max abs error    1 at index (scalar)   (expected 1.5, got 2.5)\n"
        "  max rel error    0.6666667 at index (scalar)   (expected 1.5, got 2.5)\n"
        "  tolerance        |got - expected| <= 1e-06 + 0.0001 * |expected|\n"
        "\n"
        "expected\n"
        "  1.5\n"
        "\n"
        "got\n"
        "  2.5\n"
    )


def test_equal_on_a_large_integer_array_shows_a_slice():
    expected = np.arange(40).reshape(2, 20)
    got = expected.copy()
    got[1, 3] = -1
    assert detail(check.equal, got, expected) == (
        "result\n"
        "  first difference    at index [1, 3]   (expected 23, got -1)\n"
        "\n"
        "expected[1, 0:8]\n"
        "  [20, 21, 22, 23, 24, 25, 26, 27]\n"
        "\n"
        "got[1, 0:8]\n"
        "  [20, 21, 22, -1, 24, 25, 26, 27]\n"
    )


def test_equal_on_strings_and_records_is_two_groups():
    assert detail(check.equal, "<s><s>hi\n", "<s>hi\n") == (
        "expected\n  '<s>hi\\n'\n\ngot\n  '<s><s>hi\\n'\n"
    )
    assert detail(check.equal, {"ttft": 2.0}, {"ttft": 1.0}) == (
        "expected\n  {'ttft': 1.0}\n\ngot\n  {'ttft': 2.0}\n"
    )


def test_the_context_comes_first_and_is_not_laid_out_again():
    context = "token bytes: [b'a',   b'\\xc3']\n  second line"
    assert detail(check.equal, ["a", ""], ["a", "?"], context=context) == (
        f"{context}\n"
        "\n"
        "result\n"
        "  first difference    position 1   (expected '?', got '')\n"
        "\n"
        "expected[0:10]\n"
        "  ['a', '?']\n"
        "\n"
        "got[0:10]\n"
        "  ['a', '']\n"
    )
    # A detail of one line stays one line, after the context.
    assert detail(check.match, [1, 2], (1, 2, 3), context=context) == (
        f"{context}\n\nexpected 3 values\n"
    )
    assert detail(check.match, [1], {"zebra": 1}, context=context) == (
        f"{context}\n\nexpected keys: ['zebra']\n"
    )


def test_a_tests_own_detail_is_passed_through_unchanged():
    own = "probs = [[0.1 0.4]\n [0.  0.5]]\nu = [0.55]\n\n\nresult = [2 1 4]"
    assert detail(check.that, False, "broken", own) == own


def test_layout_helpers():
    assert layout.facts("result", [("a", 1), ("longer", "x\ny")]) == (
        "result\n  a         1\n  longer    x\n            y"
    )
    assert layout.block("got", "[[1],\n [2]]") == "got\n  [[1],\n   [2]]"
    assert layout.join("one\n", None, "", "two") == "one\n\ntwo\n"
