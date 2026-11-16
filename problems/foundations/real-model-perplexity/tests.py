"""Tests for real-model-perplexity.

Three kinds of `forward` are handed to the solution, all through the class `Forward`
below, which refuses more than `max_len` tokens:

- the counting model of the statement's example, whose probabilities are small
  fractions (no download);
- the kit's `TinyLM`, a random two-layer transformer with a 64-token vocabulary (no
  download). The expected value is computed here in float64 by `oracle`, which is
  written per target token instead of per window, so it shares no loop with the
  reference solution;
- the real granite-4.0-350m through the kit's `RealLM`. The expected perplexities come
  from `transformers` (`fixtures/make.py`).

Window lengths are kept to a few values (16 and 4, plus the short last windows), since
JAX compiles once per shape.
"""

import math

import numpy as np

from learnai.judge import case, check
from learnai.kit import RealModel, TinyLM

FN = "perplexity"
ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
TOL = {"rtol": 1e-3, "atol": 1e-4}  # the tolerance stated in Constraints


class CountingModel:
    """The model of the statement's example: add-one counts over the visible context.

    After reading ids[0..t], token x has probability (1 + number of x among them) /
    (V + t + 1). Row t of the logits is the logarithm of those probabilities.
    """

    def __init__(self, vocab_size, fw=None):
        self.vocab_size = vocab_size
        self.fw = fw

    def with_framework(self, fw):
        return CountingModel(self.vocab_size, fw)

    def __call__(self, tokens):
        ids = [int(t) for t in (tokens.tolist() if hasattr(tokens, "tolist") else tokens)]
        counts = np.ones(self.vocab_size)
        rows = []
        for token in ids:
            counts[token] += 1
            rows.append(np.log(counts / counts.sum()))
        logits = np.asarray(rows, dtype=np.float32)
        return (logits, None) if self.fw is None else (self.fw.tensor(logits), None)


class Forward:
    """What the solution receives as `forward`: logits for at most `max_len` tokens.

    `lm` is called like TinyLM: lm(tokens) -> (logits, cache). `ctx.call` rebinds this
    object to the solution's framework (`with_framework`), like every kit object.
    """

    def __init__(self, lm, max_len):
        self.lm = lm
        self.max_len = max_len
        self.fw = lm.fw

    def with_framework(self, fw):
        return Forward(self.lm.with_framework(fw), self.max_len)

    def __call__(self, token_ids):
        n = len(token_ids)
        if n > self.max_len:
            raise ValueError(
                f"forward: got {n} tokens, but the model reads at most {self.max_len} at a time"
            )
        if n == 0:
            raise ValueError("forward: got no tokens")
        logits, _ = self.lm(token_ids)
        return logits


def oracle(lm, ids, max_len, stride):
    """The specified perplexity in float64, one target token at a time.

    Token p is scored by window k(p), the first window that reaches it:
    the smallest k with p < k * stride + max_len.
    """
    n = len(ids)
    rows = {}  # window index -> float64 log-probabilities

    def window(k):
        if k not in rows:
            begin = k * stride
            out = lm(list(ids[begin : begin + max_len]))[0]
            logits = np.asarray(out.tolist() if hasattr(out, "tolist") else out, dtype=np.float64)
            top = logits.max(axis=-1, keepdims=True)
            rows[k] = logits - top - np.log(np.exp(logits - top).sum(axis=-1, keepdims=True))
        return rows[k]

    total = 0.0
    for p in range(1, n):
        k = max(0, math.ceil((p + 1 - max_len) / stride))
        total += window(k)[p - 1 - k * stride, ids[p]]
    return math.exp(-total / (n - 1))


def run(ctx, lm, ids, max_len, stride, expected, label):
    got = ctx.call(FN, Forward(lm, max_len), np.asarray(ids, dtype=np.int64), max_len, stride)
    check.that(
        isinstance(got, np.ndarray),
        f"{label}: returned a {type(got).__name__}, not a tensor of the framework",
    )
    check.shape(got, (), label=label)
    check.that(got.dtype == np.float32, f"{label}: got dtype {got.dtype}, not float32")
    check.allclose(got, np.float32(expected), **TOL, label=label)
    return float(got)


def tiny(ctx, seed, n, max_len, stride, label):
    lm = TinyLM(ctx.fw, seed=seed)
    ids = [int(t) for t in ctx.rng.integers(2, lm.vocab_size, n)]
    twin = TinyLM(ctx.oracle_fw, seed=seed)
    return run(ctx, lm, ids, max_len, stride, oracle(twin, ids, max_len, stride), label)


# ---------------------------------------------------------------------------------
# No download needed.
# ---------------------------------------------------------------------------------

EXAMPLE = [0, 1, 0, 0, 2, 0, 1]


@case("the worked example: 7 tokens, windows of 4, stride 2, the counting model",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    lm = CountingModel(4)
    # 1/5 * 1/3 * 3/7 * 1/6 * 3/7 * 1/6 = 1/2940, six scored tokens.
    expected = 2940 ** (1 / 6)
    check.that(abs(oracle(lm, EXAMPLE, 4, 2) - expected) < 1e-5, "test setup: example mismatch")
    run(ctx, lm, EXAMPLE, 4, 2, expected, "perplexity(example)")


# Rejects nothing a windowing bug would: one window is plain sequence perplexity.
@case("a text that fits in one window: 16 tokens, max_len 16", public=True, assets=[])
def single_window(ctx):
    tiny(ctx, 3, 16, 16, 8, "perplexity(16 tokens, max_len 16, stride 8)")


# Rejects double_counts_overlap, disjoint_windows, mean_of_window_means, ignores_max_len.
@case("40 tokens of a small random model: windows of 16, stride 8", public=True, assets=[])
def full_windows(ctx):
    tiny(ctx, 3, 40, 16, 8, "perplexity(40 tokens, max_len 16, stride 8)")


# Rejects drops_partial_window: the last window holds 6 tokens, 2 of them unscored.
@case("a short last window: 42 tokens, windows of 16, stride 4", public=True, assets=[])
def short_last_window(ctx):
    tiny(ctx, 5, 42, 16, 4, "perplexity(42 tokens, max_len 16, stride 4)")


@case("the smallest inputs: 2 tokens, and one token more than a window", assets=[])
def smallest(ctx):
    tiny(ctx, 5, 2, 16, 8, "perplexity(2 tokens)")
    tiny(ctx, 5, 17, 16, 8, "perplexity(a text one token longer than max_len)")


@case("stride 1 and stride max_len - 1, windows of 4", assets=[])
def extreme_strides(ctx):
    tiny(ctx, 7, 12, 4, 1, "perplexity(stride 1)")
    tiny(ctx, 7, 12, 4, 3, "perplexity(stride max_len - 1)")


@case("the counting model on a longer text, three strides", assets=[])
def counting_model(ctx):
    lm = CountingModel(4)
    ids = [int(t) for t in ctx.rng.integers(0, 4, 12)]
    for stride in (1, 2, 3):
        run(ctx, lm, ids, 4, stride, oracle(lm, ids, 4, stride), "perplexity(counting model)")


# ---------------------------------------------------------------------------------
# granite-4.0-350m: windows of 16 tokens, 100,352 logits per row.
# ---------------------------------------------------------------------------------


def real(ctx, index, *, hidden):
    fixture = ctx.fixture("granite-4.0-350m.json")
    check.that(fixture["revision"] == REVISION, "the fixture was made from another revision")
    passage = fixture["passages"][index]
    lm = RealModel(ctx.asset(ASSET)).lm()
    what = "the passage" if hidden else f"{len(passage['ids'])} tokens of {passage['text'][:28]!r}"
    label = f"perplexity({what}, max_len {passage['max_len']}, stride {passage['stride']})"
    got = run(ctx, lm, passage["ids"], passage["max_len"], passage["stride"],
              passage["perplexity"], label)  # fmt: skip
    ctx.metric("perplexity", round(got, 3))


@case("Granite 4.0 350M: a paragraph about whetstones, windows of 16, stride 8", public=True)
def real_prose(ctx):
    real(ctx, 0, hidden=False)


@case("Granite 4.0 350M: a Python function, windows of 16, stride 4")
def real_code(ctx):
    real(ctx, 1, hidden=True)


# Rejects log_with_epsilon: one token of this passage has a probability near 1e-16,
# which no toy model produces.
@case("Granite 4.0 350M: a sentence with a made-up word, windows of 16, stride 12")
def real_surprise(ctx):
    real(ctx, 2, hidden=True)
