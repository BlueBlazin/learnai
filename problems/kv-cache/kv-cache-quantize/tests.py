"""Tests for kv-cache-quantize.

The state of the cache after S tokens depends only on S and on the parameters, not on
how the tokens arrived, so every case feeds a stream in chunks and compares, after each
call, with what `expected_state` computes in numpy from the whole stream: which tokens
are quantized, their codes, scales, and biases, and the float window. That is
independent of the oracle.

Codes are compared exactly, so no input may sit near a rounding tie (frameworks could
round (x - bias) / scale to different sides because of a one-ulp difference): streams
go through `tie_free`, as in `groupwise-4bit`. The model cases use a TinyLM's own keys
and values, which are not tie-free; they check sizes, the half-step bound, and the
house metric (`logit_kl`), never codes.

JAX compiles every new shape, so the cases share a few layouts and chunk sizes.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import KVCache, TinyLM, logit_kl

APPEND, READ = "quantized_append", "read_cache"
B, H, D, GROUP = 1, 2, 8, 4  # the layout most cases share
MARGIN = 0.05
TOL = {"rtol": 1e-4, "atol": 1e-5}
EXACT = {"rtol": 0.0, "atol": 0.0}


# -- the specification, in numpy --------------------------------------------------------


def group_stats(x, bits, group_size):
    b, h, s, d = x.shape
    groups = x.reshape(b, h, s, d // group_size, group_size)
    lo = groups.min(axis=-1, keepdims=True)
    hi = groups.max(axis=-1, keepdims=True)
    levels = np.float32(2**bits - 1)
    scale = np.where(hi > lo, (hi - lo) / levels, np.float32(1.0)).astype(np.float32)
    return groups, lo, hi, scale


def steps(x, bits, group_size):
    """(x - bias) / scale in float64: the number each code is rounded from."""
    groups, lo, _, scale = group_stats(x, bits, group_size)
    return (groups.astype(np.float64) - lo) / scale


def near_tie(t):
    return np.abs(t - np.floor(t) - 0.5) < MARGIN


def tie_free(x, bits, group_size):
    """A copy of x in which no element is within MARGIN of a rounding tie."""
    x = np.array(x, dtype=np.float32)
    groups, lo, hi, scale = group_stats(x, bits, group_size)
    t = steps(x, bits, group_size)
    snapped = np.clip(lo + np.round(t) * scale.astype(np.float64), lo, hi).astype(np.float32)
    fixed = np.where(near_tie(t), snapped, groups).reshape(x.shape)
    _, lo2, hi2, _ = group_stats(fixed, bits, group_size)
    ok = np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    if not ok or near_tie(steps(fixed, bits, group_size)).any():
        raise AssertionError("could not make the input tie-free")
    return fixed


def quantize(x, bits, group_size):
    """(codes, scale, bias) of the statement for a slab of tokens (B, H, S, D)."""
    _, lo, _, scale = group_stats(x, bits, group_size)
    codes = np.clip(np.round(steps(x, bits, group_size)), 0, 2**bits - 1).astype(np.uint8)
    return codes.reshape(x.shape), scale[..., 0], lo[..., 0]


def dequantize(codes, scale, bias):
    b, h, s, d = codes.shape
    groups = codes.reshape(b, h, s, scale.shape[-1], d // scale.shape[-1]).astype(np.float32)
    return (groups * scale[..., None] + bias[..., None]).reshape(b, h, s, d)


# -- feeding a stream and checking the cache after every call ----------------------------


def check_structure(cache, what):
    check.that(
        isinstance(cache, tuple) and len(cache) == 4,
        f"{APPEND} must return a tuple (k_quant, v_quant, k_recent, v_recent)",
    )
    for name, quant in (("k_quant", cache[0]), ("v_quant", cache[1])):
        check.that(
            isinstance(quant, tuple) and len(quant) == 3,
            f"{APPEND}: {name} must be a tuple (codes, scale, bias)",
        )
        check.that(
            quant[0].dtype == np.uint8,
            f"{APPEND}: the codes of {name} must be a uint8 tensor",
            f"got dtype {quant[0].dtype} {what}",
        )


def check_side(name, quant, recent, x, seen, bits, group_size, keep, what, *, codes=True):
    """One of keys or values against the stream x, of which `seen` tokens were appended."""
    n_recent = min(seen, keep)
    n_quant = seen - n_recent
    b, h, _, d = x.shape
    check.shape(quant[0], (b, h, n_quant, d), label=f"{name} codes {what}")
    check.shape(quant[1], (b, h, n_quant, d // group_size), label=f"{name} scale {what}")
    check.shape(quant[2], (b, h, n_quant, d // group_size), label=f"{name} bias {what}")
    check.shape(recent, (b, h, n_recent, d), label=f"float {name} window {what}")
    check.allclose(recent, x[:, :, n_quant:seen], **EXACT, label=f"float {name} window {what}")
    for part, label in ((quant[1], "scale"), (quant[2], "bias")):
        check.finite(part, label=f"{name} {label} {what}")
    if codes:
        want_codes, want_scale, want_bias = quantize(x[:, :, :n_quant], bits, group_size)
        check.allclose(quant[1], want_scale, **TOL, label=f"{name} scale {what}")
        check.allclose(quant[2], want_bias, **TOL, label=f"{name} bias {what}")
        check.equal(quant[0], want_codes, label=f"{name} codes {what}")


def check_read(ctx, cache, k, v, seen, bits, group_size, keep, what):
    """read_cache: old tokens within half a step of the stream, the float window exact."""
    result = ctx.call(READ, cache)
    check.that(
        isinstance(result, tuple) and len(result) == 2,
        f"{READ} must return a tuple (keys, values)",
    )
    n_quant = seen - min(seen, keep)
    for name, got, x in (("keys", result[0], k), ("values", result[1], v)):
        check.shape(got, x[:, :, :seen].shape, label=f"{name} read back {what}")
        check.finite(got, label=f"{name} read back {what}")
        check.allclose(got[:, :, n_quant:], x[:, :, n_quant:seen], **EXACT,
                       label=f"{name} of the float window read back {what}")  # fmt: skip
        if n_quant:
            old = x[:, :, :n_quant]
            _, _, _, scale = group_stats(old, bits, group_size)
            half = np.repeat(scale[..., 0], group_size, axis=-1).reshape(old.shape) / 2
            worst = np.abs(got[:, :, :n_quant].astype(np.float64) - old) / half
            check.that(
                float(worst.max()) <= 1.001,
                f"{name} read back {what}: a quantized element is more than half a step "
                "from its float value",
                f"largest error: {float(worst.max()):.3f} half-steps (limit 1); the step of "
                "a group is (max - min) / (2**bits - 1)",
            )
    return result


def feed(ctx, k, v, sizes, bits, group_size, keep, *, codes=True):
    """Append the stream in chunks of the given sizes; check the cache after each call."""
    cache, seen = None, 0
    for size in sizes:
        stop = seen + size
        cache = ctx.call(APPEND, cache, k[:, :, seen:stop], v[:, :, seen:stop],
                         bits, group_size, keep)  # fmt: skip
        what = f"after {stop} tokens (last call: {size})"
        check_structure(cache, what)
        check_side("key", cache[0], cache[2], k, stop, bits, group_size, keep, what, codes=codes)
        check_side("value", cache[1], cache[3], v, stop, bits, group_size, keep, what, codes=codes)
        seen = stop
    what = f"after {seen} tokens"
    keys, values = check_read(ctx, cache, k, v, seen, bits, group_size, keep, what)
    if codes:
        n_quant = seen - min(seen, keep)
        for name, got, x in (("keys", keys, k), ("values", values, v)):
            want = dequantize(*quantize(x[:, :, :n_quant], bits, group_size))
            check.allclose(got[:, :, :n_quant], want, **TOL,
                           label=f"dequantized {name} {what}")  # fmt: skip
    return cache


def stream(ctx, n, bits, group_size=GROUP, b=B, h=H, d=D):
    """Keys and values for n tokens whose groups have ranges and centers of their own."""

    def one():
        shape = (b, h, n, d // group_size, 1)
        width = ctx.uniform(*shape, low=0.05, high=3.0)
        center = ctx.uniform(*shape, low=-4.0, high=4.0)
        x = ctx.randn(b, h, n, d // group_size, group_size) * width + center
        return tie_free(x.reshape(b, h, n, d), bits, group_size)

    return one(), one()


# -- cases ------------------------------------------------------------------------------

EXAMPLE_K = np.array(
    [
        [0.0, 3.0, 0.9, 2.2, 5.0, 5.0, 5.0, 5.0],
        [-1.0, 0.5, -0.4, 0.1, 10.0, 40.0, 22.0, 31.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ],
    dtype=np.float32,
)[None, None]


@case("the worked example: bits 2, group_size 4, keep_recent 1", public=True)
def worked_example(ctx):
    k, v = EXAMPLE_K, 2 * EXAMPLE_K
    cache = ctx.check(APPEND, None, k[:, :, :2], v[:, :, :2], 2, 4, 1)
    codes, scale, bias = cache[0]
    check.equal(codes[0, 0], [[0, 3, 1, 2, 0, 0, 0, 0]], label="key codes after call 1")
    check.allclose(scale[0, 0], [[1.0, 1.0]], label="key scale after call 1")
    check.allclose(bias[0, 0], [[0.0, 5.0]], label="key bias after call 1")
    check.allclose(cache[2], k[:, :, 1:2], **EXACT, label="float key window after call 1")

    cache = ctx.check(APPEND, cache, k[:, :, 2:], v[:, :, 2:], 2, 4, 1)
    codes, scale, bias = cache[0]
    check.equal(codes[0, 0], [[0, 3, 1, 2, 0, 0, 0, 0], [0, 3, 1, 2, 0, 3, 1, 2]],
                label="key codes after call 2")  # fmt: skip
    check.allclose(scale[0, 0], [[1.0, 1.0], [0.5, 10.0]], label="key scale after call 2")
    check.allclose(bias[0, 0], [[0.0, 5.0], [-1.0, 10.0]], label="key bias after call 2")
    check.allclose(cache[2], k[:, :, 2:], **EXACT, label="float key window after call 2")

    keys, values = ctx.check(READ, cache)
    want = np.array(
        [
            [0.0, 3.0, 1.0, 2.0, 5.0, 5.0, 5.0, 5.0],
            [-1.0, 0.5, -0.5, 0.0, 10.0, 40.0, 20.0, 30.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )[None, None]
    check.allclose(keys, want, label="keys read back")
    check.allclose(values, 2 * want, label="values read back")


# Rejects quantizes_on_arrival (the window holds round-tripped values) and
# keep_recent_off_by_one (the window has the wrong length from the fourth token on).
@case("one token at a time, 8 bits: a token stays float until 3 newer ones exist", public=True)
def token_by_token(ctx):
    k, v = stream(ctx, 8, bits=8)
    feed(ctx, k, v, [1] * 8, bits=8, group_size=GROUP, keep=3)


# Rejects one_range_per_call: three tokens leave the window in one call, and each of
# their groups has a range of its own.
@case("4 bits: every group of every token has its own scale and bias", public=True)
def own_scale_per_group(ctx):
    k, v = stream(ctx, 8, bits=4)
    feed(ctx, k, v, [2, 3, 3], bits=4, group_size=GROUP, keep=2)


# Rejects moves_old_window_only: a 6-token chunk into a window of 3 must quantize the
# old window and the first three tokens of the chunk itself.
@case("a chunk longer than keep_recent: part of it is quantized on arrival", public=True)
def long_chunk(ctx):
    k, v = stream(ctx, 12, bits=8)
    feed(ctx, k, v, [2, 6, 1, 3], bits=8, group_size=GROUP, keep=3)


def model_stream(ctx, seed, n_prompt):
    """A TinyLM (head dimension 16), a prompt for it, and a continuation."""
    lm = TinyLM(ctx.fw, seed=seed, d_model=64, n_heads=4)
    rng = np.random.default_rng(seed)
    prompt = rng.integers(2, lm.vocab_size, size=n_prompt)
    rest = rng.integers(2, lm.vocab_size, size=8)
    return lm, prompt, rest


def model_kl(ctx, seed, bits, keep, sizes):
    """Mean KL of the model's next-token distributions: float cache against the solution's."""
    lm, prompt, rest = model_stream(ctx, seed, sum(sizes))
    _, cache = lm(prompt)
    keys, values = [], []
    for k, v in zip(cache.keys, cache.values, strict=True):
        k, v = np.array(k), np.array(v)  # (1, 4, P, 16)
        state = feed(ctx, k, v, sizes, bits, 8, keep, codes=False)
        got_k, got_v = ctx.call(READ, state)
        keys.append(np.asarray(got_k, dtype=np.float32))
        values.append(np.asarray(got_v, dtype=np.float32))
    lossy = KVCache(tuple(keys), tuple(values), cache.length, cache.batched)
    exact_logits, _ = lm(rest, cache)
    lossy_logits, _ = lm(rest, lossy)
    kl = logit_kl(ctx.fw.numpy(exact_logits), ctx.fw.numpy(lossy_logits))
    return float(kl.mean())


# The reference measures 2.9e-06 nats here. Measured on the same stream: truncating
# codes 3.2e-05, one range per call 1.8e-05.
@case("a tiny model reads the 8-bit cache: its predictions move by at most 1e-5 nats",
      public=True)  # fmt: skip
def model_8bit(ctx):
    kl = model_kl(ctx, seed=11, bits=8, keep=4, sizes=[16, 1, 1, 6])
    check.at_most("logit_kl", kl, 1e-5, unit="nats")


@case("keep_recent = 0: every token is quantized in the call that appends it")
def no_float_window(ctx):
    k, v = stream(ctx, 8, bits=8)
    feed(ctx, k, v, [2, 3, 3], bits=8, group_size=GROUP, keep=0)


@case("keep_recent larger than the stream: nothing is quantized and reads are exact")
def nothing_quantized(ctx):
    k, v = stream(ctx, 8, bits=8)
    cache = feed(ctx, k, v, [2, 3, 3], bits=8, group_size=GROUP, keep=20)
    keys, values = ctx.call(READ, cache)
    check.allclose(keys, k, **EXACT, label="keys read back")
    check.allclose(values, v, **EXACT, label="values read back")


@case("one group per token (group_size = D) and one channel per group (group_size = 1)")
def group_size_extremes(ctx):
    k, v = stream(ctx, 8, bits=4, group_size=8)
    feed(ctx, k, v, [2, 3, 3], bits=4, group_size=8, keep=2)
    # A group of one channel is constant: scale 1, code 0, and an exact read.
    k, v = stream(ctx, 8, bits=4, group_size=1)
    cache = feed(ctx, k, v, [2, 3, 3], bits=4, group_size=1, keep=2)
    keys, _ = ctx.call(READ, cache)
    check.allclose(keys, k, **EXACT, label="keys read back with group_size 1")


# Rejects levels_off_by_one at every width, and truncates.
@case("bits from 2 to 8: the largest code is 2**bits - 1 and the grid ends on the maximum")
def every_width(ctx):
    for bits in (2, 3, 5, 6, 7, 8):
        k, v = stream(ctx, 8, bits=bits)
        cache = feed(ctx, k, v, [2, 3, 3], bits=bits, group_size=GROUP, keep=2)
        check.that(
            int(cache[0][0].max()) == 2**bits - 1,
            f"bits={bits}: the largest key code is {int(cache[0][0].max())}, expected "
            f"{2**bits - 1}",
            f"expected 2**bits - 1 = {2**bits - 1}: the maximum of a group maps to the top code",
        )


@case("constant groups (all zeros, all equal): scale 1, codes 0, no NaN")
def constant_groups(ctx):
    k, v = stream(ctx, 8, bits=4)
    k[:, :, 1] = 0.0  # a whole token of zeros
    k[:, 0, 3, :4] = -2.5  # one constant group
    v[:, :, 0, 4:] = 7.0
    v[:, 1, 5] = 0.0
    cache = feed(ctx, k, v, [2, 3, 3], bits=4, group_size=GROUP, keep=2)
    keys, _ = ctx.call(READ, cache)
    check.allclose(keys[:, :, 1], k[:, :, 1], **EXACT, label="the all-zero token read back")


@case("random chunkings and window sizes give the state the token count determines")
def chunking_property(ctx):
    def prop(rng):
        keep = int(rng.choice([0, 1, 3, 5]))
        sizes = [int(s) for s in rng.choice([1, 2, 3, 6], size=4)]
        bits = int(rng.choice([4, 8]))
        k, v = stream(ctx, sum(sizes), bits=bits)
        feed(ctx, k, v, sizes, bits=bits, group_size=GROUP, keep=keep)

    check.forall(prop, trials=6)


@case("other sizes: B = 2, H = 3, D = 16, group_size 8, 6 bits")
def other_sizes(ctx):
    k, v = stream(ctx, 8, bits=6, group_size=8, b=2, h=3, d=16)
    feed(ctx, k, v, [2, 3, 3], bits=6, group_size=8, keep=2)


# The reference measures 5.3e-04 nats here. Measured on the same stream: one range per
# call 2.6e-03, 2**bits steps 2.0e-03, truncating codes 5.8e-03.
@case("a tiny model reads a 4-bit cache: its predictions move by at most 1.5e-3 nats")
def model_4bit(ctx):
    kl = model_kl(ctx, seed=5, bits=4, keep=4, sizes=[16, 1, 1, 6])
    check.at_most("logit_kl", kl, 1.5e-3, unit="nats")
