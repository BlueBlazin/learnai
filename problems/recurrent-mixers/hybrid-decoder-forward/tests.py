"""Tests for hybrid-decoder-forward.

Weights come from the kit (`hybrid_params`). Expected values come from three places:
the oracle reference, the kit's numpy forward pass (`hybrid_forward_ref`), and, for the
three configurations of `fixtures/make.py`, logits and caches computed by a tiny
`Qwen3_5ForCausalLM` of `transformers` holding the same weights (`fixtures/shipped.npz`).

Stage 1 tests `layer_types`. Stage 2 calls `hybrid_forward` with `state=None` and looks
at the logits only, so it can be passed before any state is returned. Stage 3 feeds the
solution its own returned state, and states made by the kit.

JAX compiles once per input shape, and every call that continues a cache has a new
shape (the number of cached tokens), so the shapes are few. A sequence has 8 tokens. It
is fed as a prefill of 6 and then single tokens (`PREFILL`), or as 1 + 1 + 1 + 5
(`SINGLES`: a prefill of one token, single tokens while the 3 rows of convolution state
fill up, and a 5-token piece that continues a cache). With the continuation of 3 tokens
by 5, the pairs (cached, new) are (0, 6), (6, 1), (7, 1), (0, 1), (1, 1), (2, 1), and
(3, 5). Stage 2 feeds the first
6 tokens in one call; the model is causal, so the fixture's 8-token logits give the
expected rows. Every configuration has the same tensor sizes.
"""

from dataclasses import replace

import numpy as np

from learnai.judge import case, check
from learnai.kit import HybridConfig, hybrid_forward_ref, hybrid_params

SCHEDULE, FN = "layer_types", "hybrid_forward"
S1, S2, S3 = "1: layer schedule", "2: one pass", "3: state and cached decoding"
# The tolerance stated in Constraints. The largest difference measured between a correct
# solution (this problem's float32 references, and the kit's forward pass in float64) and
# the `transformers` model of the fixture is 2.0e-5 on logits of magnitude 7 and 1.0e-5
# on a state tensor (3.6e-5 and 2.9e-5 with a 1 + 2 + 5 split, measured while authoring): the library sums in another order and runs every multi-token call
# through the chunked delta rule. The tolerance leaves a factor of ten.
TOL = {"rtol": 5e-4, "atol": 5e-4}
FULL, LINEAR = "full_attention", "linear_attention"
N, FIRST = 8, 6  # tokens of a sequence; tokens of a stage 2 call
PREFILL, SINGLES = [FIRST, 1, 1], [1, 1, 1, 5]
SHIPPED_CHUNKINGS = (PREFILL, SINGLES)  # the order of fixtures/make.py

# The three configurations of the fixture (fixtures/make.py builds the same objects).
BASE = HybridConfig(rope_base=1e7)  # linear, linear, linear, full
DEEP = HybridConfig(n_layers=8, rope_base=1e7)  # attention at layers 3 and 7
MIXED = HybridConfig(n_layers=5, rope_base=1e7, layer_types=(FULL, LINEAR, FULL, LINEAR, LINEAR))
SHIPPED = {"base": BASE, "deep": DEEP, "mixed": MIXED}
# The worked example: a linear layer, then an attention layer, in the kit's default sizes.
EXAMPLE = HybridConfig(n_layers=2, full_attention_interval=2, rope_base=1e7)
EXAMPLE_TOKENS = np.array([3, 1, 4, 1, 5, 9])
EXAMPLE_LOGITS = np.array(  # logits[:, :4] of the six tokens
    [
        [-1.386595, -0.069653, 4.239098, 0.426853],
        [-0.144362, 2.301395, 3.292907, 0.386430],
        [0.573805, -0.529514, 3.407029, 0.064500],
        [1.511575, -0.140076, 4.267532, -1.260582],
        [-1.357017, -0.347809, 2.133819, -3.233957],
        [0.675812, -1.043934, 0.745240, -0.798664],
    ],
    dtype=np.float32,
)


def shipped(ctx, name):
    """(cfg, params, tokens, fixture) of one configuration of fixtures/shipped.npz."""
    f = ctx.fixture("shipped.npz")
    cfg = SHIPPED[name]
    return cfg, hybrid_params(cfg, int(f["seed"])), f[f"{name}_tokens"], f


def shipped_state(f, name, cfg):
    return [
        (f[f"{name}_state_{layer}_0"], f[f"{name}_state_{layer}_1"])
        for layer in range(cfg.n_layers)
    ]


def call(ctx, params, cfg, tokens, state):
    """Call hybrid_forward; the result must be a pair whose first part is (T, vocab)."""
    result = ctx.call(FN, params, cfg, tokens, state)
    check.that(
        isinstance(result, tuple) and len(result) == 2,
        f"{FN}: the result is not a pair (logits, state)",
    )
    check.shape(result[0], (len(tokens), cfg.vocab_size), label="logits")
    return result


def one_pass(ctx, params, cfg, tokens, want=None, what="logits"):
    """Stage 2: the logits of a call with state=None. The returned state is not looked at."""
    logits, _ = call(ctx, params, cfg, tokens, None)
    if want is None:
        want = hybrid_forward_ref(cfg, params, tokens)[0]
    check.allclose(logits, want, **TOL, label=what, dtype=True)
    return logits


def compare_state(got, want, cfg, what=""):
    """One pair per layer, shapes first, then values. `what` only in public cases."""
    what = f"{what}: " if what else ""
    check.that(
        isinstance(got, (list, tuple)) and len(got) == cfg.n_layers,
        f"{what}the returned state is not a list with one entry per layer",
    )
    names = {LINEAR: ("conv_state", "S"), FULL: ("k", "v")}
    for layer, (pair, expected, kind) in enumerate(zip(got, want, cfg.layers, strict=True)):
        check.that(
            isinstance(pair, (list, tuple)) and len(pair) == 2,
            f"{what}state[{layer}] is not a pair",
        )
        for part, (value, target) in enumerate(zip(pair, expected, strict=True)):
            # A hidden case does not say which kind of layer sits where.
            described = f" ({kind}), {names[kind][part]}" if what else f"[{part}]"
            label = f"{what}state[{layer}]{described}"
            check.that(isinstance(value, np.ndarray), f"{label}: not a tensor")
            check.shape(value, target.shape, label=label)
            check.allclose(value, target, **TOL, label=label, dtype=True)  # float32, as stated


def run_chunks(ctx, params, cfg, tokens, sizes, state=None):
    """Feed `tokens` in consecutive pieces, handing each call the state of the one before."""
    assert sum(sizes) == len(tokens)
    outs, start = [], 0
    for size in sizes:
        logits, state = call(ctx, params, cfg, tokens[start : start + size], state)
        outs.append(logits)
        start += size
    return np.concatenate(outs), state


def chunked(ctx, params, cfg, tokens, sizes, want_logits, want_state, what=""):
    """Run one chunking and compare logits and final state. `what` only in public cases."""
    logits, state = run_chunks(ctx, params, cfg, tokens, sizes)
    prefix = f"{what}: " if what else ""
    check.allclose(logits, want_logits, **TOL, label=f"{prefix}logits")
    compare_state(state, want_state, cfg, what)
    return logits


# -- stage 1 --------------------------------------------------------------------------


def schedule(ctx, n_layers, interval):
    got = ctx.call(SCHEDULE, n_layers, interval)
    check.that(
        isinstance(got, (list, tuple)) and all(isinstance(kind, str) for kind in got),
        f"{SCHEDULE}: the result is not a list of strings",
    )
    return list(got)


@case("the worked example: layer_types(4, 4), (6, 2), and (3, 1)", public=True, stage=S1)
def example_schedule(ctx):
    check.equal(schedule(ctx, 4, 4), [LINEAR, LINEAR, LINEAR, FULL], label="layer_types(4, 4)")
    check.equal(schedule(ctx, 6, 2), [LINEAR, FULL] * 3, label="layer_types(6, 2)")
    check.equal(schedule(ctx, 3, 1), [FULL] * 3, label="layer_types(3, 1)")


@case("the schedules Qwen3_5TextConfig derives, 24 layers at interval 4 among them",
      public=True, stage=S1)  # fmt: skip
def shipped_schedules(ctx):
    f = ctx.fixture("shipped.npz")
    for n_layers, interval in ((24, 4), (4, 4), (7, 3), (5, 1), (4, 6), (9, 2)):
        check.equal(
            schedule(ctx, n_layers, interval),
            [str(kind) for kind in f[f"schedule_{n_layers}_{interval}"]],
            label=f"layer_types({n_layers}, {interval})",
        )


@case("every schedule up to 12 layers", stage=S1)
def schedule_sweep(ctx):
    for n_layers in range(1, 13):
        for interval in range(1, 14):
            want = [FULL if (i + 1) % interval == 0 else LINEAR for i in range(n_layers)]
            check.equal(schedule(ctx, n_layers, interval), want, label=SCHEDULE)


# -- stage 2 --------------------------------------------------------------------------


@case("the worked example: a linear layer and an attention layer, tokens [3, 1, 4, 1, 5, 9]",
      public=True, stage=S2)  # fmt: skip
def example_one_pass(ctx):
    params = hybrid_params(EXAMPLE, 3)
    logits = one_pass(ctx, params, EXAMPLE, EXAMPLE_TOKENS)
    check.allclose(logits[:, :4], EXAMPLE_LOGITS, **TOL, label="logits[:, :4]")


@case("one attention layer alone: 6 tokens", public=True, stage=S2)
def attention_alone(ctx):
    cfg = replace(BASE, n_layers=1, layer_types=(FULL,))
    one_pass(ctx, hybrid_params(cfg, 2), cfg, ctx.randint(0, cfg.vocab_size, (FIRST,)))


@case("one linear-attention layer alone: 6 tokens", public=True, stage=S2)
def linear_alone(ctx):
    cfg = replace(BASE, n_layers=1, layer_types=(LINEAR,))
    one_pass(ctx, hybrid_params(cfg, 3), cfg, ctx.randint(0, cfg.vocab_size, (FIRST,)))


@case("the same model in Hugging Face Transformers: 4 layers, 6 tokens in one call",
      public=True, stage=S2)  # fmt: skip
def shipped_base_one_pass(ctx):
    cfg, params, tokens, f = shipped(ctx, "base")
    one_pass(ctx, params, cfg, tokens[:FIRST], f["base_one_pass"][:FIRST])
    one_pass(ctx, params, cfg, tokens[:FIRST], what="logits, against the kit's forward pass")


# Written for the mutants that leave an epsilon at a default: with norm_eps = 0.3 every
# norm of the model (block norms, QK norms, the gated norm of the delta layers, the final
# norm) depends on it visibly.
@case("norm_eps = 0.3: every norm reads it from cfg", public=True, stage=S2)
def large_norm_eps(ctx):
    cfg = replace(BASE, norm_eps=0.3)
    params, tokens = hybrid_params(cfg, 4), ctx.randint(0, cfg.vocab_size, (FIRST,))
    one_pass(ctx, params, cfg, tokens)


# Binds the two rotation settings: the fixture's are 1e7 and 0.25 everywhere else.
@case("rope_base = 50 and partial_rotary_factor = 0.5 are read from cfg", public=True, stage=S2)
def other_rope(ctx):
    cfg = replace(BASE, rope_base=50.0, partial_rotary_factor=0.5)
    params, tokens = hybrid_params(cfg, 5), ctx.randint(0, cfg.vocab_size, (FIRST,))
    one_pass(ctx, params, cfg, tokens)


@case("the library again: 8 layers", stage=S2)
def shipped_deep_one_pass(ctx):
    cfg, params, tokens, f = shipped(ctx, "deep")
    one_pass(ctx, params, cfg, tokens[:FIRST], f["deep_one_pass"][:FIRST])


@case("the library again: an explicit layer_types that starts with attention", stage=S2)
def shipped_mixed_one_pass(ctx):
    cfg, params, tokens, f = shipped(ctx, "mixed")
    one_pass(ctx, params, cfg, tokens[:FIRST], f["mixed_one_pass"][:FIRST])


@case("a single token", stage=S2)
def single_token(ctx):
    for seed in (6, 7):
        params, tokens = hybrid_params(BASE, seed), ctx.randint(0, BASE.vocab_size, (1,))
        one_pass(ctx, params, BASE, tokens)


# -- stage 3 --------------------------------------------------------------------------


@case("the worked example: [3], then [1] with the returned state", public=True, stage=S3)
def example_continue(ctx):
    params = hybrid_params(EXAMPLE, 3)
    _, state = call(ctx, params, EXAMPLE, EXAMPLE_TOKENS[:1], None)
    compare_state(state, hybrid_forward_ref(EXAMPLE, params, EXAMPLE_TOKENS[:1])[1], EXAMPLE, "[3]")
    logits, state = call(ctx, params, EXAMPLE, EXAMPLE_TOKENS[1:2], state)
    check.allclose(
        logits[:, :4], EXAMPLE_LOGITS[1:2], **TOL, label="logits[:, :4] of [1] after [3]"
    )
    want = hybrid_forward_ref(EXAMPLE, params, EXAMPLE_TOKENS[:2])[1]
    compare_state(state, want, EXAMPLE, "[1] after [3]")


@case("the state after 6 tokens in one call", public=True, stage=S3)
def state_after_one_call(ctx):
    cfg, params, tokens, _ = shipped(ctx, "base")
    _, state = call(ctx, params, cfg, tokens[:FIRST], None)
    compare_state(state, hybrid_forward_ref(cfg, params, tokens[:FIRST])[1], cfg, "6 tokens")


@case("a prefill of 6 tokens, then 2 calls of one token; the library's cache",
      public=True, stage=S3)  # fmt: skip
def prefill_then_tokens(ctx):
    cfg, params, tokens, f = shipped(ctx, "base")
    want = shipped_state(f, "base", cfg)
    logits = chunked(
        ctx, params, cfg, tokens, PREFILL, f["base_cached_0"], want, "against the library"
    )
    check.allclose(logits, f["base_one_pass"], **TOL, label="logits, against a single pass")


# Single tokens from the first one: the calls with one and with two cached tokens.
@case("pieces of 1, 1, 1, and 5 tokens; the library's cache", public=True, stage=S3)
def single_tokens_first(ctx):
    cfg, params, tokens, f = shipped(ctx, "base")
    want = shipped_state(f, "base", cfg)
    logits = chunked(
        ctx, params, cfg, tokens, SINGLES, f["base_cached_1"], want, "against the library"
    )
    check.allclose(logits, f["base_one_pass"], **TOL, label="logits, against a single pass")


# A model with no attention layer has no cached keys to count and needs no position, and
# one with only attention layers has no recurrent state.
@case("a model with no attention layer, and one with nothing but attention layers",
      public=True, stage=S3)  # fmt: skip
def one_kind_only(ctx):
    for interval, what in ((6, "no attention layer"), (1, "only attention layers")):
        cfg = replace(BASE, full_attention_interval=interval)
        params, tokens = hybrid_params(cfg, 8), ctx.randint(0, cfg.vocab_size, (N,))
        want, want_state = hybrid_forward_ref(cfg, params, tokens)
        chunked(ctx, params, cfg, tokens, SINGLES, want, want_state, what)


def unchanged(ctx, tensors, arrays, what):
    for name, before in arrays.items():
        check.allclose(
            ctx.fw.numpy(tensors[name]), before, rtol=0, atol=0, label=f"{what}{name} after the call"
        )


# Rejects the mutants that write into what they were given. The caller keeps its state
# (to branch, to retry, to serve two continuations of one prefix).
@case("nothing passed in is modified: one state is continued twice", public=True, stage=S3)
def inputs_unchanged(ctx):
    cfg = MIXED
    params = hybrid_params(cfg, 7)
    tokens = ctx.randint(0, cfg.vocab_size, (N,))
    want, _ = hybrid_forward_ref(cfg, params, tokens)
    _, held = hybrid_forward_ref(cfg, params, tokens[:3])
    weights = ctx.fw.to_framework(params)
    state = [tuple(ctx.fw.tensor(part) for part in pair) for pair in held]
    entries = list(state)
    first = ctx.fw.tensor(tokens[3:])
    for _ in range(2):  # the same continuation twice, from the same state
        logits, _ = ctx.solution.hybrid_forward(weights, cfg, first, state)
        check.allclose(ctx.fw.numpy(logits), want[3:], **TOL, label="logits of a continuation")
        check.that(
            len(state) == len(entries) and all(a is b for a, b in zip(state, entries, strict=True)),
            "the state list passed in was changed",
        )
        for layer, pair in enumerate(held):
            unchanged(ctx, dict(enumerate(state[layer])), dict(enumerate(pair)),
                      f"state[{layer}], part ")  # fmt: skip
        check.allclose(ctx.fw.numpy(first), tokens[3:], rtol=0, atol=0, label="tokens")
    unchanged(ctx, weights, params, "params: ")
    check.that(set(weights) == set(params), "params: the dict has other keys after the call")


# The state is made by the kit, not by the solution: its layout is the statement's.
@case("continuing from a state the test supplies", stage=S3)
def supplied_state(ctx):
    cfg, params, tokens, _ = shipped(ctx, "mixed")
    want, want_state = hybrid_forward_ref(cfg, params, tokens)
    _, state = hybrid_forward_ref(cfg, params, tokens[:3])
    logits, got = call(ctx, params, cfg, tokens[3:], state)
    check.allclose(logits, want[3:], **TOL, label="logits")
    compare_state(got, want_state, cfg)


@case("the library with a cache: 8 layers", stage=S3)
def shipped_deep_cached(ctx):
    cfg, params, tokens, f = shipped(ctx, "deep")
    want = shipped_state(f, "deep", cfg)
    for index, sizes in enumerate(SHIPPED_CHUNKINGS):
        chunked(ctx, params, cfg, tokens, sizes, f[f"deep_cached_{index}"], want)


@case("the library with a cache: attention first", stage=S3)
def shipped_mixed_cached(ctx):
    cfg, params, tokens, f = shipped(ctx, "mixed")
    want = shipped_state(f, "mixed", cfg)
    for index, sizes in enumerate(SHIPPED_CHUNKINGS):
        chunked(ctx, params, cfg, tokens, sizes, f[f"mixed_cached_{index}"], want)


# The property of the statement, on fresh tokens and weights.
@case("a chunking of fresh tokens and weights equals a single pass", stage=S3)
def any_chunking(ctx):
    def prop(rng):
        cfg = (BASE, MIXED)[int(rng.integers(2))]
        sizes = (SINGLES, PREFILL)[int(rng.integers(2))]
        params = hybrid_params(cfg, int(rng.integers(100, 1000)))
        tokens = rng.integers(0, cfg.vocab_size, N)
        want, want_state = hybrid_forward_ref(cfg, params, tokens)
        chunked(ctx, params, cfg, tokens, sizes, want, want_state)

    check.forall(prop, trials=3)


@case("other settings with a cache", stage=S3)
def other_settings_cached(ctx):
    cfg = replace(BASE, rope_base=50.0, norm_eps=0.3)
    params, tokens = hybrid_params(cfg, 10), ctx.randint(0, cfg.vocab_size, (N,))
    want, want_state = hybrid_forward_ref(cfg, params, tokens)
    chunked(ctx, params, cfg, tokens, PREFILL, want, want_state)
