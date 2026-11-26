"""Tests for capstone-quantized-runner.

Every expectation is computed here, in numpy, from the statement's formulas, so each
stage is judged on its own: stage 1 against `quantize` below (codes exactly, except
where a quotient is within 1e-3 of a rounding tie), stages 2 and 3 on checkpoints that
`make_checkpoint` builds (the solution's own quantizer is not involved) against the
kit's float forward pass run on the dequantized weights, and stage 4 on the solution's
mixed checkpoint, its true size, and the KL divergence of its logits from the float
model's.

One model shape is used throughout (JAX compiles per shape): the kit decoder with
d_model 64, d_mlp 128, two layers, tied or untied, and token batches of shape (2, 8).

`AttentionRows` is a local helper, not part of the kit: it counts the query positions
that pass through `ops.attention` while the solution runs, which is how a generation
loop that ignores its cache is told from one that uses it.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import DecoderConfig, decoder_forward_ref, decoder_params, logit_kl, ops

S1 = "1: quantized checkpoint"
S2 = "2: forward pass on codes"
S3 = "3: cached generation"
S4 = "4: mixed recipe"

CFG = DecoderConfig(d_model=64, d_mlp=128, n_layers=2)
UNTIED = DecoderConfig(d_model=64, d_mlp=128, n_layers=2, tie_embeddings=False)
GROUP = 64
EIGHT_BIT = ("model.embed_tokens.weight", "lm_head.weight")
NEAR_TIE = 1e-3  # a quotient this close to k + 0.5 may round either way
MARGIN = 1e-2  # greedy tokens: the two largest logits are at least this far apart
KL_LIMIT = 0.14  # nats, stage 4
RTOL, ATOL = 1e-3, 2e-4

# The worked example of stage 1: the matrix of `groupwise-4bit` and a gain vector.
EXAMPLE = {
    "proj.weight": np.array(
        [
            [0.0, 0.33, -0.5, 1.0, 2.0, 2.2, 2.9, 2.6],
            [-1.0, -1.0, -1.0, -1.0, 0.5, -0.25, 0.1, 0.3],
        ],
        dtype=np.float32,
    ),
    "norm.weight": np.array([1.0, 0.5, 2.0, 1.5, 1.0, 1.0, 0.25, 4.0], dtype=np.float32),
}


# -- the specification, in numpy ----------------------------------------------------------


def quantize(w, bits, group_size):
    """(codes uint8, scale, bias, near): `near` marks quotients within NEAR_TIE of a tie."""
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo, hi = groups.min(axis=-1), groups.max(axis=-1)
    scale = np.where(hi > lo, (hi - lo) / np.float32(levels), np.float32(1.0)).astype(np.float32)
    codes = np.clip(np.round((groups - lo[..., None]) / scale[..., None]), 0, levels)
    exact = (groups.astype(np.float64) - lo[..., None]) / scale.astype(np.float64)[..., None]
    near = np.abs(exact - np.floor(exact) - 0.5) < NEAR_TIE
    return codes.astype(np.uint8).reshape(w.shape), scale, lo, near.reshape(w.shape)


def pack(codes, bits):
    if bits == 8:
        return codes
    pairs = codes.reshape(codes.shape[0], -1, 2)
    return (pairs[..., 0] + 16 * pairs[..., 1]).astype(np.uint8)


def unpack(packed, bits):
    if bits == 8:
        return packed
    return np.stack((packed % 16, packed // 16), axis=-1).reshape(packed.shape[0], -1)


def uniform(bits):
    return lambda name: bits


def mixed(name):
    return 8 if name in EIGHT_BIT else 4


def make_checkpoint(params, bits_of, group_size):
    """The checkpoint the statement specifies for `params`."""
    ckpt = {}
    for name, w in params.items():
        if w.ndim != 2:
            ckpt[name] = w
            continue
        bits = bits_of(name)
        codes, scale, bias, _ = quantize(w, bits, group_size)
        ckpt[name] = pack(codes, bits)
        ckpt[name[:-6] + "scales"] = scale
        ckpt[name[:-6] + "biases"] = bias
    return ckpt


def matrix_bits(ckpt, name, group_size):
    return 8 * ckpt[name].shape[1] // (ckpt[name[:-6] + "scales"].shape[1] * group_size)


def dequantize(ckpt, group_size):
    """Float parameters that a checkpoint stands for, under the original names."""
    params = {}
    for name, tensor in ckpt.items():
        if name.endswith((".scales", ".biases")):
            continue
        if tensor.dtype != np.uint8:
            params[name] = tensor
            continue
        scale, bias = ckpt[name[:-6] + "scales"], ckpt[name[:-6] + "biases"]
        codes = unpack(tensor, matrix_bits(ckpt, name, group_size)).astype(np.float32)
        groups = codes.reshape(codes.shape[0], -1, group_size)
        params[name] = (groups * scale[..., None] + bias[..., None]).reshape(codes.shape)
    return params


def stored_bytes(ckpt):
    return sum(int(tensor.nbytes) for tensor in ckpt.values())


def expected_bytes(params, bits_of, group_size):
    total = 0
    for name, w in params.items():
        if w.ndim != 2:
            total += 4 * w.size
        else:
            total += w.size * bits_of(name) // 8 + 2 * 4 * (w.size // group_size)
    return total


def model(cfg, seed):
    return decoder_params(cfg, seed)


def tokens_for(ctx, cfg, shape=(2, 8)):
    return ctx.randint(0, cfg.vocab_size, shape)


# -- stage 1 checks -----------------------------------------------------------------------


def check_checkpoint(got, params, bits_of, group_size):
    """Keys, dtypes, shapes, codes, scales, biases, and untouched gains."""
    check.that(isinstance(got, dict), "the checkpoint must be a dict", f"got {type(got).__name__}")
    want = make_checkpoint(params, bits_of, group_size)
    check.that(
        set(got) == set(want),
        "the checkpoint's keys are not the specified ones",
        f"missing: {sorted(set(want) - set(got))}\nunexpected: {sorted(set(got) - set(want))}",
    )
    for name, w in params.items():
        if w.ndim != 2:
            check.allclose(got[name], w, rtol=0, atol=0, label=name, dtype=True)
            continue
        bits = bits_of(name)
        packed = got[name]
        check.that(
            packed.dtype == np.uint8,
            f"{name}: the packed codes have dtype {packed.dtype}, expected dtype uint8",
        )
        check.shape(packed, want[name].shape, label=name)
        codes, scale, bias, near = quantize(w, bits, group_size)
        theirs = unpack(packed, bits).astype(np.int64)
        off = np.abs(theirs - codes.astype(np.int64))
        wrong = (off > near) | (off > 1)  # exact, except one step either way at a tie
        if wrong.any():
            index = tuple(int(i) for i in np.argwhere(wrong)[0])
            check.that(
                False,
                f"{name}: {int(wrong.sum())} of {wrong.size} codes differ from the specified "
                f"codes (first at {list(index)})",
                f"at {list(index)}: expected code {int(codes[index])}, got {int(theirs[index])} "
                f"(w = {w[index]!r}, bits = {bits}, group_size = {group_size})\n"
                f"expected first row of bytes: {want[name][0][:16].tolist()}\n"
                f"got first row of bytes:      {packed[0][:16].tolist()}",
            )
        check.allclose(got[name[:-6] + "scales"], scale, label=name[:-6] + "scales", dtype=True)
        check.allclose(got[name[:-6] + "biases"], bias, label=name[:-6] + "biases", dtype=True)


@case("the worked example: a 2 x 8 matrix and a gain vector at 4 bits", public=True, stage=S1)
def worked_example(ctx):
    got = ctx.call("quantize_checkpoint", EXAMPLE, 4, 4)
    check.that(isinstance(got, dict), "the checkpoint must be a dict")
    check.equal(
        sorted(got), ["norm.weight", "proj.biases", "proj.scales", "proj.weight"], label="keys"
    )
    check.equal(got["proj.weight"], [[133, 240, 48, 175], [0, 0, 15, 183]], label="proj.weight")
    check.that(got["proj.weight"].dtype == np.uint8, "proj.weight: the bytes must be uint8")
    check.allclose(got["proj.scales"], [[0.1, 0.06], [1.0, 0.05]], label="proj.scales")
    check.allclose(got["proj.biases"], [[-0.5, 2.0], [-1.0, -0.25]], label="proj.biases")
    check.allclose(got["norm.weight"], EXAMPLE["norm.weight"], rtol=0, atol=0, label="norm.weight")
    # The same matrix at 8 bits: one byte per code, 255 steps.
    check_checkpoint(ctx.call("quantize_checkpoint", EXAMPLE, 8, 4), EXAMPLE, uniform(8), 4)


@case("the kit decoder at 4 bits, groups of 64: every tensor of the checkpoint", public=True,
      stage=S1)  # fmt: skip
def kit_checkpoint_4bit(ctx):
    params = model(CFG, 1)
    check_checkpoint(ctx.call("quantize_checkpoint", params, 4, GROUP), params, uniform(4), GROUP)


@case("the kit decoder at 8 bits: one byte per code", public=True, stage=S1)
def kit_checkpoint_8bit(ctx):
    params = model(CFG, 1)
    check_checkpoint(ctx.call("quantize_checkpoint", params, 8, GROUP), params, uniform(8), GROUP)


@case("stored bytes: codes, scales, biases, and gains, and nothing else", public=True, stage=S1)
def stored_bytes_exact(ctx):
    # Rejects keeps_float_embedding and codes_not_packed: both store more than the format.
    params = model(CFG, 1)
    for bits in (4, 8):
        got = ctx.call("quantize_checkpoint", params, bits, GROUP)
        check.that(isinstance(got, dict), "the checkpoint must be a dict")
        check.at_most(
            f"stored_bytes_{bits}bit",
            stored_bytes(got),
            expected_bytes(params, uniform(bits), GROUP),
            unit="bytes",
        )


@case("error bound: every weight comes back within half a step of its group", public=True,
      stage=S1)  # fmt: skip
def error_bound(ctx):
    def prop(rng):
        bits = int(rng.choice([4, 8]))
        group_size = int(rng.choice([32, 64]))
        # Heavy-tailed rows, so that groups of one row have very different ranges.
        spread = np.exp(rng.normal(size=(64, 1))).astype(np.float32)
        w = (rng.standard_normal((64, 128)).astype(np.float32) * spread).astype(np.float32)
        got = ctx.call("quantize_checkpoint", {"w.weight": w}, bits, group_size)
        for key in ("w.weight", "w.scales", "w.biases"):
            check.that(key in got, f"the checkpoint has no key {key!r}")
        check.shape(got["w.weight"], (64, 128 * bits // 8), label="w.weight")
        check.shape(got["w.scales"], (64, 128 // group_size), label="w.scales")
        check.shape(got["w.biases"], (64, 128 // group_size), label="w.biases")
        w_hat = dequantize(got, group_size)["w.weight"]
        step = np.repeat(np.abs(got["w.scales"]), group_size, axis=1)
        error = np.abs(w_hat - w)
        worst = float((error / step).max())
        check.that(
            bool((error <= 0.5 * step * (1 + 1e-3) + 1e-6).all()),
            f"a weight comes back {worst:.2f} steps from its value, more than half a step",
            f"bits = {bits}, group_size = {group_size}",
        )
        groups = w.reshape(64, -1, group_size)
        hat = w_hat.reshape(64, -1, group_size)
        check.allclose(hat.min(axis=-1), groups.min(axis=-1), label="minimum of every group")
        check.allclose(hat.max(axis=-1), groups.max(axis=-1), label="maximum of every group")

    check.forall(prop, trials=4)


@case("constant groups, groups of 32, and an untied head", stage=S1)
def constant_groups(ctx):
    # Rejects constant_group_nan: a group of equal values has no range to divide by.
    params = dict(model(UNTIED, 2))
    w = params["model.layers.0.mlp.down_proj.weight"].copy()
    w[3, 32:64] = 0.25
    w[5, :] = 0.0
    w[7, 96:] = -1.5
    params["model.layers.0.mlp.down_proj.weight"] = w
    for bits in (4, 8):
        got = ctx.call("quantize_checkpoint", params, bits, 32)
        for name, tensor in got.items():
            check.finite(tensor, label=name)
        check_checkpoint(got, params, uniform(bits), 32)


@case("another model: bytes and codes at both widths", stage=S1)
def another_model(ctx):
    params = model(CFG, 5)
    for bits, group_size in ((4, 32), (8, 64)):
        got = ctx.call("quantize_checkpoint", params, bits, group_size)
        check_checkpoint(got, params, uniform(bits), group_size)
        check.at_most(
            f"stored_bytes_{bits}bit",
            stored_bytes(got),
            expected_bytes(params, uniform(bits), group_size),
            unit="bytes",
        )


# -- stage 2: the forward pass ------------------------------------------------------------


def check_forward(ctx, cfg, seed, bits_of, group_size, tokens=None):
    ckpt = make_checkpoint(model(cfg, seed), bits_of, group_size)
    if tokens is None:
        tokens = tokens_for(ctx, cfg)
    got = ctx.call("forward_q", ckpt, cfg, tokens, group_size)
    want = decoder_forward_ref(cfg, dequantize(ckpt, group_size), tokens)
    check.allclose(got, want, rtol=RTOL, atol=ATOL, label="logits", dtype=True)
    return got, want


@case("4-bit checkpoint, groups of 64: the logits of the dequantized model", public=True,
      stage=S2)  # fmt: skip
def forward_4bit(ctx):
    check_forward(ctx, CFG, 1, uniform(4), GROUP, np.array([[0, 17, 42, 5, 9, 33, 17, 60]] * 2))
    check_forward(ctx, CFG, 1, uniform(4), GROUP)


@case("8-bit checkpoint: the width of a row of bytes gives the bits", public=True, stage=S2)
def forward_8bit(ctx):
    check_forward(ctx, CFG, 1, uniform(8), GROUP)


@case("groups of 32: two to four groups per row, each with its own scale and bias", public=True,
      stage=S2)  # fmt: skip
def forward_groups_of_32(ctx):
    # Rejects embedding_one_scale_per_row (with groups of 64 a row of the embedding is a
    # single group and the mistake vanishes) and bias_times_row_sum.
    check_forward(ctx, CFG, 1, uniform(4), 32)
    check_forward(ctx, CFG, 3, uniform(8), 32)


@case("a checkpoint whose matrices differ in bits", stage=S2)
def forward_mixed_bits(ctx):
    # Rejects assumes_four_bits: the bits are a property of each matrix.
    check_forward(ctx, CFG, 2, mixed, GROUP)
    check_forward(ctx, CFG, 2, lambda name: 4 if name in EIGHT_BIT else 8, 32)


@case("an untied head: lm_head.weight has its own codes", stage=S2)
def forward_untied(ctx):
    check_forward(ctx, UNTIED, 4, uniform(4), GROUP)
    check_forward(ctx, UNTIED, 4, mixed, 32)


@case("the forward pass is causal and the rows of a batch are independent", stage=S2)
def forward_causal(ctx):
    ckpt = make_checkpoint(model(CFG, 6), uniform(4), GROUP)
    tokens = tokens_for(ctx, CFG)
    changed = tokens.copy()
    changed[0, 5] = (changed[0, 5] + 7) % CFG.vocab_size
    a = ctx.call("forward_q", ckpt, CFG, tokens, GROUP)
    b = ctx.call("forward_q", ckpt, CFG, changed, GROUP)
    check.allclose(b[0, :5], a[0, :5], atol=1e-5, label="logits before the changed token")
    check.allclose(b[1], a[1], atol=1e-5, label="logits of the other row")
    check.that(
        float(np.abs(b[0, 5:] - a[0, 5:]).max()) > 1e-3,
        "changing a token left the logits at and after its position unchanged",
    )


# -- stage 3: the cached step and generation ----------------------------------------------


class AttentionRows:
    """Counts the query positions passed to `ops.attention` while it is installed."""

    def __init__(self, ctx):
        self.rows = 0
        original = ops.attention

        def counted(q, k, v, mask=None):
            shape = tuple(q.shape)
            self.rows += int(np.prod(shape[:-3])) * shape[-2] if len(shape) >= 3 else 1
            return original(q, k, v, mask)

        ops.attention = counted
        self._restore = lambda: setattr(ops, "attention", original)
        ctx.defer(self._restore)

    def stop(self):
        self._restore()
        return self.rows


def greedy(cfg, ckpt, group_size, context, n):
    """The definition of generation: n argmax tokens, each from a full float pass."""
    params = dequantize(ckpt, group_size)
    sequence, out = list(context), []
    for _ in range(n):
        row = decoder_forward_ref(cfg, params, np.array(sequence))[-1]
        top = np.sort(row)[-2:]
        if top[1] - top[0] < MARGIN:
            raise RuntimeError("authoring error: a near-tie in a greedy fixture")
        out.append(int(row.argmax()))
        sequence.append(out[-1])
    return out


def check_tokens(got, want):
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        "generate_q: must return a list of Python ints",
        f"got {got!r}",
    )
    check.equal(got, want, label="generated tokens")


def run_chunks(ctx, cfg, ckpt, group_size, tokens, sizes):
    """Feed `tokens` (B, T) through step_q in chunks; returns (logits, cache)."""
    cache, rows, start = None, [], 0
    for size in sizes:
        logits, cache = ctx.call("step_q", ckpt, cfg, tokens[:, start : start + size], cache,
                                 group_size)  # fmt: skip
        check.shape(logits, (tokens.shape[0], size, cfg.vocab_size), label="logits of a chunk")
        rows.append(logits)
        start += size
    return np.concatenate(rows, axis=1), cache


def check_cache(ctx, cfg, ckpt, group_size, tokens, cache):
    """The cache after `tokens`: per layer, rotated keys and values of every token."""
    _, want = ctx.expected("step_q", ckpt, cfg, tokens, None, group_size)
    check.that(
        isinstance(cache, (list, tuple)) and len(cache) == cfg.n_layers,
        "the cache must have one (k, v) pair per layer",
        f"got {type(cache).__name__} of length {len(cache) if hasattr(cache, '__len__') else '?'}",
    )
    for i, (pair, wanted) in enumerate(zip(cache, want, strict=True)):
        check.that(len(pair) == 2, f"cache[{i}] must be a pair (k, v)")
        check.allclose(pair[0], wanted[0], rtol=RTOL, atol=ATOL, label=f"cache[{i}] keys")
        check.allclose(pair[1], wanted[1], rtol=RTOL, atol=ATOL, label=f"cache[{i}] values")


@case("a sequence fed as 5 tokens, then 3: the logits and the cache of one full pass",
      public=True, stage=S3)  # fmt: skip
def step_two_chunks(ctx):
    # Rejects positions_restart, chunk_without_causal_mask, and attends_cache_only.
    ckpt = make_checkpoint(model(CFG, 1), uniform(4), GROUP)
    tokens = tokens_for(ctx, CFG)
    want = decoder_forward_ref(CFG, dequantize(ckpt, GROUP), tokens)
    whole, cache = run_chunks(ctx, CFG, ckpt, GROUP, tokens, [8])
    check.allclose(whole, want, rtol=RTOL, atol=ATOL, label="logits, one call")
    check_cache(ctx, CFG, ckpt, GROUP, tokens, cache)
    got, cache = run_chunks(ctx, CFG, ckpt, GROUP, tokens, [5, 3])
    check.allclose(got, want, rtol=RTOL, atol=ATOL, label="logits, chunks of 5 and 3")
    check_cache(ctx, CFG, ckpt, GROUP, tokens, cache)


PROMPT = [0, 5, 24, 41, 8]  # the 4-bit model continues it with 51, 2, 52, 53


@case("greedy generation: 4 tokens after a 5-token prompt", public=True, stage=S3)
def generate_greedy(ctx):
    ckpt = make_checkpoint(model(CFG, 1), uniform(4), GROUP)
    want = greedy(CFG, ckpt, GROUP, PROMPT, 4)
    if want != [51, 2, 52, 53]:
        raise RuntimeError("authoring error: the statement's generation example changed")
    check_tokens(ctx.call("generate_q", ckpt, CFG, list(PROMPT), 4, GROUP), want)
    check_tokens(ctx.call("generate_q", ckpt, CFG, list(PROMPT), 1, GROUP), want[:1])


@case("cost: every token passes through attention once per layer", public=True, stage=S3)
def generate_cost(ctx):
    # Rejects cache_bypassed: recomputing the sequence at every step asks attention for
    # 2 * (5 + 6 + 7 + 8) = 52 query positions instead of 2 * 8 = 16.
    ckpt = make_checkpoint(model(CFG, 1), uniform(4), GROUP)
    want = greedy(CFG, ckpt, GROUP, PROMPT, 4)
    counter = AttentionRows(ctx)
    got = ctx.call("generate_q", ckpt, CFG, list(PROMPT), 4, GROUP)
    rows = counter.stop()
    check_tokens(got, want)
    check.at_most("attention_queries", rows, CFG.n_layers * (len(PROMPT) + 4 - 1), unit="rows")


@case("generation continues a cache whose tokens it never sees", stage=S3)
def generate_from_prefix_cache(ctx):
    # A prefix prefilled elsewhere: only its keys and values are available, so the
    # loop cannot recompute it. Also rejects positions_restart on the decode path.
    for cfg, seed, bits_of in ((CFG, 2, uniform(4)), (UNTIED, 4, mixed)):
        ckpt = make_checkpoint(model(cfg, seed), bits_of, GROUP)
        prefix = [int(t) for t in ctx.randint(0, cfg.vocab_size, 2)]
        prompt = [int(t) for t in ctx.randint(0, cfg.vocab_size, 3)]
        _, cache = ctx.expected("step_q", ckpt, cfg, np.array([prefix]), None, GROUP)
        want = greedy(cfg, ckpt, GROUP, prefix + prompt, 4)
        counter = AttentionRows(ctx)
        got = ctx.call("generate_q", ckpt, cfg, prompt, 4, GROUP, cache)
        rows = counter.stop()
        check_tokens(got, want)
        check.at_most("attention_queries", rows, cfg.n_layers * (len(prompt) + 4 - 1), unit="rows")


@case("other chunkings: a prefill, then one token at a time", stage=S3)
def step_many_chunkings(ctx):
    ckpt = make_checkpoint(model(CFG, 3), mixed, 32)
    for shape, sizes in (((1, 8), [5, 1, 1, 1]), ((2, 8), [5, 3])):
        tokens = tokens_for(ctx, CFG, shape)
        want = decoder_forward_ref(CFG, dequantize(ckpt, 32), tokens)
        got, cache = run_chunks(ctx, CFG, ckpt, 32, tokens, sizes)
        check.allclose(got, want, rtol=RTOL, atol=ATOL, label=f"logits, chunks {sizes}")
        check_cache(ctx, CFG, ckpt, 32, tokens, cache)


@case("a step leaves the cache it was given unchanged", stage=S3)
def step_keeps_old_cache(ctx):
    ckpt = make_checkpoint(model(CFG, 1), uniform(4), GROUP)
    tokens = tokens_for(ctx, CFG)
    fw = ctx.fw
    ckpt_t = fw.to_framework(ckpt)
    _, cache = ctx.solution.step_q(ckpt_t, CFG, fw.tensor(tokens[:, :5]), None, GROUP)
    before = fw.to_numpy([tuple(pair) for pair in cache])
    first, _ = ctx.solution.step_q(ckpt_t, CFG, fw.tensor(tokens[:, 5:]), cache, GROUP)
    after = fw.to_numpy([tuple(pair) for pair in cache])
    for i, (old, new) in enumerate(zip(before, after, strict=True)):
        check.shape(new[0], old[0].shape, label=f"cache[{i}] keys after a later step")
        check.allclose(new[0], old[0], rtol=0, atol=0, label=f"cache[{i}] keys after a later step")
        check.allclose(
            new[1], old[1], rtol=0, atol=0, label=f"cache[{i}] values after a later step"
        )
    # Branching: the same cache continued a second time gives the same logits.
    second, _ = ctx.solution.step_q(ckpt_t, CFG, fw.tensor(tokens[:, 5:]), cache, GROUP)
    check.allclose(fw.numpy(second), fw.numpy(first), atol=1e-5, label="logits of a second branch")


# -- stage 4: the mixed recipe ------------------------------------------------------------


def true_bpw(ckpt, params):
    return 8 * stored_bytes(ckpt) / sum(w.size for w in params.values())


def mixed_run(ctx, cfg, seed, group_size, max_bpw, kl=True, layout=True):
    """Quantize with the recipe; check layout, size, and the KL of the logits."""
    params = model(cfg, seed)
    got = ctx.call("quantize_mixed", params, group_size, max_bpw)
    if layout:
        check_checkpoint(got, params, mixed, group_size)
    check.that(isinstance(got, dict), "the checkpoint must be a dict")
    check.at_most("bits_per_weight", true_bpw(got, params), max_bpw, unit="bits")
    if not kl:
        return
    # 64 positions: four batches of the one token shape the tests use.
    batches = np.random.default_rng([seed, 99]).integers(0, cfg.vocab_size, (4, 2, 8))
    four = dequantize(make_checkpoint(params, uniform(4), group_size), group_size)
    kl_recipe, kl_four = [], []
    for tokens in batches:
        reference = decoder_forward_ref(cfg, params, tokens)
        kl_four.append(logit_kl(reference, decoder_forward_ref(cfg, four, tokens)))
        logits = ctx.call("forward_q", got, cfg, tokens, group_size)
        check.shape(logits, reference.shape, label="logits")
        check.finite(logits, label="logits")
        kl_recipe.append(logit_kl(reference, logits))
    kl_recipe, kl_four = float(np.mean(kl_recipe)), float(np.mean(kl_four))
    if kl_four < 1.15 * KL_LIMIT:
        raise RuntimeError(f"authoring error: uniform 4-bit has KL {kl_four:.4f}")
    if 0.85 * KL_LIMIT < kl_recipe <= KL_LIMIT:
        raise RuntimeError(f"authoring error: the recipe's KL {kl_recipe:.4f} is near the limit")
    check.at_most("logit_kl", kl_recipe, KL_LIMIT, unit="nats")


@case("the recipe's checkpoint: 8-bit embedding, 4-bit everywhere else", public=True, stage=S4)
def mixed_layout(ctx):
    params = model(CFG, 1)
    got = ctx.call("quantize_mixed", params, GROUP, 5.5)
    check_checkpoint(got, params, mixed, GROUP)
    check.that(
        got["model.embed_tokens.weight"].shape == (64, 64)
        and got["model.layers.0.self_attn.q_proj.weight"].shape == (64, 32),
        "the embedding must have one byte per code and the other matrices half a byte",
    )


@case("bits per weight: scales, biases, and gains are part of the size", public=True, stage=S4)
def bits_per_weight_values(ctx):
    # Rejects budget_ignores_metadata.
    params = model(CFG, 1)
    for bits_of, group_size in ((uniform(4), 64), (uniform(8), 64), (mixed, 64), (mixed, 32)):
        ckpt = make_checkpoint(params, bits_of, group_size)
        got = ctx.call("bits_per_weight", ckpt, group_size)
        check.that(type(got) is float, "bits_per_weight: must return a Python float",
                   f"got {type(got).__name__}")  # fmt: skip
        check.allclose(got, true_bpw(ckpt, params), rtol=1e-9, atol=0, label="bits_per_weight")
    ckpt = make_checkpoint(params, uniform(4), GROUP)
    check.allclose(ctx.call("bits_per_weight", ckpt, GROUP), 399360 / 78144, rtol=1e-9,
                   label="bits_per_weight of the 4-bit kit checkpoint")  # fmt: skip


@case("the recipe fits 5.5 bits per weight and keeps the KL under 0.14 nats", public=True,
      stage=S4)  # fmt: skip
def mixed_budget_and_kl(ctx):
    # The solution's own checkpoint through its own forward pass. Rejects recipe_ignored:
    # uniform 4-bit is under the budget and over the KL limit (0.176 nats here).
    mixed_run(ctx, CFG, 3, GROUP, 5.5, layout=False)


@case("a budget the recipe does not fit is refused; one it fits exactly is not", stage=S4)
def mixed_over_budget(ctx):
    # Rejects budget_ignores_metadata: without scales and biases the recipe seems to fit.
    params = model(CFG, 1)
    exact = true_bpw(make_checkpoint(params, mixed, GROUP), params)
    check.raises(ValueError, ctx.call, "quantize_mixed", params, GROUP, 5.25, label="max_bpw 5.25")
    check.raises(ValueError, ctx.call, "quantize_mixed", params, GROUP, exact - 1e-6,
                 label="max_bpw just under the size")  # fmt: skip
    got = ctx.call("quantize_mixed", params, GROUP, exact)
    check_checkpoint(got, params, mixed, GROUP)
    check.raises(ValueError, ctx.call, "quantize_mixed", params, 32, 6.0, label="groups of 32")


@case("an untied model: the head is a matrix of its own", stage=S4)
def mixed_untied(ctx):
    # Rejects head_left_at_four_bits.
    mixed_run(ctx, UNTIED, 11, GROUP, 5.6)
    params = model(UNTIED, 11)
    ckpt = make_checkpoint(params, mixed, GROUP)
    check.allclose(ctx.call("bits_per_weight", ckpt, GROUP), true_bpw(ckpt, params), rtol=1e-9,
                   label="bits_per_weight")  # fmt: skip


@case("other models and groups of 32", stage=S4)
def mixed_other_models(ctx):
    mixed_run(ctx, CFG, 7, GROUP, 5.5)
    mixed_run(ctx, CFG, 2, 32, 6.4, kl=False)
