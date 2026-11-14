"""Tests for mlx-real-model-forward.

Every expected logit comes from `transformers` (float32, CPU), stored by
`fixtures/make.py`; the reference solution is never the oracle.

- The cases marked `assets=[]` use two three-layer Granite models with random weights
  and a 64-token vocabulary: `a` with Granite 4.0's multipliers and a tied head, `b`
  with other multipliers, RoPE base 10,000, another epsilon, and an untied head. Their
  weights are drawn here from the same seeds as in `make.py`, with values bfloat16
  represents exactly, and written to a bfloat16 safetensors file in a temporary
  directory: the same kind of file as the real checkpoint.
- The other cases load the real granite-4.0-350m checkpoint (705 MB of bfloat16).

In both, the judge does what a user of the class would do:

    model = Granite(cfg)
    model.load_weights(path, strict=True)     # mlx.nn.Module.load_weights, unchanged
    model.set_dtype(mx.float32)               # or mx.float16
    logits = model(tokens)

Nothing here needs the device, so no case is marked `mlx-gpu`.
"""

import io
import re
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx import nn
from whetstone.judge import case, check
from whetstone.kit import RealConfig, RealModel

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
TOL = {"rtol": 1e-3, "atol": 1e-3}  # float32, as stated in Constraints
# Absolute tolerances of the 16-bit dtypes: measured errors are 0.07 and 0.52.
HALF_ATOL = {mx.float16: 0.25, mx.bfloat16: 1.0}
MARGIN = {mx.float32: 0.2, mx.bfloat16: 2.0}  # argmax is compared beyond this margin

TINY = {"vocab": 64, "d_model": 256, "n_heads": 4, "n_kv_heads": 2, "d_ff": 96, "n_layers": 3}
# logits[0, 5, :5] of the worked example, as printed in the statement
WORKED = [-3.320329, -0.238478, -5.581573, 13.10126, -0.75092]
_cache: dict[object, object] = {}


def tiny_weights(seed: int, tied: bool) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout, exact in bfloat16.

    Keep in step with make.py.
    """
    rng = np.random.default_rng(seed)
    d, f, vocab = TINY["d_model"], TINY["d_ff"], TINY["vocab"]
    kv_width = d * TINY["n_kv_heads"] // TINY["n_heads"]

    def bf16(values: np.ndarray) -> np.ndarray:
        bits = values.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
        return bits.view(np.float32)

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return bf16(rng.standard_normal((rows, cols)) * std)

    def gain() -> np.ndarray:
        return bf16(rng.uniform(0.5, 1.5, d))

    weights = {"model.embed_tokens.weight": normal(vocab, d, 1.0)}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(d, d, 0.3)
        weights[p + "self_attn.k_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.v_proj.weight"] = normal(kv_width, d, 0.1)
        weights[p + "self_attn.o_proj.weight"] = normal(d, d, 0.1)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.1)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.1)
    weights["model.norm.weight"] = gain()
    if not tied:
        weights["lm_head.weight"] = normal(vocab, d, 1.0)
    return weights


def graph_ops(array):
    """Names of the operations still pending behind `array` (empty: evaluated)."""
    dot = io.StringIO()
    mx.export_to_dot(dot, array)
    return re.findall(r'label ="(\w+)"', dot.getvalue())


def scratch(ctx):
    """A temporary directory for the running case, removed when the case ends."""
    key = ("dir", id(ctx))
    if key not in _cache:
        path = Path(tempfile.mkdtemp(prefix="whetstone-granite-"))
        _cache[key] = path

        def remove():
            shutil.rmtree(path, ignore_errors=True)
            _cache.pop(key, None)

        ctx.defer(remove)
    return _cache[key]


def tiny(ctx, name):
    """(checkpoint path, cfg, fixture) of the random model `a` or `b`."""
    key = f"tiny_{name}"
    if key not in _cache:
        spec = ctx.fixture("tiny-configs.json")[name]
        cfg = RealConfig.from_json(spec["config"])
        weights = tiny_weights(spec["seed"], cfg.tied_embeddings)
        total = float(sum(np.abs(w.astype(np.float64)).sum() for w in weights.values()))
        fixture = ctx.fixture("tiny.npz")
        check.that(
            abs(total - float(fixture[f"checksum_{name}"])) < 1e-6 * total,
            "test setup: the random weights differ from the ones the fixture was made with "
            "(numpy changed its generator?); rerun fixtures/make.py",
        )
        stored = {k: mx.array(w).astype(mx.bfloat16) for k, w in weights.items()}
        mx.eval(stored)
        _cache[key] = (stored, cfg, fixture)
    stored, cfg, fixture = _cache[key]
    # The same kind of file as the real checkpoint: bfloat16 safetensors.
    path = scratch(ctx) / f"tiny-{name}.safetensors"
    if not path.exists():
        mx.save_safetensors(str(path), stored, metadata={"format": "pt"})
    return path, cfg, fixture


def build(ctx, cfg, path, dtype):
    """Granite(cfg), loaded from the checkpoint at `path` and converted to `dtype`."""
    model = ctx.solution.Granite(cfg)
    check.that(isinstance(model, nn.Module), "Granite: must be an mlx.nn.Module")
    stored = mx.load(str(path))  # lazy: names and shapes only
    have = {name: value.shape for name, value in tree_flatten(model.parameters())}
    missing = sorted(set(stored) - set(have))
    extra = sorted(set(have) - set(stored))
    check.that(
        not missing and not extra,
        f"Granite: the parameter names of the module tree are not the tensor names of the "
        f"checkpoint ({len(missing)} tensors have no parameter, {len(extra)} parameters "
        "have no tensor)",
        f"tensors without a parameter, first 5 of {len(missing)}: {missing[:5]}\n"
        f"parameters without a tensor, first 5 of {len(extra)}: {extra[:5]}",
    )
    wrong = sorted(name for name in have if tuple(have[name]) != tuple(stored[name].shape))
    check.that(
        not wrong,
        f"Granite: {len(wrong)} parameters do not have the shape of their tensor in the "
        f"checkpoint, for example {wrong[0] if wrong else ''}",
        "\n".join(f"{n}: parameter {have[n]}, checkpoint {stored[n].shape}" for n in wrong[:5]),
    )
    model.load_weights(str(path), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def forward(model, tokens, dtype, shape, label):
    """model(tokens) as a NumPy array, after checking type, dtype, and shape."""
    out = model(mx.array(tokens))
    check.that(
        isinstance(out, mx.array), f"{label}: returned a {type(out).__name__}, not an mx.array"
    )
    check.that(
        out.dtype == dtype,
        f"{label}: got dtype {out.dtype}, which is not the expected dtype",
        f"expected {dtype}, the dtype of the parameters",
    )
    got = np.array(out.astype(mx.float32))
    check.shape(got, shape, label=label)
    return got


def run_tiny(ctx, name, which, dtype=mx.float32, label=None):
    path, cfg, fixture = tiny(ctx, name)
    model = build(ctx, cfg, path, dtype)
    label = label or f"logits of model {name}"
    got = forward(model, fixture[f"tokens_{which}"], dtype, (2, 6, TINY["vocab"]), label)
    want = fixture[f"logits_{name}_{which}"]
    if dtype == mx.float32:
        check.allclose(got, want, **TOL, label=label)
    else:
        check.allclose(got, want, rtol=0.0, atol=HALF_ATOL[dtype], label=label)
    return got


# ---------------------------------------------------------------------------------
# Two three-layer random models: no download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: a three-layer model with Granite's multipliers and a tied head",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    got = run_tiny(ctx, "a", 0)
    check.allclose(got[0, 5, :5], WORKED, **TOL, label="logits of model a at [0, 5, :5]")


# Rejects rope_base_default's twin (a hard-coded 1e7), head_always_embedding, and any
# hard-coded multiplier or epsilon.
@case("another configuration: other multipliers, RoPE base 10,000, an untied head",
      public=True, assets=[])  # fmt: skip
def other_configuration(ctx):
    run_tiny(ctx, "b", 0)


# Rejects manual_attention: the same numbers without the fused kernel.
@case("the forward pass is built from the fused kernels: attention, RoPE, RMSNorm",
      public=True, assets=[])  # fmt: skip
def fused_kernels(ctx):
    path, cfg, fixture = tiny(ctx, "a")
    model = build(ctx, cfg, path, mx.float32)
    tokens = mx.array(fixture["tokens_0"])
    mx.eval(tokens)
    ops = graph_ops(model(tokens))
    n = cfg.n_layers
    for name, count, what in (
        ("ScaledDotProductAttention", n, "mx.fast.scaled_dot_product_attention, once per layer"),
        ("RoPE", 2 * n, "nn.RoPE or mx.fast.rope, on the queries and the keys of every layer"),
        (
            "RMSNorm",
            2 * n + 1,
            "nn.RMSNorm or mx.fast.rms_norm, twice per layer and once at the end",
        ),
    ):
        check.that(
            ops.count(name) == count,
            f"Granite: the graph of a forward pass holds {ops.count(name)} {name} operations, "
            f"which is not what {what} gives",
            f"expected {count} for {n} layers; operations in the graph: {sorted(set(ops))}",
        )


# Rejects float32_constants: an array constant promotes a 16-bit model to float32.
@case("16-bit models keep their dtype: float16 to 0.25, bfloat16 (as stored) to 1.0",
      public=True, assets=[])  # fmt: skip
def sixteen_bits(ctx):
    run_tiny(ctx, "a", 0, mx.float16, "logits of model a in float16")
    run_tiny(ctx, "b", 0, mx.float16, "logits of model b in float16")
    run_tiny(ctx, "a", 0, mx.bfloat16, "logits of model a in bfloat16")
    run_tiny(ctx, "b", 0, mx.bfloat16, "logits of model b in bfloat16")


# Rejects no_causal_mask without a reference.
@case("causality: changing tokens from position t on leaves the logits before t unchanged",
      assets=[])  # fmt: skip
def causality(ctx):
    path, cfg, fixture = tiny(ctx, "a")
    model = build(ctx, cfg, path, mx.float32)

    def prop(rng):
        tokens = fixture["tokens_0"].copy()
        t = int(rng.integers(1, 6))
        changed = tokens.copy()
        changed[:, t:] = (tokens[:, t:] + rng.integers(1, 64, (2, 6 - t))) % 64
        before = forward(model, tokens, mx.float32, (2, 6, 64), "logits")
        after = forward(model, changed, mx.float32, (2, 6, 64), "logits")
        check.allclose(after[:, :t], before[:, :t], rtol=1e-4, atol=1e-4,
                       label="logits before the first changed position")  # fmt: skip
        moved = np.abs(after[:, t:] - before[:, t:]).max(axis=-1)
        check.that(
            bool(np.all(moved > 1e-3)),
            "the logits at the changed positions did not change",
            f"largest change per row and position: {moved}",
        )

    check.forall(prop, trials=3)


@case("repeated tokens and the largest token id, in both configurations", assets=[])
def repeated_tokens(ctx):
    run_tiny(ctx, "a", 1)
    run_tiny(ctx, "b", 1)


@case("one row and one token: shapes (1, 6) and (2, 1)", assets=[])
def small_inputs(ctx):
    path, cfg, fixture = tiny(ctx, "b")
    model = build(ctx, cfg, path, mx.float32)
    tokens, want = fixture["tokens_0"], fixture["logits_b_0"]
    got = forward(model, tokens[:1], mx.float32, (1, 6, 64), "logits")
    check.allclose(got, want[:1], **TOL, label="logits")
    got = forward(model, tokens[:, :1], mx.float32, (2, 1, 64), "logits")
    check.allclose(got, want[:, :1], **TOL, label="logits")


# ---------------------------------------------------------------------------------
# granite-4.0-350m: 28 layers, 100,352 logits per position.
# ---------------------------------------------------------------------------------


def real_model(ctx, dtype):
    """The solution's model with the real weights in `dtype`, built once per run."""
    key = ("real", str(dtype))
    if key not in _cache:
        directory = ctx.asset(ASSET)
        cfg = RealModel(directory).config
        _cache[key] = build(ctx, cfg, ctx.asset(ASSET, "model.safetensors"), dtype)
    return _cache[key]


def run_real(ctx, indices, *, hidden, dtype=mx.float32):
    """Logits of the prompts `indices` (one batch), checked against the fixture."""
    fixture = ctx.fixture("granite-4.0-350m.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    model = real_model(ctx, dtype)
    ids = np.stack([fixture[f"ids_{index}"] for index in indices])
    batch = forward(model, ids, dtype, (len(indices), 12, 100352), "logits")
    for row, index in enumerate(indices):
        got = batch[row]
        what = "the prompt" if hidden else repr(ctx.fixture("prompts.json")[index])
        tol = TOL if dtype == mx.float32 else {"rtol": 0.0, "atol": HALF_ATOL[dtype]}
        check.allclose(got[:, fixture["columns"]], fixture[f"logits_{index}"], **tol,
                       label=f"logits for {what}, at 64 vocabulary columns")  # fmt: skip
        clear = fixture[f"margin_{index}"] > MARGIN[dtype]
        check.equal(got.argmax(-1)[clear], fixture[f"argmax_{index}"][clear],
                    label=f"argmax tokens for {what}")  # fmt: skip
        top = fixture[f"top_ids_{index}"]
        check.allclose(got[-1, top], fixture[f"top_logits_{index}"], **tol,
                       label=f"the 8 largest logits of the last position for {what}")  # fmt: skip
    return batch


@case("Granite 4.0 350M: 'The capital of France is Paris. The capital of Japan is' -> ' Tokyo'",
      public=True)  # fmt: skip
def real_capital(ctx):
    got = run_real(ctx, [0], hidden=False)
    check.equal(
        int(got[0, -1].argmax()), 27286, label="the most likely next token (' Tokyo' is 27286)"
    )


# float16 is not an option for this checkpoint: see the editorial.
@case("Granite 4.0 350M in bfloat16, as the checkpoint stores it: the same prompt, to 1.0",
      public=True)  # fmt: skip
def real_bfloat16(ctx):
    got = run_real(ctx, [0], hidden=False, dtype=mx.bfloat16)
    check.equal(
        int(got[0, -1].argmax()), 27286, label="the most likely next token (' Tokyo' is 27286)"
    )
    _cache.pop(("real", str(mx.bfloat16)), None)  # free 0.7 GB


@case("Granite 4.0 350M: a line of Python and a chat prompt, as one batch of two")
def real_batch(ctx):
    run_real(ctx, [1, 2], hidden=True)


@case("Granite 4.0 350M: emoji and Japanese")
def real_unicode(ctx):
    run_real(ctx, [3], hidden=True)
    _cache.pop(("real", str(mx.float32)), None)
