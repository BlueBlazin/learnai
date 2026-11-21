"""Tests for real-hybrid-logits.

Every expected logit comes from `transformers` (`Qwen3_5ForCausalLM`, float32, CPU,
one call), stored by `fixtures/make.py`; the reference solution is never the oracle.

- The cases marked `assets=[]` run small models with random weights in the checkpoint's
  layout (`checkpoint_layout(hybrid_params(cfg, seed))`). Models `a` and `b` are judged
  against `tiny.npz`, the logits a tiny `Qwen3_5ForCausalLM` gives with the same
  weights: `a` has the real model's period (linear, linear, linear, full) and RoPE
  settings, `b` an explicit schedule of six layers that starts with attention, RoPE base
  10,000, half of each head rotating, and a norm epsilon of 0.01, large enough that
  every norm shows whether it read `cfg.norm_eps`.
- Two more synthetic cases draw their weights, configuration, and tokens while they
  run and are judged against the kit's numpy forward pass, `hybrid_forward_ref`, on the
  same weights (the kit's unit tests hold it against `transformers`). Nothing on disk
  holds their answers, so a solution that replays the fixture files fails them.
- All synthetic models have the same tensor sizes and see 6 tokens (and once 1 token),
  so JAX compiles two sets of shapes.
- The other cases run the real Qwen3.5 0.8B on two prompts of exactly 12 tokens (one
  more set of shapes), one public and one hidden. The 752 M weights of the text decoder
  are decoded from bfloat16 and handed to the framework once per run, in the first of
  these cases. Two and not more: an eager JAX forward pass of this model takes 0.45 s
  whatever the prompt length (1.4 s the first time, with the decoding), and with a
  third real case the JAX reference crosses the 3 s that `learnai validate` warns at.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import (
    HybridConfig,
    RealHybrid,
    checkpoint_layout,
    hybrid_forward_ref,
    hybrid_params,
)

FN = "qwen_logits"
ASSET = "qwen3.5-0.8b"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TOL = {"rtol": 1e-3, "atol": 1e-3}  # the tolerance stated in Constraints
MARGIN = 0.01  # argmax is compared where the two largest fixture logits are further apart

SEED = 251
FULL, LINEAR = "full_attention", "linear_attention"
# Keep in step with fixtures/make.py.
CONFIGS = {
    "a": HybridConfig(n_key_heads=4, rope_base=1e7),
    "b": HybridConfig(
        n_layers=6, n_key_heads=4, rope_base=1e4, partial_rotary_factor=0.5, norm_eps=1e-2,
        layer_types=(FULL, LINEAR, LINEAR, FULL, LINEAR, FULL),
    ),
}  # fmt: skip
VOCAB = 64
# logits[5, :5] of the worked example, as printed in the statement
WORKED = [-3.244581, 1.593436, 2.538823, 0.680665, -0.069021]
_cache: dict[str, object] = {}


def tiny(ctx, name):
    """(checkpoint-layout weights, cfg, fixture) of the random model `a` or `b`."""
    key = f"tiny_{name}"
    if key not in _cache:
        cfg = CONFIGS[name]
        params = hybrid_params(cfg, SEED)
        total = float(sum(np.abs(w.astype(np.float64)).sum() for w in params.values()))
        fixture = ctx.fixture("tiny.npz")
        check.that(
            abs(total - float(fixture[f"checksum_{name}"])) < 1e-6 * total,
            "test setup: the random weights differ from the ones the fixture was made with "
            "(numpy changed its generator?); rerun fixtures/make.py",
        )
        _cache[key] = (checkpoint_layout(params), cfg, fixture)
    return _cache[key]


def run_tiny(ctx, name, which, weights=None, label=None):
    own, cfg, fixture = tiny(ctx, name)
    tokens = fixture[f"tokens_{which}"]
    got = ctx.call(FN, own if weights is None else weights, cfg, tokens)
    label = label or f"qwen_logits(model {name})"
    check.shape(got, (len(tokens), VOCAB), label=label)
    check.allclose(got, fixture[f"logits_{name}_{which}"], **TOL, label=label, dtype=True)
    return got


# ---------------------------------------------------------------------------------
# Two small random models: no download needed.
# ---------------------------------------------------------------------------------


# Rejects no_output_gate, rope_whole_head, final_norm_no_one_plus, qk_norm_no_one_plus,
# gate_query_swapped, gate_split_across_projection, and embedding_scaled: each changes
# the logits of every model with an attention layer.
@case("the worked example: four layers (linear, linear, linear, full), tokens [3, 1, 4, 1, 5, 9]",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    got = run_tiny(ctx, "a", "0")
    check.allclose(got[5, :5], WORKED, **TOL, label="qwen_logits(model a)[5, :5]")
    check.equal(got.argmax(-1), [49, 28, 60, 48, 22, 48], label="argmax tokens of model a")


# Rejects hardcoded_rope (this model's base and rotating share are not Qwen3.5 0.8B's)
# and, by a wide margin, norm_eps_default and qk_norm_eps_default: with an epsilon of
# 0.01 a norm that uses 1e-5 or 1e-6 is off by a percent or more.
@case("another configuration: six layers by layer_types, RoPE base 1e4 on half a head, epsilon 0.01",
      public=True, assets=[])  # fmt: skip
def other_configuration(ctx):
    run_tiny(ctx, "b", "0")


# Rejects no_causal_mask without a reference: only the last position survives it.
@case("causality: changing tokens from position t on leaves the logits before t unchanged",
      public=True, assets=[])  # fmt: skip
def causality(ctx):
    weights, cfg, fixture = tiny(ctx, "b")
    tokens = fixture["tokens_0"].copy()
    before = ctx.call(FN, weights, cfg, tokens)
    check.shape(before, (6, VOCAB), label=FN)
    for t in (3,):
        changed = tokens.copy()
        changed[t:] = (tokens[t:] + 1 + np.arange(6 - t)) % VOCAB
        after = ctx.call(FN, weights, cfg, changed)
        check.allclose(after[:t], before[:t], rtol=1e-4, atol=1e-4,
                       label=f"logits before position {t} after changing tokens {t} to 5")  # fmt: skip
        moved = np.abs(after[t:] - before[t:]).max(axis=-1)
        check.that(
            bool(np.all(moved > 1e-3)),
            "the logits at the changed positions did not change",
            f"largest change per position: {moved}",
        )


def fresh_model(ctx, cfg, label, extra=False):
    """Weights drawn now, tokens drawn now, expected logits computed now by the kit."""
    params = hybrid_params(cfg, int(ctx.rng.integers(1, 2**31)))
    weights = checkpoint_layout(params)
    if extra:  # tensors of the kinds the real file carries and a text decoder skips
        weights["model.visual.merger.mlp.0.weight"] = ctx.randn(4, 32)
        weights["mtp.layers.0.eh_proj.weight"] = ctx.randn(32, 64)
    tokens = ctx.randint(0, VOCAB, 6)
    for ids in (tokens, tokens[:1]):  # six tokens, then a sequence of one
        want, _ = hybrid_forward_ref(cfg, params, ids)
        got = ctx.call(FN, weights, cfg, ids)
        what = f"{label}, {len(ids)} token{'s' if len(ids) > 1 else ''}"
        check.shape(got, (len(ids), VOCAB), label=what)
        check.allclose(got, want, **TOL, label=what, dtype=True)


# Rejects a solution that replays the fixture files instead of computing: these weights
# exist only while the case runs.
@case("weights and tokens drawn while the test runs (6 tokens, then 1), epsilon 1e-3",
      public=True, assets=[])  # fmt: skip
def fresh_weights(ctx):
    cfg = HybridConfig(
        n_layers=3, n_key_heads=4, rope_base=1e6, norm_eps=1e-3,
        layer_types=(LINEAR, FULL, LINEAR),
    )  # fmt: skip
    fresh_model(ctx, cfg, "qwen_logits(fresh model: linear, full, linear)")


# The real file has 168 such tensors; this is the public witness without the download.
@case("tensors a text decoder never reads (a vision tower, a draft head) are ignored",
      public=True, assets=[])  # fmt: skip
def extra_tensors(ctx):
    weights, _, _ = tiny(ctx, "a")
    extra = dict(weights)
    extra["model.visual.patch_embed.proj.weight"] = ctx.randn(8, 32)
    extra["mtp.layers.0.post_norm.weight"] = ctx.randn(32)
    run_tiny(ctx, "a", "0", weights=extra, label="qwen_logits(model a plus unused tensors)")


@case("repeated tokens, the largest token id, and token 0, in both configurations", assets=[])
def repeated_tokens(ctx):
    run_tiny(ctx, "a", "1", label=FN)
    run_tiny(ctx, "b", "1", label=FN)


@case("another model drawn while the test runs: a random schedule, base, and epsilon", assets=[])
def fresh_weights_hidden(ctx):
    kinds = [FULL, LINEAR, LINEAR, FULL, LINEAR]
    ctx.rng.shuffle(kinds)
    cfg = HybridConfig(
        n_layers=5, n_key_heads=4, rope_base=float(ctx.rng.choice([1e3, 1e5])),
        partial_rotary_factor=float(ctx.rng.choice([0.25, 0.5])),
        norm_eps=float(ctx.rng.choice([1e-4, 3e-3])), layer_types=tuple(kinds),
    )  # fmt: skip
    fresh_model(ctx, cfg, FN, extra=True)


# ---------------------------------------------------------------------------------
# Qwen3.5 0.8B: 24 layers, 248,320 logits per position.
# ---------------------------------------------------------------------------------


def run_real(ctx, index, *, hidden):
    """Logits of prompt `index` from the solution, checked against the fixture."""
    fixture = ctx.fixture("qwen3.5-0.8b.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    model = RealHybrid(ctx.asset(ASSET))
    key = f"weights_{ctx.framework}"
    if key not in _cache:
        # Bound to the framework once: each tensor is converted the first time it is used.
        _cache[key] = model.weights(ctx.fw)
    ids = fixture[f"ids_{index}"]
    out = ctx.solution.qwen_logits(_cache[key], model.config, ctx.fw.tensor(ids))
    check.that(
        ctx.fw.is_tensor(out),
        f"{FN} returned a {type(out).__name__}, not a tensor of the framework",
    )
    got = np.asarray(ctx.fw.to_numpy(out))
    what = "the prompt" if hidden else repr(ctx.fixture("prompts.json")[index])
    check.shape(got, (12, 248320), label=f"logits for {what}")
    check.that(got.dtype == np.float32, f"logits for {what}: got dtype {got.dtype}, not float32")
    check.allclose(got[:, fixture["columns"]], fixture[f"logits_{index}"], **TOL,
                   label=f"logits for {what}, at 64 vocabulary columns")  # fmt: skip
    clear = fixture[f"margin_{index}"] >= MARGIN
    check.equal(got.argmax(-1)[clear], fixture[f"argmax_{index}"][clear],
                label=f"argmax tokens for {what}")  # fmt: skip
    top = fixture[f"top_ids_{index}"]
    check.allclose(got[-1, top], fixture[f"top_logits_{index}"], **TOL,
                   label=f"the 8 largest logits of the last position for {what}")  # fmt: skip
    return got


# Any real case rejects layers_in_name_order (only a model with more than ten layers has
# names that sort differently from their numbers) and norm_eps_default (the real hidden
# states are small enough for the epsilon to count).
@case("Qwen3.5 0.8B: 'The capital of France is Paris. The capital of Japan is' -> ' Tokyo'",
      public=True)  # fmt: skip
def real_capital(ctx):
    got = run_real(ctx, 0, hidden=False)
    check.equal(
        int(got[-1].argmax()), 25358, label="the most likely next token (' Tokyo' is 25358)"
    )


@case("Qwen3.5 0.8B: a chat prompt with special tokens")
def real_chat(ctx):
    run_real(ctx, 1, hidden=True)
