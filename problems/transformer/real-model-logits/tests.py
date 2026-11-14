"""Tests for real-model-logits.

Every expected logit comes from `transformers` (float32, CPU), stored by
`fixtures/make.py`; the reference solution is never the oracle.

- The cases marked `assets=[]` run two three-layer Granite models with random weights
  and a 64-token vocabulary: `a` with Granite 4.0's multipliers and a tied head, `b`
  with other multipliers, RoPE base 10,000, and an untied head. Their weights are
  drawn here from the same seeds as in `make.py`; `tiny.npz` stores tokens and logits.
  All of these cases use tokens of shape (2, 6), and both models have the same tensor
  shapes, so JAX compiles once.
- The other cases run the real granite-4.0-350m on four prompts of exactly 12 tokens
  (one more set of shapes). The 352 M weights are decoded from bfloat16 and handed to
  the framework once per run, in the first of these cases.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import RealConfig, RealModel

FN = "granite_logits"
ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
TOL = {"rtol": 1e-3, "atol": 1e-3}  # the tolerance stated in Constraints
MARGIN = 0.2  # argmax is compared where the two largest logits are further apart

TINY = {"vocab": 64, "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "d_ff": 96, "n_layers": 3}
# logits[0, 5, :5] of the worked example, as printed in the statement
WORKED = [-0.095271, 0.260368, 4.941798, 4.609704, -0.712952]
_cache: dict[str, object] = {}


def tiny_weights(seed: int, tied: bool) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout. Keep in step with make.py."""
    rng = np.random.default_rng(seed)
    d, f, vocab = TINY["d_model"], TINY["d_ff"], TINY["vocab"]
    q_width = d
    kv_width = d * TINY["n_kv_heads"] // TINY["n_heads"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, d).astype(np.float32)

    weights = {"model.embed_tokens.weight": normal(vocab, d, 1.0)}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = gain()
        weights[p + "self_attn.q_proj.weight"] = normal(q_width, d, 0.3)
        weights[p + "self_attn.k_proj.weight"] = normal(kv_width, d, 0.3)
        weights[p + "self_attn.v_proj.weight"] = normal(kv_width, d, 0.1)
        weights[p + "self_attn.o_proj.weight"] = normal(d, q_width, 0.1)
        weights[p + "post_attention_layernorm.weight"] = gain()
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.1)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.1)
    weights["model.norm.weight"] = gain()
    if not tied:
        weights["lm_head.weight"] = normal(vocab, d, 1.0)
    return weights


def tiny(ctx, name):
    """(weights, cfg, fixture) of the random model `a` or `b`."""
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
        _cache[key] = (weights, cfg, fixture)
    return _cache[key]


def run_tiny(ctx, name, which, weights=None, label=None):
    own, cfg, fixture = tiny(ctx, name)
    tokens = fixture[f"tokens_{which}"]
    got = ctx.call(FN, own if weights is None else weights, cfg, tokens)
    label = label or f"granite_logits(model {name})"
    check.shape(got, (2, 6, TINY["vocab"]), label=label)
    check.allclose(got, fixture[f"logits_{name}_{which}"], **TOL, label=label, dtype=True)
    return got


# ---------------------------------------------------------------------------------
# Two three-layer random models: no download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: a three-layer model with Granite's multipliers and a tied head",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    got = run_tiny(ctx, "a", 0)
    check.allclose(got[0, 5, :5], WORKED, **TOL, label="granite_logits(model a)[0, 5, :5]")


# Rejects hardcoded_multipliers and always_tied_head.
@case("another configuration: other multipliers, RoPE base 10,000, an untied head",
      public=True, assets=[])  # fmt: skip
def other_configuration(ctx):
    run_tiny(ctx, "b", 0)


# Rejects no_causal_mask without a reference.
@case("causality: changing tokens from position t on leaves the logits before t unchanged",
      public=True, assets=[])  # fmt: skip
def causality(ctx):
    weights, cfg, fixture = tiny(ctx, "a")

    def prop(rng):
        tokens = fixture["tokens_0"].copy()
        t = int(rng.integers(1, 6))
        changed = tokens.copy()
        changed[:, t:] = (tokens[:, t:] + rng.integers(1, 64, (2, 6 - t))) % 64
        before = ctx.call(FN, weights, cfg, tokens)
        after = ctx.call(FN, weights, cfg, changed)
        check.shape(before, (2, 6, 64), label=FN)
        check.allclose(after[:, :t], before[:, :t], rtol=1e-4, atol=1e-4,
                       label="logits before the first changed position")  # fmt: skip
        moved = np.abs(after[:, t:] - before[:, t:]).max(axis=-1)
        check.that(
            bool(np.all(moved > 1e-3)),
            "the logits at the changed positions did not change",
            f"largest change per row and position: {moved}",
        )

    check.forall(prop, trials=2)


@case("repeated tokens and the largest token id, in both configurations", assets=[])
def repeated_tokens(ctx):
    run_tiny(ctx, "a", 1)
    run_tiny(ctx, "b", 1)


# Rejects head_from_dict: the config says tied, so an lm_head.weight in the mapping is
# not the head.
@case("a tied model whose mapping also holds a stale lm_head.weight", assets=[])
def stale_head(ctx):
    weights, _, _ = tiny(ctx, "a")
    extra = dict(weights)
    extra["lm_head.weight"] = ctx.randn(64, 128)
    run_tiny(ctx, "a", 0, weights=extra, label="granite_logits(tied model with an extra lm_head)")


# ---------------------------------------------------------------------------------
# granite-4.0-350m: 28 layers, 100,352 logits per position.
# ---------------------------------------------------------------------------------


def run_real(ctx, index, *, hidden):
    """Logits of prompt `index` from the solution, checked against the fixture."""
    fixture = ctx.fixture("granite-4.0-350m.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    model = RealModel(ctx.asset(ASSET))
    key = f"weights_{ctx.framework}"
    if key not in _cache:
        # Bound to the framework once: each tensor is converted the first time it is used.
        _cache[key] = model.weights(ctx.fw)
    ids = fixture[f"ids_{index}"]
    out = ctx.solution.granite_logits(_cache[key], model.config, ctx.fw.tensor(ids[None, :]))
    check.that(
        ctx.fw.is_tensor(out),
        f"{FN} returned a {type(out).__name__}, not a tensor of the framework",
    )
    got = np.asarray(ctx.fw.to_numpy(out))
    what = "the prompt" if hidden else repr(ctx.fixture("prompts.json")[index])
    check.shape(got, (1, 12, 100352), label=f"logits for {what}")
    check.that(got.dtype == np.float32, f"logits for {what}: got dtype {got.dtype}, not float32")
    got = got[0]
    check.allclose(got[:, fixture["columns"]], fixture[f"logits_{index}"], **TOL,
                   label=f"logits for {what}, at 64 vocabulary columns")  # fmt: skip
    clear = fixture[f"margin_{index}"] > MARGIN
    check.equal(got.argmax(-1)[clear], fixture[f"argmax_{index}"][clear],
                label=f"argmax tokens for {what}")  # fmt: skip
    top = fixture[f"top_ids_{index}"]
    check.allclose(got[-1, top], fixture[f"top_logits_{index}"], **TOL,
                   label=f"the 8 largest logits of the last position for {what}")  # fmt: skip
    return got


@case("Granite 4.0 350M: 'The capital of France is Paris. The capital of Japan is' -> ' Tokyo'",
      public=True)  # fmt: skip
def real_capital(ctx):
    got = run_real(ctx, 0, hidden=False)
    check.equal(
        int(got[-1].argmax()), 27286, label="the most likely next token (' Tokyo' is 27286)"
    )


@case("Granite 4.0 350M: a line of Python", public=True)
def real_code(ctx):
    run_real(ctx, 1, hidden=False)


# Any real case rejects layers_in_name_order: only a model with more than ten layers has
# names that sort differently from their numbers.
@case("Granite 4.0 350M: a chat prompt with role tokens")
def real_chat(ctx):
    run_real(ctx, 2, hidden=True)


@case("Granite 4.0 350M: emoji and Japanese")
def real_unicode(ctx):
    run_real(ctx, 3, hidden=True)
