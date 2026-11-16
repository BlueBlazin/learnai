"""Tests for decoder-forward.

Numeric cases compare the solution with the kit's numpy oracle, `decoder_forward_ref`,
and with the reference. The property case states causality without either: changing
the tokens from position t on must leave the logits before t unchanged.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import DecoderConfig, decoder_forward_ref, decoder_params

FN = "forward"
TOL = {"rtol": 1e-3, "atol": 1e-4}  # the tolerance stated in Constraints (a deep stack)


def run(ctx, cfg, seed, tokens):
    """Check the solution against the reference and against the kit's oracle."""
    params = decoder_params(cfg, seed)
    got = ctx.check(FN, params, cfg, tokens, **TOL)
    check.allclose(got, decoder_forward_ref(cfg, params, tokens), **TOL, label=FN)
    return got


@case("the worked example: one layer, width 4, two query heads on one key/value head", public=True)
def worked_example(ctx):
    cfg = DecoderConfig(vocab_size=5, d_model=4, n_layers=1, n_heads=2, n_kv_heads=1, d_mlp=6)
    got = run(ctx, cfg, 0, np.array([[1, 3, 0]]))
    expected = np.array(
        [[[0.142855, 1.460640, -0.848208, -0.733341, 0.875444],
          [-0.519588, 0.535796, 2.398094, 6.208618, 1.158294],
          [0.527882, -0.490735, -2.421456, -5.390296, -1.365599]]],
        dtype=np.float32,
    )  # fmt: skip
    check.allclose(got, expected, **TOL, label=FN)


@case("the default model: 2 layers, width 32, 4 query heads on 2 key/value heads", public=True)
def default_model(ctx):
    cfg = DecoderConfig()
    run(ctx, cfg, 1, ctx.randint(0, cfg.vocab_size, (2, 6)))


# Rejects no_causal_mask without the oracle.
@case("causality: changing tokens from position t on leaves the logits before t unchanged",
      public=True)  # fmt: skip
def causality(ctx):
    cfg = DecoderConfig()
    params = decoder_params(cfg, 2)

    def prop(rng):
        tokens = ctx.randint(0, cfg.vocab_size, (2, 6))
        t = int(rng.integers(1, 6))
        changed = tokens.copy()
        # Every token from t on is replaced by a different one.
        changed[:, t:] = (tokens[:, t:] + rng.integers(1, cfg.vocab_size, (2, 6 - t))) % 64
        before = ctx.call(FN, params, cfg, tokens)
        after = ctx.call(FN, params, cfg, changed)
        check.shape(before, (2, 6, cfg.vocab_size), label=FN)
        check.allclose(
            after[:, :t], before[:, :t], rtol=1e-4, atol=1e-4,
            label=f"logits before position {t} after changing the tokens from position {t} on",
        )  # fmt: skip
        moved = np.abs(after[:, t:] - before[:, t:]).max(axis=-1)
        check.that(
            bool(np.all(moved > 1e-3)),
            f"the logits from position {t} on did not change when the tokens there changed",
            f"largest change per row and position: {moved}",
        )

    check.forall(prop, trials=3)


# Rejects always_tied_head; head_defaults_to_zeros passes here and fails everywhere else.
@case("an untied model: the head is lm_head.weight, not the embedding matrix", public=True)
def untied_head(ctx):
    cfg = DecoderConfig(tie_embeddings=False)
    run(ctx, cfg, 3, ctx.randint(0, cfg.vocab_size, (2, 6)))


# The two ends of grouped-query attention.
@case("key/value head counts: n_kv_heads = n_heads = 4, and a single key/value head")
def kv_head_counts(ctx):
    for n_kv_heads in (4, 1):
        cfg = DecoderConfig(n_kv_heads=n_kv_heads)
        run(ctx, cfg, 4, ctx.randint(0, cfg.vocab_size, (2, 6)))


# Rejects hardcoded_hyperparameters: an eps large enough to matter and a short RoPE base.
@case("hyperparameters from the config: norm_eps = 0.05 and rope_base = 50")
def config_hyperparameters(ctx):
    cfg = DecoderConfig(norm_eps=0.05, rope_base=50.0)
    run(ctx, cfg, 5, ctx.randint(0, cfg.vocab_size, (2, 6)))


@case("a deeper model: 4 layers")
def deeper_model(ctx):
    cfg = DecoderConfig(n_layers=4)
    run(ctx, cfg, 6, ctx.randint(0, cfg.vocab_size, (2, 6)))


# The same token at every position: the rows of the output differ only through what
# each position can see. Also a batch whose two rows are the same sequence.
@case("repeated tokens, and identical rows in a batch")
def repeated_tokens(ctx):
    cfg = DecoderConfig()
    tokens = np.array([[7, 7, 7, 7, 7, 7], [9, 4, 9, 4, 9, 4]])
    run(ctx, cfg, 8, tokens)
    got = run(ctx, cfg, 8, np.array([[5, 1, 5, 1, 30, 2], [5, 1, 5, 1, 30, 2]]))
    check.allclose(got[0], got[1], rtol=1e-5, atol=1e-5, label="two identical rows of a batch")
