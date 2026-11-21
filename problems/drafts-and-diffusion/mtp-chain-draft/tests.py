"""Tests for mtp-chain-draft.

Expected drafts come from two places: `MtpModel` of `transformers`, run on a Llama
decoder that holds the kit's weights (`fixtures/shipped.npz`, written by
`fixtures/make.py`), and the kit's numpy oracle `mtp_draft_ref`, which was checked
against the same module. The solution's own framework reference is never the oracle.

A drafted token feeds the next stage, so tokens can only be compared exactly when the
two largest logits of every stage are clearly apart: `expect` asserts a gap of 0.05 for
every input it uses (an author's check, not the solution's).

Prompts have 8 tokens, the largest the statement allows, and 1 in one case (JAX
compiles once per shape).
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import (
    DecoderConfig,
    decoder_forward_ref,
    decoder_hidden_ref,
    mtp_draft_ref,
    mtp_params,
)

FN = "mtp_chain_draft"
TOL = {"rtol": 1e-3, "atol": 1e-4}  # the tolerance stated in Constraints
CFG = DecoderConfig(n_layers=2)
MIN_GAP = 0.05
WORKED_BEST = np.array([5.890466, 5.082431, 4.953575], dtype=np.float32)


def prompt(cfg, params, seed, length):
    """(tokens, next_token, hidden) as the main model would hand them over."""
    tokens = np.random.default_rng([seed, length]).integers(0, cfg.vocab_size, length)
    next_token = int(decoder_forward_ref(cfg, params, tokens)[-1].argmax())
    return tokens, next_token, decoder_hidden_ref(cfg, params, tokens)


def compare(ctx, params, cfg, tokens, next_token, hidden, n_draft, want_tokens, want_logits):
    top2 = np.sort(want_logits, axis=-1)[:, -2:]
    assert float((top2[:, 1] - top2[:, 0]).min()) >= MIN_GAP, "test input: argmax margin too small"
    got = ctx.call(FN, params, cfg, tokens, next_token, hidden, n_draft)
    check.that(
        isinstance(got, tuple) and len(got) == 2,
        f"{FN}: the result is not a pair (draft_tokens, draft_logits)",
    )
    draft_tokens, draft_logits = got
    check.shape(draft_logits, want_logits.shape, label="draft_logits")
    check.allclose(draft_logits, want_logits, **TOL, label="draft_logits", dtype=True)
    check.that(
        draft_tokens.dtype.kind in "iu",
        f"draft_tokens: got dtype {draft_tokens.dtype}, which is not an integer dtype",
    )
    check.equal(draft_tokens, want_tokens, label="draft_tokens")
    return draft_tokens, draft_logits


def expect(ctx, cfg, params, tokens, next_token, hidden, n_draft):
    """Compare with the kit's numpy oracle."""
    want = mtp_draft_ref(cfg, params, tokens, next_token, hidden, n_draft)
    return compare(ctx, params, cfg, tokens, next_token, hidden, n_draft, *want)


def shipped(ctx, run):
    """Compare with what `MtpModel` drafted for run `run` of the fixture."""
    f = ctx.fixture("shipped.npz")
    stages = int(f[f"r{run}_stages"])
    params = mtp_params(CFG, stages, int(f[f"r{run}_seed"]))
    return compare(
        ctx,
        params,
        CFG,
        f[f"r{run}_tokens"],
        int(f[f"r{run}_next_token"]),
        f[f"r{run}_hidden"],
        stages,
        f[f"r{run}_draft_tokens"],
        f[f"r{run}_draft_logits"],
    )


@case("the worked example: 8 tokens, 3 stages, against MtpModel", public=True)
def worked_example(ctx):
    f = ctx.fixture("shipped.npz")
    check.equal(f["r0_tokens"], [9, 10, 50, 1, 20, 2, 12, 9], label="fixture tokens")
    check.equal(int(f["r0_next_token"]), 49, label="fixture next_token")
    draft_tokens, draft_logits = shipped(ctx, 0)
    check.equal(draft_tokens, [41, 25, 7], label="draft_tokens")
    best = draft_logits[np.arange(3), [41, 25, 7]]
    check.allclose(best, WORKED_BEST, atol=2e-3, label="the largest logit of each stage")


@case("one stage, two stages, three stages against MtpModel", public=True)
def shipped_stage_counts(ctx):
    for run in (1, 2):
        shipped(ctx, run)
    f = ctx.fixture("shipped.npz")
    params = mtp_params(CFG, 3, int(f["r2_seed"]))
    args = (f["r2_tokens"], int(f["r2_next_token"]), f["r2_hidden"])
    compare(ctx, params, CFG, *args, 1, f["r2_draft_tokens"][:1], f["r2_draft_logits"][:1])


@case("a prompt of one token", public=True)
def one_token(ctx):
    shipped(ctx, 3)


@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    params = mtp_params(CFG, 2, seed=11)
    tokens, next_token, hidden = prompt(CFG, params, 11, 8)
    held = {"tokens": tokens, "hidden": hidden, **params}
    tensors = {name: ctx.fw.tensor(a) for name, a in held.items()}
    weights = {name: tensors[name] for name in params}
    ctx.solution.mtp_chain_draft(weights, CFG, tensors["tokens"], next_token, tensors["hidden"], 2)
    for name, before in held.items():
        check.allclose(
            ctx.fw.numpy(tensors[name]), before, rtol=0, atol=0, label=f"{name} after the call"
        )


@case("other seeds and prompts, three stages")
def other_seeds(ctx):
    for seed, length in ((0, 8), (6, 8), (13, 8), (10, 8)):
        params = mtp_params(CFG, 3, seed)
        expect(ctx, CFG, params, *prompt(CFG, params, seed, length), 3)


# Rejects no_causal_mask and reads_main_hidden where they are cheapest to miss: the last
# row of stage 0 attends to everything either way, so one stage shows neither.
@case("fewer stages than the checkpoint has")
def fewer_stages(ctx):
    params = mtp_params(CFG, 3, seed=0)
    args = prompt(CFG, params, 0, 8)
    for n_draft in (1, 2):
        expect(ctx, CFG, params, *args, n_draft)


# Rejects head_is_embedding: an untied checkpoint has its own lm_head.weight.
@case("a checkpoint whose head is not tied to the embedding")
def untied_head(ctx):
    cfg = DecoderConfig(n_layers=2, tie_embeddings=False)
    for seed in (3, 4):
        params = mtp_params(cfg, 2, seed)
        expect(ctx, cfg, params, *prompt(cfg, params, seed, 8), 2)


# Rejects default_norm_eps; a block with other head counts and another rotary base.
@case("another configuration: norm_eps, rope_base, and head counts from cfg")
def other_config(ctx):
    cfg = DecoderConfig(n_layers=1, n_heads=2, n_kv_heads=1, norm_eps=0.02, rope_base=50.0)
    for seed in (5, 8):
        params = mtp_params(cfg, 3, seed)
        expect(ctx, cfg, params, *prompt(cfg, params, seed, 8), 3)


# The token after the prompt need not be the main model's argmax (it may be sampled), and
# a prompt may repeat tokens: the module is given `next_token`, it does not recompute it.
@case("a next token that is not the main model's argmax; a prompt of one repeated token")
def given_next_token(ctx):
    params = mtp_params(CFG, 3, seed=5)
    tokens, next_token, hidden = prompt(CFG, params, 5, 8)
    expect(ctx, CFG, params, tokens, (next_token + 16) % CFG.vocab_size, hidden, 3)
    tokens = np.full(8, 9)
    hidden = decoder_hidden_ref(CFG, params, tokens)
    expect(ctx, CFG, params, tokens, 9, hidden, 3)
