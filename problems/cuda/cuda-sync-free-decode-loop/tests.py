"""Tests for cuda-sync-free-decode-loop.

The model is `StepLM`, defined below: a one-layer transformer with a KV cache written
in torch, so that it can live on a CUDA device (the kit's `TinyLM` computes in NumPy
and would read its inputs on the host). It never reads a value on the host itself and
counts its calls.

Each case plans the generation first, on a twin of the model: it chooses the tokens
(and where the EOS falls), and derives from the model's own probabilities the uniform
numbers that make inverse-CDF sampling produce exactly those tokens, each uniform in
the middle of its token's interval. A solution whose float32 arithmetic differs in the
last bits therefore still samples the same tokens, on any CPU and on a GPU.

Then the solution runs once under `SyncSpy` and `DeviceSpy`, and the case checks

1. the tokens, exactly;
2. host synchronizations made by the solution file: at most ceil(n / check_every) + 1;
3. that every factory call passed `device=`;
4. the work asked of the model: no call after the check that could have seen the EOS.

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import math

import numpy as np
import torch
from torch import nn

from whetstone.judge import case, check
from whetstone.kit import DeviceSpy, SyncSpy
from whetstone.kit.counters import Counters

FN = "decode"
CPU = torch.device("cpu")
EOS = 1
#: The uniforms that sample the first three tokens of the worked example, as printed
#: in the statement.
EXAMPLE_UNIFORMS = [0.3735, 0.8579, 0.0272]


class StepLM(nn.Module):
    """A small seeded causal language model with a KV cache, in torch.

        logits, cache = lm(tokens)           # tokens: 1-D int64 tensor on the model's device
        logits, cache = lm(tokens, cache)    # tokens continue the cached prefix

    `logits` is (T, vocab_size) float32, one row per token of this call; `cache` is a
    pair of tensors and is never modified. `lm.counters` has `forward_calls` and
    `tokens_processed`. Nothing here reads a tensor's value on the host.
    """

    def __init__(self, seed, vocab_size=32, dim=16, max_context=160):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        def weight(*shape, scale):
            return torch.randn(*shape, generator=gen) * scale

        self.vocab_size, self.dim, self.max_context = vocab_size, dim, max_context
        self.register_buffer("emb", weight(vocab_size, dim, scale=1.0))
        self.register_buffer("pos", weight(max_context, dim, scale=0.5))
        for name in ("w_q", "w_k", "w_v"):
            self.register_buffer(name, weight(dim, dim, scale=dim**-0.5))
        self.register_buffer("w_in", weight(dim, 2 * dim, scale=dim**-0.5))
        self.register_buffer("w_out", weight(2 * dim, dim, scale=(2 * dim) ** -0.5))
        self.register_buffer("head", weight(vocab_size, dim, scale=0.1))
        self.counters = Counters("forward_calls", "tokens_processed")

    def forward(self, tokens, cache=None):
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.ndim != 1:
            raise TypeError("the model takes a 1-D int64 tensor of token ids")
        if tokens.device != self.emb.device:
            raise RuntimeError(
                f"the model is on {self.emb.device} and received tokens on {tokens.device}"
            )
        steps = tokens.shape[0]
        start = 0 if cache is None else cache[0].shape[0]
        if steps < 1 or start + steps > self.max_context:
            raise ValueError("a call needs at least one token and must fit the context")
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", steps)

        x = self.emb[tokens] + self.pos[start : start + steps]
        q, k, v = x @ self.w_q, x @ self.w_k, x @ self.w_v
        if cache is not None:
            k, v = torch.cat([cache[0], k]), torch.cat([cache[1], v])
        rows = torch.arange(steps, device=x.device)
        cols = torch.arange(start + steps, device=x.device)
        scores = (q @ k.T) / math.sqrt(self.dim)
        scores = scores.masked_fill(cols[None, :] > start + rows[:, None], float("-inf"))
        h = x + torch.softmax(scores, dim=-1) @ v
        h = h + torch.tanh(h @ self.w_in) @ self.w_out
        return h @ self.head.T, (k, v)


def plan(seed, prompt, n, check_every, rng, device=CPU, forced=(), eos_at=None):
    """Choose the tokens to generate and the uniforms that sample them.

    `forced` fixes the first tokens; `eos_at` (1-based) puts the EOS at that step.
    Returns (expected tokens, uniforms as float32, the step after which a correct
    loop stops).
    """
    lm = StepLM(seed).to(device)
    tokens, uniforms = [], []
    with torch.no_grad():
        logits, cache = lm(torch.tensor(prompt, device=device))
        for step in range(1, n + 1):
            probs = torch.softmax(logits[-1].double(), dim=-1).cpu().numpy()
            cdf = np.cumsum(probs)
            if step <= len(forced):
                token = forced[step - 1]
            elif step == eos_at:
                token = EOS
            else:
                likely = [t for t in range(lm.vocab_size) if t != EOS and probs[t] >= 0.01]
                token = int(rng.choice(likely))
            if probs[token] < 5e-4:
                raise RuntimeError("authoring error: the planned token is too unlikely")
            low = cdf[token - 1] if token > 0 else 0.0
            uniforms.append((low + cdf[token]) / 2 / cdf[-1])  # the middle of its interval
            tokens.append(token)
            if token == EOS:
                break
            if step < n:
                logits, cache = lm(torch.tensor([token], device=device), cache)
    stop = n
    if tokens[-1] == EOS:
        # The first check at or after the EOS, or the end of the budget.
        stop = min(n, math.ceil(len(tokens) / check_every) * check_every)
    # Whatever is sampled after the EOS must not show: give those steps real uniforms.
    uniforms += [float(u) for u in rng.uniform(0.0, 1.0, size=n - len(uniforms))]
    expected = tokens + [EOS] * (n - len(tokens))
    return expected, np.asarray(uniforms, dtype=np.float32), stop


def run(ctx, seed, prompt, n, check_every, device=CPU, **options):
    """Plan, call the solution once, and check tokens, synchronizations, and cost."""
    expected, uniforms, stop = plan(seed, prompt, n, check_every, ctx.rng, device, **options)
    lm = StepLM(seed).to(device)
    with SyncSpy() as syncs, DeviceSpy() as factories:
        out = ctx.solution.decode(
            lm, torch.tensor(prompt), n, torch.from_numpy(uniforms), device, EOS, check_every
        )

    # 1. The tokens.
    check.that(
        isinstance(out, torch.Tensor), f"{FN}: returned a {type(out).__name__}, not a tensor"
    )
    check.that(
        out.dtype == torch.int64, f"{FN}: got dtype {out.dtype}, which is not the expected dtype"
    )
    check.that(
        out.device.type == "cpu", f"{FN}: the result is on {out.device}; it must be on the CPU"
    )
    check.equal(out.numpy(), np.asarray(expected), label="tokens")

    # 2. Host synchronizations.
    limit = math.ceil(n / check_every) + 1
    ctx.metric("host_syncs", syncs.syncs, None, limit)
    check.that(
        syncs.syncs <= limit,
        f"host_syncs: {syncs.syncs} exceeds the limit of {limit}",
        "Host synchronizations made by your file, in order:\n" + syncs.describe(),
    )

    # 3. Where tensors were created.
    check.that(
        not factories.implicit,
        f"{FN}: {len(factories.implicit)} tensors were created without device= "
        "(they land on the default device, the CPU)",
        factories.describe(),
    )

    # 4. The work asked of the model.
    check.cost(lm.counters, "forward_calls", at_most=stop)
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + stop - 1, unit="tokens")
    return out, expected, uniforms


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: EOS at token 3 of 8, flag checked every 4 tokens", public=True)
def worked_example(ctx):
    # Rejects no_mask_after_eos (token 4 is sampled before the check) and never_checks.
    _, expected, uniforms = run(ctx, 3, [0, 7, 19, 4], 8, 4, forced=(9, 27, EOS))
    if expected != [9, 27, 1, 1, 1, 1, 1, 1]:
        raise RuntimeError("authoring error: the worked example changed")
    if not np.allclose(uniforms[:3], EXAMPLE_UNIFORMS, atol=2e-3):
        raise RuntimeError(f"authoring error: the example's uniforms are {uniforms[:3]}")


@case("no EOS: 12 tokens, at most 4 host synchronizations", public=True)
def no_eos(ctx):
    # Rejects item_per_token, python_eos_check, host_token_list, nonzero_sampling,
    # and buffer_default_device.
    run(ctx, 3, [0, 7, 19, 4], 12, 4)


@case("EOS exactly at a check: the loop stops there, with no further model call", public=True)
def eos_at_check(ctx):
    # Rejects check_off_by_one: the EOS is token 4 and the flag is read after token 4.
    run(ctx, 3, [0, 12, 30], 12, 4, eos_at=4)
    run(ctx, 5, [0, 2], 16, 8, eos_at=8)


@case("EOS between two checks: the tokens after it are EOS, and the result has n tokens")
def eos_between_checks(ctx):
    # Rejects returns_trimmed.
    run(ctx, 5, [0, 21, 6, 6], 16, 4, eos_at=6)
    run(ctx, 4, [0, 9], 10, 5, eos_at=1)
    run(ctx, 4, [0, 9, 3], 24, 8, eos_at=17)


@case("64 tokens, flag checked every 8: at most 9 host synchronizations")
def long_generation(ctx):
    run(ctx, 6, [0, 3, 14, 15, 9, 26], 64, 8)
    run(ctx, 6, [0, 5], 64, 8, eos_at=41)


@case("EOS in the last, incomplete block of checks, and EOS as the very last token")
def eos_near_the_end(ctx):
    run(ctx, 7, [0, 18, 2], 13, 4, eos_at=11)  # stops after token 12
    run(ctx, 7, [0, 18, 2], 13, 4, eos_at=13)
    run(ctx, 7, [0, 18, 2], 12, 4, eos_at=12)
    run(ctx, 7, [0, 18, 2], 14, 4, eos_at=13)  # no check left: runs to token 14


@case("check_every = 1, check_every larger than n, n = 1, and a one-token prompt")
def schedules(ctx):
    run(ctx, 8, [0, 4, 4], 9, 1, eos_at=5)
    run(ctx, 8, [0, 4, 4], 6, 1)
    run(ctx, 8, [0, 4, 4], 7, 16)
    run(ctx, 8, [0, 4, 4], 7, 16, eos_at=2)
    run(ctx, 9, [0], 1, 4)
    run(ctx, 9, [0], 1, 4, eos_at=1)
    run(ctx, 9, [0], 20, 3)


@case("random prompts, lengths, schedules, and EOS positions")
def sweep(ctx):
    for seed in (11, 12, 13, 14, 15, 16):
        n = int(ctx.rng.integers(2, 40))
        check_every = int(ctx.rng.integers(1, 12))
        prompt = [0, *(int(t) for t in ctx.rng.integers(2, 32, size=int(ctx.rng.integers(0, 6))))]
        eos_at = int(ctx.rng.integers(1, n + 1)) if seed % 2 else None
        run(ctx, seed, prompt, n, check_every, eos_at=eos_at)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case(
    "on the GPU: 48 tokens with the model on the CUDA device, 7 synchronizations", requires="cuda"
)
def gpu_no_eos(ctx):
    # The model raises if it is handed tokens that are not on its device.
    run(ctx, 3, [0, 7, 19, 4], 48, 8, device=torch.device("cuda"))


@case("on the GPU: EOS between two checks and exactly at a check", requires="cuda")
def gpu_eos(ctx):
    device = torch.device("cuda")
    run(ctx, 5, [0, 21, 6, 6], 32, 8, device=device, eos_at=11)
    run(ctx, 5, [0, 2], 32, 8, device=device, eos_at=16)
