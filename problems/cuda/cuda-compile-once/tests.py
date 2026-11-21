"""Tests for cuda-compile-once.

The judge owns the model (a two-layer causal transformer over a static KV cache, both
defined below) and hands the solution `model` and `cache`. A case plays requests
through `make_step` and `generate` and checks, on the CPU with `backend="eager"`
(Dynamo traces and guards exactly as for any backend; no C++ compiler is involved):

- **the tokens**, exactly, against greedy decoding without a cache, without padding,
  and without `torch.compile`: the whole sequence is run through the model again for
  every token. The step's logits are compared with the same oracle to a tolerance;
- **the number of graphs** Dynamo built (`CompileCounter`): one for the decode shape
  and one per distinct padded prompt length. A Python int position, an unpadded
  prompt, or a sequence that grows costs one graph per value;
- **graph breaks**: none;
- **the calls of `torch.compile`**, through a wrapper installed for the case: exactly
  one, in `make_step`, with `fullgraph=True`, `dynamic=False`, and the `backend` and
  `mode` it was given. Compiling again inside the loop builds no new graph (Dynamo's
  cache is keyed by the function's code), so only this wrapper sees it;
- **the calls of `step`**, through a recording wrapper: the padded prompt at position
  0, then one `(batch, 1)` block per new token except the last, at consecutive
  positions, every position a 0-d int64 tensor;
- **no host reads and no tensors on the default device** in `generate` (`SyncSpy`,
  `DeviceSpy`).

Before every request the cache is filled with large finite garbage, so a step that
attends to a slot it should not see, or a decode loop that starts at the wrong slot,
changes the logits by orders of magnitude.

The recording wrapper and the `torch.compile` wrapper are behind the kit's `Guarded`: a
solution may call the `step` it was handed and `torch.compile`, and can read or change
nothing of the objects behind them (the log, the unrecorded step, the record of
`torch.compile` calls).

`CompileCounter.compilations` reads Dynamo's `frames.ok` counter, which torch 2.15 does
not increment under `fullgraph=True`; `CompileCounter.graphs` (graphs handed to the
backend) counts in both settings. The judge uses the larger of the two.

The GPU case (`requires="cuda"`) was written from the PyTorch 2.15 documentation and
has never run on the machine that validates the catalog.
"""

import math
from contextlib import contextmanager

import numpy as np
import torch
from torch._dynamo.exc import FailOnRecompileLimitHit

from learnai.judge import case, check
from learnai.kit import CompileCounter, DeviceSpy, Guarded, SyncSpy, check_guard, guard

CPU = torch.device("cpu")
VOCAB, DIM, HEADS, LAYERS, MAX_LEN = 32, 16, 2, 2, 96
RECOMPILE_HINT = (
    "With dynamic=False every guard failure is a full compilation. Guards are placed on "
    "input shapes and on the values of Python numbers, never on the values of tensors."
)


# --------------------------------------------------------------------------------------
# The judge's model: a causal transformer over a static KV cache
# --------------------------------------------------------------------------------------


class StaticCache:
    """The cache of `cuda-static-kv-cache`: full-length buffers, written in place."""

    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, device, dtype):
        shape = (batch, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.max_len = max_len
        self._slots = torch.arange(max_len, device=device)

    def update(self, layer, k, v, pos):
        index = pos + self._slots[: k.shape[2]]
        self.k[layer].index_copy_(2, index, k)
        self.v[layer].index_copy_(2, index, v)
        return self.k[layer], self.v[layer]

    def mask(self, pos, n_new):
        last_visible = pos + self._slots[:n_new]
        return self._slots[None, :] <= last_visible[:, None]


def rms(x):
    return x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)


class TinyDecoder(torch.nn.Module):
    """model(tokens, pos, cache) -> logits (batch, T, VOCAB).

    Writes the keys and values of the T tokens into slots pos .. pos + T - 1 of the
    cache and attends, for each of them, to slots 0 .. its own. `pos` may be a 0-d
    tensor or a Python int: the arithmetic is the same, which is why an int goes
    unnoticed until the compiler guards on it.
    """

    def __init__(self, seed=7):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        def weight(*shape, scale):
            values = torch.randn(*shape, generator=gen) * scale
            return torch.nn.Parameter(values, requires_grad=False)

        self.embed = weight(VOCAB, DIM, scale=1.0)
        self.position = weight(MAX_LEN, DIM, scale=1.0)
        names = ("wq", "wk", "wv", "wo")
        for name in names:
            layers = [weight(DIM, DIM, scale=DIM**-0.5) for _ in range(LAYERS)]
            setattr(self, name, torch.nn.ParameterList(layers))
        self.head = weight(DIM, VOCAB, scale=1.0)

    def heads(self, x):
        batch, n_tokens, _ = x.shape
        return x.view(batch, n_tokens, HEADS, DIM // HEADS).transpose(1, 2)

    def forward(self, tokens, pos, cache):
        n_tokens = tokens.shape[1]
        x = self.embed[tokens] + self.position[pos + cache._slots[:n_tokens]]
        mask = cache.mask(pos, n_tokens)
        for layer in range(LAYERS):
            h = rms(x)
            q = self.heads(h @ self.wq[layer])
            k, v = cache.update(
                layer, self.heads(h @ self.wk[layer]), self.heads(h @ self.wv[layer]), pos
            )
            scores = q @ k.transpose(-1, -2) / math.sqrt(DIM // HEADS)
            scores = scores.masked_fill(~mask, float("-inf"))
            mixed = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(x.shape)
            x = x + mixed @ self.wo[layer]
        return rms(x) @ self.head


def new_cache(batch, device=CPU, max_len=MAX_LEN):
    return StaticCache(LAYERS, batch, HEADS, max_len, DIM // HEADS, device, torch.float32)


def oracle(model, prompt, n_new):
    """Greedy decoding with no cache reuse, no padding, and no compilation.

    Returns (tokens (batch, n_new), logits (batch, n_new, VOCAB)) on the CPU.
    """
    sequence, tokens, logits = prompt.clone(), [], []
    for _ in range(n_new):
        last = model(sequence, torch.tensor(0), new_cache(prompt.shape[0]))[:, -1]
        top = last.topk(2, dim=-1).values
        if float((top[:, 0] - top[:, 1]).min()) < 1e-3:
            raise RuntimeError("authoring error: the oracle's argmax is nearly tied")
        token = last.argmax(dim=-1, keepdim=True)
        sequence = torch.cat([sequence, token], dim=1)
        tokens.append(token)
        logits.append(last)
    return torch.cat(tokens, dim=1), torch.stack(logits, dim=1)


# --------------------------------------------------------------------------------------
# Spies
# --------------------------------------------------------------------------------------


class CompileSpy(Guarded):
    """Records every call of `torch.compile` made while the block is active."""

    label = "torch.compile"
    allowed = frozenset({"__class__"})

    def __init__(self, passthrough=True):
        self.calls = []
        self.passthrough = passthrough

    def compile(self, model=None, **kwargs):
        self.calls.append(kwargs)
        if self.passthrough:
            return self.real(model, **kwargs)
        return model if model is not None else (lambda fn: fn)

    def __enter__(self):
        self.real = torch.compile
        torch.compile = self.compile
        return self

    def __exit__(self, *exc):
        torch.compile = self.real


def describe_position(pos):
    if isinstance(pos, torch.Tensor):
        if pos.dim() == 0 and pos.dtype == torch.int64:
            return ("0-d int64 tensor", int(pos))
        return (f"{str(pos.dtype).removeprefix('torch.')} tensor of shape {tuple(pos.shape)}", None)
    return (f"Python {type(pos).__name__}", pos)


class Session(Guarded):
    """One `make_step` and the requests played through it, under the spies."""

    label = "step"
    allowed = frozenset({"__class__"})

    def __init__(self, ctx, batch, compiles, counter):
        self.ctx, self.batch, self.compiles, self.counter = ctx, batch, compiles, counter
        self.model = TinyDecoder()
        self.cache = new_cache(batch)
        self.log = []  # one list of (tokens shape, position) per request
        self.requests = []  # (n_prompt, n_new, bucket) per request
        self.spies = []  # (SyncSpy, DeviceSpy) per request
        guard(ctx)
        compiled = ctx.solution.make_step(self.model, self.cache, backend="eager")
        check_guard()
        check.that(callable(compiled), "make_step: must return the compiled step function")
        self.compiled = compiled

    def step(self, tokens, pos):
        """What `generate` is handed: the solution's step, with its calls recorded."""
        self.log[-1].append((tuple(tokens.shape), describe_position(pos)))
        return self.compiled(tokens, pos)

    def garbage(self):
        for buffer in (*self.cache.k, *self.cache.v):
            values = self.ctx.rng.standard_normal(tuple(buffer.shape)).astype(np.float32) * 30.0
            buffer.copy_(torch.from_numpy(values))

    def request(self, n_prompt, n_new, bucket=None):
        """Generate n_new tokens for a random prompt; check the tokens and the spies."""
        ctx = self.ctx
        prompt = torch.from_numpy(ctx.randint(0, VOCAB, (self.batch, n_prompt)))
        given = prompt.clone()
        want, _ = oracle(self.model, prompt, n_new)
        self.garbage()
        self.log.append([])
        self.requests.append((n_prompt, n_new, 16 if bucket is None else bucket))
        extra = () if bucket is None else (bucket,)
        guard(ctx)
        with DeviceSpy() as factories, SyncSpy() as syncs:
            try:
                out = ctx.solution.generate(self.step, prompt, n_new, CPU, *extra)
            except FailOnRecompileLimitHit:
                check.that(
                    False,
                    "generate: the step was recompiled until torch gave up (8 graphs for "
                    "one function is Dynamo's recompile_limit, and fullgraph=True makes "
                    "reaching it an error)",
                    RECOMPILE_HINT,
                )
        check_guard()
        check.that(isinstance(out, torch.Tensor), "generate: must return a tensor")
        check.that(
            out.dtype == torch.int64 and tuple(out.shape) == (self.batch, n_new),
            f"generate: got dtype {str(out.dtype).removeprefix('torch.')} and shape "
            f"{tuple(out.shape)}, which is not what is expected",
            f"expected int64 tokens of shape (batch, n_new) = {(self.batch, n_new)}",
        )
        check.equal(out.numpy(), want.numpy(), label="tokens")
        check.that(bool(torch.equal(prompt, given)), "generate: the prompt was modified in place")
        self.spies.append((syncs, factories))

    def finish(self, limit):
        """After the last request: graphs, graph breaks, compile calls, step calls."""
        counter = self.counter.snapshot()
        check.that(
            counter.graph_breaks == 0,
            f"the step has {counter.graph_breaks} graph breaks: Dynamo split it into "
            "separately compiled pieces around code it could not trace",
            counter.describe(),
        )
        check.at_most(
            "compilations", max(counter.compilations, counter.graphs), limit, unit="graphs"
        )
        check_compile_calls(self.compiles, backend="eager", mode=None)
        for number, (calls, (n_prompt, n_new, bucket)) in enumerate(
            zip(self.log, self.requests, strict=True), start=1
        ):
            check_calls(calls, self.batch, n_prompt, n_new, bucket, number)
        for syncs, factories in self.spies:
            check.that(
                syncs.syncs == 0,
                f"generate: {syncs.syncs} host synchronizations (a tensor's value was read "
                "in Python)",
                syncs.describe(),
            )
            check.that(
                not factories.implicit,
                f"generate: {len(factories.implicit)} tensors were created without device= "
                "(they land on the default device, the CPU)",
                factories.describe(),
            )


def check_compile_calls(compiles, backend, mode):
    calls = compiles.calls
    check.that(
        len(calls) == 1,
        f"torch.compile was called {len(calls)} times; make_step must call it exactly "
        "once, and nothing may call it afterwards",
        "The judge counts calls of `torch.compile` (spelled that way) from make_step to "
        "the end of the last request.",
    )
    kwargs = calls[0]
    for name, value in (
        ("fullgraph", True),
        ("dynamic", False),
        ("backend", backend),
        ("mode", mode),
    ):
        got = kwargs.get(name)
        check.that(
            got is value or (isinstance(value, str) and got == value),
            f"torch.compile was called with {name}={got!r}, which is not what the statement "
            "specifies",
            f"expected {name}={value!r}",
        )


def check_calls(calls, batch, n_prompt, n_new, bucket, number):
    """The step calls of one request: the padded prompt, then n_new - 1 single tokens."""
    padded = -(-n_prompt // bucket) * bucket
    where = f"generate (request {number})"
    detail = (
        f"prompt of {n_prompt} tokens, n_new = {n_new}, bucket = {bucket}; expected one call "
        f"with tokens of shape {(batch, padded)} at position 0, then {n_new - 1} calls with "
        f"shape {(batch, 1)} at positions {n_prompt}, {n_prompt + 1}, ...\n"
        "calls made (tokens shape, position):\n"
        + "".join(f"  {shape}, {kind}{'' if value is None else f' {value}'}\n"
                  for shape, (kind, value) in calls)
    )  # fmt: skip
    for index, (_, (kind, _)) in enumerate(calls, start=1):
        check.that(
            kind == "0-d int64 tensor",
            f"{where}: call {index} of step got a {kind} as the position, not a 0-d int64 tensor",
            detail,
        )
    check.that(
        len(calls) == n_new,
        f"{where}: step was called {len(calls)} times, which is not the number of calls the "
        "statement specifies",
        detail,
    )
    shape, (_, value) = calls[0]
    check.that(
        shape == (batch, padded) and value == 0,
        f"{where}: the first call of step got tokens of shape {shape} at position {value}, "
        "which is not the padded prompt at position 0",
        detail,
    )
    for index, (shape, (_, value)) in enumerate(calls[1:], start=1):
        check.that(
            shape == (batch, 1) and value == n_prompt + index - 1,
            f"{where}: call {index + 1} of step got tokens of shape {shape} at position "
            f"{value}, which is not the next single token at its slot",
            detail,
        )


@contextmanager
def session(ctx, batch=1):
    with CompileSpy() as compiles, CompileCounter() as counter:
        yield Session(ctx, batch, compiles, counter)


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: a 5-token prompt, 4 new tokens, two graphs", public=True)
def worked_example(ctx):
    # Rejects python_int_position (4 graphs), no_padding and growing_sequence (the call
    # log), compiles_every_call, default_dynamic, and branch_without_fullgraph.
    with session(ctx) as s:
        s.request(5, 4)
        s.finish(limit=2)


@case("the compiled step returns the model's logits, for a prompt block and for single tokens",
      public=True)  # fmt: skip
def step_matches_eager(ctx):
    with session(ctx, batch=2) as s:
        prompt = torch.from_numpy(ctx.randint(0, VOCAB, (2, 16)))
        want_tokens, want_logits = oracle(s.model, prompt, 4)
        s.garbage()
        got = s.compiled(prompt, torch.tensor(0))
        check.that(isinstance(got, torch.Tensor), "step: must return the logits tensor")
        check.that(
            got.dtype == torch.float32 and tuple(got.shape) == (2, 16, VOCAB),
            f"step: got dtype {str(got.dtype).removeprefix('torch.')} and shape "
            f"{tuple(got.shape)}, which is not what is expected",
            f"expected float32 logits of shape (batch, T, vocab) = {(2, 16, VOCAB)}",
        )
        check.allclose(got[:, -1].numpy(), want_logits[:, 0].numpy(), atol=1e-4,
                       label="step logits of the last prompt token")  # fmt: skip
        for i in range(3):
            got = s.compiled(want_tokens[:, i : i + 1], torch.tensor(16 + i))
            check.that(
                tuple(got.shape) == (2, 1, VOCAB),
                f"step: got shape {tuple(got.shape)} for a single-token block, which is not "
                "the expected shape",
            )
            check.allclose(got[:, 0].numpy(), want_logits[:, i + 1].numpy(), atol=1e-4,
                           label="step logits of a decode step")  # fmt: skip
        counter = s.counter.snapshot()
        check.at_most("compilations", max(counter.compilations, counter.graphs), 2, unit="graphs")
        check_compile_calls(s.compiles, backend="eager", mode=None)


@case("five prompts of 5, 12, 16, 17, and 30 tokens: two padded lengths, three graphs",
      public=True)  # fmt: skip
def prompts_share_buckets(ctx):
    # Rejects no_padding (6 graphs), last_logits_of_padding, and decode_from_padded_length.
    with session(ctx, batch=2) as s:
        for n_prompt in (5, 12, 16, 17, 30):
            s.request(n_prompt, 5)
        s.finish(limit=3)


@case("64 decode steps after a 9-token prompt: still two graphs", public=True)
def sixty_four_steps(ctx):
    # python_int_position and growing_sequence reach Dynamo's recompile limit here.
    with session(ctx) as s:
        s.request(9, 64)
        s.finish(limit=2)


# Hidden cases.


@case("torch.compile receives fullgraph=True, dynamic=False, and the backend and mode it was given")
def compile_arguments(ctx):
    # Rejects default_dynamic and branch_without_fullgraph without compiling anything:
    # here the wrapper returns the function it was given.
    for backend, mode in (("inductor", "reduce-overhead"), ("eager", None)):
        guard(ctx)
        with CompileSpy(passthrough=False) as compiles:
            ctx.solution.make_step(TinyDecoder(), new_cache(1), backend=backend, mode=mode)
        check_guard()
        check_compile_calls(compiles, backend=backend, mode=mode)


@case("one new token; prompts that fill their bucket; and a bucket of 8")
def edge_lengths(ctx):
    # Rejects wasted_last_step (two calls of step for one new token).
    with session(ctx) as s:
        s.request(7, 1)
        s.request(16, 3)
        s.request(1, 2)
        s.request(32, 2)
        s.request(3, 4, bucket=8)
        s.request(12, 4, bucket=8)
        s.finish(limit=4)


@case("random requests, three rows per batch: one graph per distinct padded length, plus one")
def random_requests(ctx):
    rng = ctx.rng
    with session(ctx, batch=3) as s:
        lengths = [int(n) for n in rng.integers(1, 41, 5)]
        for n_prompt in lengths:
            s.request(n_prompt, int(rng.integers(2, 8)))
        s.finish(limit=len({-(-n // 16) for n in lengths}) + 1)


# --------------------------------------------------------------------------------------
# GPU case. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: the default backend with mode='reduce-overhead' gives the eager tokens",
      requires="cuda", time_limit_s=300)  # fmt: skip
def gpu_generate(ctx):
    device = torch.device("cuda")
    reference_model = TinyDecoder()
    model = TinyDecoder().to(device)
    cache = new_cache(2, device)
    step = ctx.solution.make_step(model, cache, mode="reduce-overhead")
    for n_prompt, n_new in ((5, 8), (12, 8), (20, 6)):
        prompt = torch.from_numpy(ctx.randint(0, VOCAB, (2, n_prompt)))
        want_tokens, _ = oracle(reference_model, prompt, n_new)
        out = ctx.solution.generate(step, prompt.to(device), n_new, device)
        check.that(
            isinstance(out, torch.Tensor) and out.device.type == "cuda",
            "generate: the tokens are not on the CUDA device",
        )
        check.equal(out.cpu().numpy(), want_tokens.numpy(), label="tokens")
    # The logits of one more step, cloned at once: with CUDA graphs a later call may
    # overwrite the memory of an earlier result.
    got = step(want_tokens[:, -1:].to(device), torch.tensor(n_prompt + n_new - 1, device=device))
    got = got.clone().float().cpu()
    sequence = torch.cat([prompt, want_tokens], dim=1)
    want = reference_model(sequence, torch.tensor(0), new_cache(2))[:, -1]
    check.allclose(got[:, 0].numpy(), want.numpy(), rtol=1e-3, atol=1e-3, label="step logits")
