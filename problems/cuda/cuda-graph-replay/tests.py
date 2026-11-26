"""Tests for cuda-graph-replay.

On the CPU the graph is the kit's `FakeCudaGraph`: between `capture_begin()` and
`capture_end()` it records every tensor operation with the tensor objects it read and
wrote, and `replay()` reruns them on those same tensors. So the consequences of
misusing a CUDA graph are the ones a GPU shows: a placeholder that was rebound instead
of copied into replays the old input, an output handed out without `clone()` changes
at the next call, a placeholder allocated inside the capture is reset by every
replay, and a host read during capture raises. `with graph.patched():` makes
`torch.cuda.graph(graph)` accept the stand-in, so a solution written with the context
manager runs unchanged.

A case wraps the step function to see when it is called (`graph.capturing` on each
call) and with what, builds a `GraphedStep`, and then calls it with a sequence of
inputs. It checks:

- **the calls of `step_fn`**: `warmup` eager calls, then exactly one while capturing,
  all on tensors that hold the example values and are not the caller's examples; and
  none ever again;
- **every output**, against `step_fn` run eagerly on the same input; and every earlier
  output again after the later calls (an aliased output has changed by then);
- **the counters**: one capture, one replay per call;
- that neither the caller's examples nor the caller's inputs were written to;
- `ValueError` for a wrong number of inputs or a wrong shape, with nothing replayed.

The wrapper must not run torch operations while the graph is capturing (they would be
recorded), so it copies its inputs through NumPy.

The graph and the wrapper are behind the kit's `Guarded`. A solution may use the
graph's `capture_begin`, `capture_end`, `replay`, `reset`, `capturing`, `captured`, and
`n_ops` (and `torch.cuda.graph(graph)`), and may call `step_fn`. The graph's counters,
the wrapper's record of calls, and the function behind the wrapper are refused.

The stateful case replays a decode step of a small transformer that writes into a
static KV cache, next to an eager twin with a cache of its own.

The GPU case (`requires="cuda"`) uses a real `torch.cuda.CUDAGraph`. It was written
from the PyTorch 2.15 documentation and has never run on the machine that validates
the catalog.
"""

import math

import numpy as np
import torch

from learnai.judge import case, check
from learnai.kit import FakeCudaGraph, Guarded, check_guard, guard

NAME = "GraphedStep"
VOCAB, DIM, HEADS, LAYERS, MAX_LEN = 24, 16, 2, 2, 24


class Graph(Guarded, FakeCudaGraph):
    label = "graph"
    # `_ops` stays readable: the recorder appends to it under the solution's own tensor
    # operations during the capture, which the guard cannot tell from the solution.
    allowed = frozenset(
        {"capture_begin", "capture_end", "replay", "reset", "capturing", "captured", "n_ops"}
        | {"__class__", "_ops"}
    )


class Harness(Guarded):
    """A step function under observation, the graph, and the GraphedStep built on them.

    The solution is handed `self.step_fn`, and nothing of the harness can be read
    through it."""

    label = "step_fn"
    allowed = frozenset({"__class__"})

    def __init__(self, ctx, fn, examples, warmup=None, graph=None):
        self.ctx, self.fn = ctx, fn
        self.graph = Graph() if graph is None else graph
        self.fake = isinstance(self.graph, FakeCudaGraph)
        self.examples = tuple(examples)
        self.example_values = [x.detach().cpu().numpy().copy() for x in self.examples]
        self.warmup = 3 if warmup is None else warmup
        self.calls = []  # (capturing, [input values], [input data_ptr])
        self.results = []  # (output or tuple of outputs, expected NumPy arrays)
        extra = {} if warmup is None else {"warmup": warmup}
        guard(ctx)
        if self.fake:
            with self.graph.patched():
                self.step = ctx.solution.GraphedStep(
                    self.step_fn, self.examples, self.graph, **extra
                )
        else:
            self.step = ctx.solution.GraphedStep(self.step_fn, self.examples, self.graph, **extra)
        check_guard()
        self.check_construction()

    def step_fn(self, *inputs):
        # No torch operation here: during the capture it would be recorded.
        capturing = self.graph.capturing if self.fake else None
        values = [x.numpy().copy() for x in inputs] if self.fake else None
        self.calls.append((capturing, values, [x.data_ptr() for x in inputs]))
        return self.fn(*inputs)

    def check_construction(self):
        flags = [capturing for capturing, _, _ in self.calls]
        detail = (
            f"expected {self.warmup} calls before the capture and one during it\n"
            f"graph.capturing at each call of step_fn: {flags}"
        )
        if self.fake:
            check.that(
                self.graph.captured and not self.graph.capturing,
                f"{NAME}: after the constructor the graph holds no finished capture",
            )
            check.that(
                flags.count(True) == 1,
                f"{NAME}: step_fn was called {flags.count(True)} times while the graph was "
                "capturing; the capture is exactly one call",
                detail,
            )
            check.that(
                flags[-1] is True,
                f"{NAME}: step_fn was called again after the capture",
                detail,
            )
            check.that(
                flags.count(False) > 0,
                f"{NAME}: step_fn was never run before the capture (no warm-up)",
                detail,
            )
        check.that(
            len(flags) == self.warmup + 1,
            f"{NAME}: the constructor called step_fn {len(flags)} times, expected "
            f"{self.warmup + 1} (the warm-up calls plus the capture)",
            detail,
        )
        example_pointers = {x.data_ptr() for x in self.examples}
        for _, values, pointers in self.calls:
            if values is not None:
                check.that(
                    len(values) == len(self.example_values)
                    and all(
                        got.shape == want.shape and np.array_equal(got, want)
                        for got, want in zip(values, self.example_values, strict=False)
                    ),
                    f"{NAME}: a warm-up or capture call of step_fn did not get the example values",
                    "the placeholders must hold the example inputs' values when step_fn "
                    "runs on them",
                )
            check.that(
                not example_pointers & set(pointers),
                f"{NAME}: step_fn was called on the caller's example tensors; the "
                "placeholders must be memory of their own",
            )
        self.n_construction_calls = len(self.calls)

    def run(self, *inputs):
        """One call of the GraphedStep, with the guard armed before it and checked after."""
        guard(self.ctx)
        try:
            return self.step(*inputs)
        finally:
            check_guard()

    def call(self, *inputs, label="replayed output"):
        """One call of the GraphedStep on fresh tensors; checked against eager."""
        given = [x.clone() for x in inputs]
        want = self.fn(*[x.clone() for x in inputs])
        got = self.run(*inputs)
        single = isinstance(want, torch.Tensor)
        check.that(
            isinstance(got, torch.Tensor) if single else isinstance(got, tuple),
            f"{NAME}: returned a {type(got).__name__}, expected a "
            f"{'tensor' if single else 'tuple'} (the kind of value step_fn returns)",
            "a tensor when step_fn returns a tensor, a tuple when it returns a tuple",
        )
        got_list, want_list = ([got], [want]) if single else (list(got), list(want))
        check.that(
            len(got_list) == len(want_list),
            f"{NAME}: returned {len(got_list)} outputs, expected {len(want_list)} (the number "
            "step_fn returns)",
        )
        want_arrays = [w.detach().cpu().numpy().copy() for w in want_list]
        for index, (g, w) in enumerate(zip(got_list, want_arrays, strict=True)):
            name = label if single else f"{label}[{index}]"
            check.that(isinstance(g, torch.Tensor), f"{name}: is not a tensor")
            check.match(g.detach().cpu().numpy(), w, label=name)
        for x, before in zip(inputs, given, strict=True):
            check.that(
                bool(torch.equal(x, before)), f"{NAME}: the caller's input was modified in place"
            )
        self.results.append((got_list, want_arrays))
        return got

    def finish(self):
        """After the last call: nothing aliased, nothing rerun, one replay per call."""
        for number, (got_list, want_arrays) in enumerate(self.results, start=1):
            for g, w in zip(got_list, want_arrays, strict=True):
                same = np.array_equal(g.detach().cpu().numpy(), w, equal_nan=True) or np.allclose(
                    g.detach().cpu().numpy(), w, rtol=1e-4, atol=1e-5, equal_nan=True
                )
                check.that(
                    same,
                    f"{NAME}: the output of call {number} changed after a later call; it "
                    "shares memory with the graph's static output",
                    "every replay overwrites the static outputs in place, so the caller "
                    "must get a copy",
                )
        pointers = [g.data_ptr() for got_list, _ in self.results for g in got_list if g.numel()]
        check.that(
            len(set(pointers)) == len(pointers),
            f"{NAME}: two outputs that are both still alive share memory",
        )
        for x, before in zip(self.examples, self.example_values, strict=True):
            check.that(
                np.array_equal(x.detach().cpu().numpy(), before),
                f"{NAME}: the caller's example inputs were written to",
                "the placeholders must be tensors of their own, not the examples",
            )
        check.that(
            len(self.calls) == self.n_construction_calls,
            f"{NAME}: step_fn ran {len(self.calls) - self.n_construction_calls} more times "
            "after the capture; a call must replay the graph, not rerun the function",
        )
        if self.fake:
            check.cost(self.graph.counters, "captures", exactly=1, unit="captures")
            check.cost(self.graph.counters, "replays", exactly=len(self.results), unit="replays")


def tensor(values, dtype=np.float32):
    return torch.from_numpy(np.asarray(values, dtype=dtype))


# --------------------------------------------------------------------------------------
# A stateful step: a decode step of a small transformer over a static KV cache
# --------------------------------------------------------------------------------------


class StaticCache:
    """The cache of `cuda-static-kv-cache`: full-length buffers, written in place."""

    def __init__(self, batch, device):
        shape = (batch, HEADS, MAX_LEN, DIM // HEADS)
        self.k = [torch.zeros(shape, device=device) for _ in range(LAYERS)]
        self.v = [torch.zeros(shape, device=device) for _ in range(LAYERS)]
        self._slots = torch.arange(MAX_LEN, device=device)

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


class TinyDecoder:
    """decoder(tokens, pos, cache) -> logits (batch, T, VOCAB), writing the cache in place."""

    def __init__(self, rng, device):
        def weight(*shape, scale):
            values = rng.standard_normal(shape).astype(np.float32) * scale
            return torch.from_numpy(values).to(device)

        self.embed = weight(VOCAB, DIM, scale=1.0)
        self.position = weight(MAX_LEN, DIM, scale=1.0)
        self.wq, self.wk, self.wv, self.wo = (
            [weight(DIM, DIM, scale=DIM**-0.5) for _ in range(LAYERS)] for _ in range(4)
        )
        self.head = weight(DIM, VOCAB, scale=1.0)

    def heads(self, x):
        batch, n_tokens, _ = x.shape
        return x.view(batch, n_tokens, HEADS, DIM // HEADS).transpose(1, 2)

    def __call__(self, tokens, pos, cache):
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


def decode_through_graph(ctx, batch, n_prompt, n_steps, device, graph=None):
    """A prefill run eagerly, then n_steps decode steps through a GraphedStep, compared
    step by step with the same decoder on a cache of its own."""
    decoder = TinyDecoder(ctx.rng, device)
    graphed_cache, eager_cache = StaticCache(batch, device), StaticCache(batch, device)

    def step_fn(token, pos):
        return decoder(token, pos, graphed_cache)[:, 0]

    examples = (
        torch.zeros((batch, 1), dtype=torch.int64, device=device),
        torch.tensor(0, device=device),
    )
    # Built before the prefill: the warm-up and the capture run the step for real, and
    # write the example token into slot 0 of the cache. The prefill overwrites it.
    harness = Harness(ctx, step_fn, examples, graph=graph)
    prompt = torch.from_numpy(ctx.randint(0, VOCAB, (batch, n_prompt))).to(device)
    start = torch.tensor(0, device=device)
    logits = decoder(prompt, start, graphed_cache)[:, -1]
    decoder(prompt, start, eager_cache)
    for i in range(n_steps):
        token = logits.argmax(dim=-1, keepdim=True)
        pos = torch.tensor(n_prompt + i, device=device)
        want = decoder(token, pos, eager_cache)[:, 0]
        got = harness.run(token, pos)
        check.that(
            isinstance(got, torch.Tensor) and tuple(got.shape) == (batch, VOCAB),
            f"{NAME}: the replayed decode step did not return logits of the expected shape",
        )
        check.allclose(
            got.cpu().numpy(), want.cpu().numpy(), rtol=1e-4, atol=1e-4, label="replayed logits"
        )
        harness.results.append(([got], [want.cpu().numpy().copy()]))
        logits = want
    used = n_prompt + n_steps
    for layer in range(LAYERS):
        for name, mine, twin in (
            ("k", graphed_cache.k[layer], eager_cache.k[layer]),
            ("v", graphed_cache.v[layer], eager_cache.v[layer]),
        ):
            check.allclose(
                mine[:, :, :used].cpu().numpy(),
                twin[:, :, :used].cpu().numpy(),
                rtol=1e-4,
                atol=1e-5,
                label=f"cache {name}[{layer}] after the replayed steps",
            )
    harness.finish()


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: x * 2 + 1, captured once and replayed twice", public=True)
def worked_example(ctx):
    harness = Harness(ctx, lambda x: x * 2 + 1, (tensor([1.0, 2.0, 3.0]),))
    a = harness.call(tensor([10.0, 20.0, 30.0]))
    check.allclose(a.numpy(), [21.0, 41.0, 61.0], label="first call")
    b = harness.call(tensor([0.0, 0.0, 1.0]))
    check.allclose(b.numpy(), [1.0, 1.0, 3.0], label="second call")
    check.allclose(a.numpy(), [21.0, 41.0, 61.0], label="the first result, after the second call")
    harness.finish()


@case("two inputs and a tuple of two outputs, six calls", public=True)
def two_inputs_two_outputs(ctx):
    # Rejects rebinds_placeholder, placeholders_made_in_capture, returns_static_output,
    # clone_before_replay, and never_replays.
    def fn(x, w):
        hidden = torch.relu(x @ w)
        return hidden, hidden.sum(dim=-1)

    harness = Harness(ctx, fn, (tensor(ctx.randn(4, 8)), tensor(ctx.randn(8, 5))))
    for _ in range(6):
        out = harness.call(tensor(ctx.randn(4, 8)), tensor(ctx.randn(8, 5)))
        check.that(
            tuple(out[0].shape) == (4, 5) and tuple(out[1].shape) == (4,),
            f"{NAME}: the outputs do not have the shapes step_fn returns",
        )
    harness.finish()


@case("a decode step that writes into a static KV cache, replayed for eight tokens",
      public=True)  # fmt: skip
def stateful_decode_step(ctx):
    decode_through_graph(ctx, batch=2, n_prompt=5, n_steps=8, device=torch.device("cpu"))


@case("a wrong number of inputs or a wrong shape raises ValueError and replays nothing",
      public=True)  # fmt: skip
def rejects_wrong_shapes(ctx):
    # Rejects no_shape_check: copy_ would broadcast a (1, 8) row into the (4, 8) placeholder.
    harness = Harness(ctx, lambda x, w: x @ w, (tensor(ctx.randn(4, 8)), tensor(ctx.randn(8, 3))))
    harness.call(tensor(ctx.randn(4, 8)), tensor(ctx.randn(8, 3)))
    good_x, good_w = tensor(ctx.randn(4, 8)), tensor(ctx.randn(8, 3))
    bad = (
        ((tensor(ctx.randn(1, 8)), good_w), "an input that broadcasts to the placeholder"),
        ((tensor(ctx.randn(8)), good_w), "an input with fewer dimensions"),
        ((tensor(ctx.randn(4, 9)), good_w), "an input with a different shape"),
        ((good_x,), "too few inputs"),
        ((good_x, good_w, good_w), "too many inputs"),
    )
    for inputs, what in bad:
        check.raises(ValueError, harness.run, *inputs, label=f"{NAME} called with {what}")
    check.cost(harness.graph.counters, "replays", exactly=1, unit="replays")
    harness.call(good_x, good_w)
    harness.finish()


# Hidden cases.


@case("warmup=1 and warmup=5: that many eager calls, then one capture")
def warmup_count(ctx):
    # Rejects no_warmup, warms_up_inside_capture, and ignores_warmup_argument.
    for warmup in (1, 5):
        harness = Harness(ctx, lambda x: torch.tanh(x) * 3, (tensor(ctx.randn(6)),), warmup=warmup)
        for _ in range(3):
            harness.call(tensor(ctx.randn(6)))
        harness.finish()


@case("integer and 0-d inputs: an index tensor and a position")
def integer_inputs(ctx):
    table = tensor(ctx.randn(10, 4))

    def fn(index, pos):
        return table[index] * (pos + 1), index + pos

    examples = (tensor([0, 1, 2], dtype=np.int64), torch.tensor(1))
    harness = Harness(ctx, fn, examples)
    for step in range(5):
        index = tensor(ctx.randint(0, 10, (3,)), dtype=np.int64)
        harness.call(index, torch.tensor(step))
    harness.finish()


@case("an in-place step function: a running sum kept in a buffer")
def in_place_step(ctx):
    # The step updates state of its own; replays must update the same memory, once each.
    total = torch.zeros(5)

    def fn(x):
        total.add_(x)
        return total * 0.5

    harness = Harness(ctx, fn, (tensor(np.zeros(5)),), warmup=2)
    expected = np.zeros(5, dtype=np.float32)
    for _ in range(6):
        x = ctx.randn(5)
        expected += x
        got = harness.run(tensor(x))
        check.allclose(got.numpy(), expected * 0.5, atol=1e-5, label="replayed output")
        harness.results.append(([got], [(expected * 0.5).copy()]))
    check.allclose(total.numpy(), expected, atol=1e-5, label="the step function's own buffer")
    harness.finish()


@case("the caller reuses one input tensor, overwriting it between calls")
def caller_reuses_input(ctx):
    # A caller that refills one buffer: the placeholder must be a copy, taken at call time.
    harness = Harness(ctx, lambda x: x.cumsum(dim=0), (tensor(ctx.randn(7)),))
    buffer = torch.zeros(7)
    for _ in range(5):
        buffer.copy_(tensor(ctx.randn(7)))
        harness.call(buffer)
    harness.finish()


@case("random step functions and call sequences")
def random_sequences(ctx):
    shapes = [(3,), (2, 6), (4, 1, 5)]

    def prop(rng):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        scale = tensor(rng.standard_normal(shape))
        shift = float(rng.standard_normal())

        def fn(x, y):
            mixed = torch.sigmoid(x * scale + y)
            return mixed + shift, (mixed * x).sum()

        def draw():
            return tensor(rng.standard_normal(shape))

        harness = Harness(ctx, fn, (draw(), draw()), warmup=int(rng.integers(1, 4)))
        for _ in range(int(rng.integers(1, 6))):
            harness.call(draw(), draw())
        harness.finish()

    check.forall(prop, trials=8)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: a real CUDA graph of a two-layer network matches the CPU", requires="cuda")
def gpu_two_layer(ctx):
    device = torch.device("cuda")
    w1, w2 = tensor(ctx.randn(16, 32, scale=0.25)), tensor(ctx.randn(32, 8, scale=0.25))
    w1_gpu, w2_gpu = w1.to(device), w2.to(device)
    calls = []

    def step_fn(x):
        calls.append(None)
        return torch.relu(x @ w1_gpu) @ w2_gpu

    example = torch.zeros((4, 16), device=device)
    step = ctx.solution.GraphedStep(step_fn, (example,), torch.cuda.CUDAGraph())
    check.that(
        len(calls) == 4,
        f"{NAME}: the constructor called step_fn {len(calls)} times, expected 4 (the "
        "warm-up calls plus the capture)",
    )
    outputs = []
    for _ in range(5):
        x = tensor(ctx.randn(4, 16))
        out = step(x.to(device))
        check.that(
            isinstance(out, torch.Tensor) and out.device.type == "cuda",
            f"{NAME}: the output is not a tensor on the CUDA device",
        )
        outputs.append((out, (torch.relu(x @ w1) @ w2).numpy()))
    torch.cuda.synchronize(device)
    for out, want in outputs:
        check.allclose(out.cpu().numpy(), want, rtol=1e-3, atol=1e-4, label="replayed output")
    check.that(len(calls) == 4, f"{NAME}: step_fn ran again after the capture")


@case("on the GPU: the decode step over a static cache, through a real CUDA graph",
      requires="cuda")  # fmt: skip
def gpu_decode_step(ctx):
    decode_through_graph(
        ctx,
        batch=2,
        n_prompt=5,
        n_steps=8,
        device=torch.device("cuda"),
        graph=torch.cuda.CUDAGraph(),
    )
