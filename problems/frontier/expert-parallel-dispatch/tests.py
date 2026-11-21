"""Tests for expert-parallel-dispatch.

`run` gives the solution a layer, the devices its tokens and experts live on, and a
`SimDevices` ledger, then checks, in this order:

1. the output, against the dense evaluation of the layer (`moe_dense_ref`);
2. the ledger: no collectives and nothing stored; the number of messages (two per
   ordered pair of devices with routed tokens: rows out, results back) and
   `comm_bytes` (each routed row once per remote device it needs, both ways);
3. the expert work: one call per active expert with all of its rows, `k * N` rows;
4. what every message carried.

`SimDevices` is a ledger and cannot see where a computation ran: a solution could
evaluate the layer in one place and send tensors of the right sizes. The local
subclass `SpyDevices` therefore records the payload of every `send`, and step 4
compares each message with the rows, or the weighted partial results, that the
statement says cross between that pair of devices. Nothing else is added to the kit.

Both objects are behind the kit's `Guarded`. Of `experts` a solution may call the
experts and read their sizes; of `devs` it may use the operations and `n_devices`. The
counters, the call log, the message log, the payloads, `with_framework`, and the
experts' weights, seed, and `reference` are refused, so neither ledger can be written
and the experts cannot be evaluated anywhere the counts do not see.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import (
    CountingExperts,
    Guarded,
    SimDevices,
    TinyMoE,
    check_guard,
    guard,
    moe_dense_ref,
    moe_route_ref,
)

FN = "ep_forward"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
MARGIN = 1e-3


class Experts(Guarded, CountingExperts):
    """The experts the solution gets: `experts(e, rows)` and the sizes, nothing else."""

    label = "experts"
    refused = Guarded.refused | {"seed", "weights", "calls", "reference"}


class SpyDevices(Guarded, SimDevices):
    """SimDevices that remembers what every message carried, and keeps its ledger."""

    label = "devs"
    refused = Guarded.refused | {"messages", "payloads"}

    def __init__(self, fw, n_devices):
        super().__init__(fw, n_devices=n_devices)
        self.payloads = []  # (src, dst, numpy array), in order

    def send(self, src, dst, array):
        out = super().send(src, dst, array)
        self.payloads.append((int(src), int(dst), np.array(self._numpy(array, "send"))))
        return out


def clear(x, w_router, k, ties=False):
    """Whether rounding cannot change the routing: a margin at the top-k boundary."""
    if k == w_router.shape[0]:
        return True
    logits = np.sort(x.astype(np.float64) @ w_router.astype(np.float64).T, axis=-1)
    gap = logits[:, -k] - logits[:, -k - 1]
    if ties:
        gap = gap[gap != 0]
    return bool((gap > MARGIN).all())


def plan(x, home, w_router, experts, placement, k):
    """What the statement prescribes.

    Returns (messages, calls): `messages` maps an ordered pair (src, dst) to the list
    of (what it is, payload) that must cross from src to dst; `calls` is the sorted
    list of (expert, rows) of the expert calls.
    """
    chosen, weights = moe_route_ref(x, w_router, k)
    home, placement = np.asarray(home), np.asarray(placement)
    target = placement[chosen]
    n_devices = int(max(home.max(), placement.max())) + 1
    messages = {}
    for dst in range(n_devices):
        for src in range(n_devices):
            tokens = np.flatnonzero((home == src) & (target == dst).any(axis=1))
            if src == dst or tokens.size == 0:
                continue
            partial = np.zeros((tokens.size, x.shape[1]), dtype=np.float32)
            for row, t in enumerate(tokens):
                for j in range(k):
                    if target[t, j] == dst:
                        partial[row] += weights[t, j] * experts.reference(int(chosen[t, j]), x[t])
            shown = tokens.tolist()
            messages.setdefault((src, dst), []).append(
                (f"the rows of tokens {shown} (home {src}, experts on {dst})", x[tokens])
            )
            messages.setdefault((dst, src), []).append(
                (f"the results for tokens {shown} from the experts on {dst}", partial)
            )
    sizes = np.bincount(chosen.reshape(-1), minlength=experts.n_experts)
    return messages, sorted((e, int(n)) for e, n in enumerate(sizes) if n)


def run(ctx, x, home, w_router, experts, placement, k, n_devices, ties=False):
    x, w_router = x.astype(np.float32), w_router.astype(np.float32)
    home, placement = [int(h) for h in home], [int(p) for p in placement]
    if not clear(x, w_router, k, ties):  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: a token is near a tie at the top-k boundary")
    messages, calls = plan(x, home, w_router, experts, placement, k)
    n_messages = sum(len(items) for items in messages.values())
    n_bytes = sum(payload.nbytes for items in messages.values() for _, payload in items)
    setup = f"home = {home}\nplacement = {placement}\nk = {k}, {n_devices} devices\n"

    devs = SpyDevices(ctx.fw, n_devices)
    guard(ctx)  # the guarded experts share the counters and the log of `experts`
    y = ctx.call(FN, x, home, w_router, Experts.of(experts), placement, devs, k)
    check_guard()

    # 1. The output.
    check.allclose(y, moe_dense_ref(x, w_router, experts, k), **TOL, label=FN, dtype=True)

    # 2. The ledger.
    counters = devs.counters
    check.that(
        counters.all_reduces == 0 and counters.all_gathers == 0 and counters.bytes_placed == 0,
        f"{FN}: used a collective or stored tensors on the devices; rows and results "
        "travel by devs.send only",
        f"{counters!r}",
    )
    check.cost(counters, "sends", exactly=n_messages, unit="messages")
    check.cost(counters, "comm_bytes", exactly=n_bytes, unit="bytes")

    # 3. The expert work.
    check.cost(experts.counters, "expert_rows", exactly=k * len(x), unit="rows")
    check.cost(experts.counters, "expert_calls", at_most=len(calls), unit="calls")
    check.that(
        sorted(experts.calls) == calls,
        f"{FN}: the calls made are not one call per active expert with all of its rows",
        f"calls made, as (expert, rows): {sorted(experts.calls)}\nexpected: {calls}",
    )

    # 4. What each message carried.
    for (src, dst), items in messages.items():
        got = [payload for a, b, payload in devs.payloads if (a, b) == (src, dst)]
        for what, want in items:
            exact = "rows of tokens" in what
            match = next(
                (
                    i
                    for i, payload in enumerate(got)
                    if payload.shape == want.shape
                    and (
                        np.array_equal(payload, want)
                        if exact
                        else np.allclose(payload, want, **TOL)
                    )
                ),
                None,
            )
            shapes = [payload.shape for payload in got]
            check.that(
                match is not None,
                f"no message from device {src} to device {dst} carries what the rules "
                f"prescribe there (it sent tensors of shapes {shapes})",
                f"{setup}missing: {what}, in increasing token order\nexpected payload =\n"
                f"{want}\nmessages from {src} to {dst} =\n"
                + "\n".join(str(payload) for payload in got)
                + "\n",
            )
            got.pop(match)
    return y, devs


def layer_inputs(ctx, moe, layer, batch, length):
    """Rows that layer `layer` of the model receives, redrawn until the routing is clear."""
    w_router = ctx.fw.numpy(moe.router(layer))
    while True:
        x = moe.mlp_inputs(ctx.randint(0, 64, (batch, length)), layer)
        if clear(x, w_router, moe.top_k):
            return x, w_router


def random_layer(ctx, n, d, e, k):
    while True:
        x = ctx.randn(n, d)
        w_router = ctx.randn(e, d, scale=d**-0.5)
        if clear(x, w_router, k):
            return x, w_router


EX_X = np.array([[3, 1], [-1, 2], [2, -3], [1, 3], [-3, 1]], dtype=np.float32)
EX_ROUTER = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]], dtype=np.float32)
EX_HOME = [0, 0, 1, 1, 1]
EX_PLACEMENT = [0, 0, 1, 1, 2]


@case("the worked example: five tokens on two devices, five experts on three", public=True)
def worked_example(ctx):
    experts = CountingExperts(ctx.fw, seed=1, n_experts=5, d_model=2, d_hidden=4)
    _, devs = run(ctx, EX_X, EX_HOME, EX_ROUTER, experts, EX_PLACEMENT, 2, 3)
    check.equal(
        sorted(devs.messages), [(0, 1, 8), (0, 1, 24), (1, 0, 8), (1, 0, 24)], label="messages"
    )
    check.equal(sorted(experts.calls), [(0, 3), (1, 4), (2, 2), (3, 1)], label="expert calls")


@case("one device holds everything: no message. Tokens and experts apart: one each way",
      public=True)  # fmt: skip
def one_device_and_two(ctx):
    moe = TinyMoE(ctx.fw, seed=2, n_experts=8, top_k=2)
    x, w_router = layer_inputs(ctx, moe, 1, 2, 6)
    run(ctx, x, [0] * 12, w_router, moe.experts(1), [0] * 8, 2, 1)
    # All tokens on device 0, all experts on device 1: 12 rows out, 12 results back.
    x, w_router = layer_inputs(ctx, moe, 0, 2, 6)
    run(ctx, x, [0] * 12, w_router, moe.experts(0), [1] * 8, 2, 2)


@case("a model's layer on four devices: 24 tokens, eight experts, two per device", public=True)
def model_layer(ctx):
    moe = TinyMoE(ctx.fw, seed=2, n_experts=8, top_k=2)
    x, w_router = layer_inputs(ctx, moe, 1, 2, 12)
    home = [t % 4 for t in range(24)]
    run(ctx, x, home, w_router, moe.experts(1), [0, 0, 1, 1, 2, 2, 3, 3], 2, 4)


@case("a token whose experts share a remote device crosses once, and so does its result",
      public=True)  # fmt: skip
def shared_remote_device(ctx):
    # Tokens 0 and 3 of the example now live on device 1 and use experts 0 and 1, both
    # on device 0: two rows out and two back, not four.
    experts = CountingExperts(ctx.fw, seed=1, n_experts=5, d_model=2, d_hidden=4)
    _, devs = run(ctx, EX_X, [1, 1, 1, 1, 1], EX_ROUTER, experts, [0, 0, 1, 1, 0], 2, 2)
    check.equal(sorted(devs.messages), [(0, 1, 40), (1, 0, 40)], label="messages")


@case("uneven placement: a device with five experts, one with none, one without tokens")
def uneven_placement(ctx):
    moe = TinyMoE(ctx.fw, seed=5, n_experts=8, top_k=2)
    x, w_router = layer_inputs(ctx, moe, 0, 3, 8)
    home = [0 if t < 14 else 3 for t in range(24)]
    run(ctx, x, home, w_router, moe.experts(0), [0, 0, 0, 0, 0, 1, 1, 3], 2, 4)


@case("k = 1, and k = E: every token to every expert")
def k_extremes(ctx):
    for k in (1, 6):
        x, w_router = random_layer(ctx, 9, 8, 6, k)
        experts = CountingExperts(ctx.fw, seed=k, n_experts=6, d_model=8, d_hidden=12)
        home = [t % 3 for t in range(9)]
        run(ctx, x, home, w_router, experts, [0, 1, 2, 2, 1, 0], k, 3)


@case("equal router logits: the expert with the lowest index decides where the row goes")
def tied_logits(ctx):
    # Experts 0 and 4, and 1 and 3, have equal router rows and live on different devices.
    w_router = np.array(
        [[1, 0, 2], [0, 1, 0], [1, 1, -1], [0, 1, 0], [1, 0, 2], [-2, 0, 1]], dtype=np.float32
    )
    x = np.array(
        [[2, 1, 0], [0, 3, 1], [1, 0, 2], [-1, 2, 0], [3, -1, 1], [0, 1, 0]], dtype=np.float32
    )
    experts = CountingExperts(ctx.fw, seed=3, n_experts=6, d_model=3, d_hidden=5)
    run(ctx, x, [0, 1, 2, 0, 1, 2], w_router, experts, [0, 1, 2, 0, 1, 2], 2, 3, ties=True)
    run(ctx, x, [0, 1, 2, 0, 1, 2], w_router, experts.with_framework(ctx.fw), [2, 2, 1, 1, 0, 0],
        3, 3, ties=True)  # fmt: skip


@case("a single token, and tokens that are all identical")
def degenerate_batches(ctx):
    moe = TinyMoE(ctx.fw, seed=7, n_experts=8, top_k=2)
    x, w_router = layer_inputs(ctx, moe, 1, 1, 6)
    run(ctx, x[:1], [2], w_router, moe.experts(1), [0, 1, 2, 0, 1, 2, 0, 1], 2, 3)
    same = np.repeat(x[2:3], 6, axis=0)
    experts = moe.experts(1).with_framework(ctx.fw)
    run(ctx, same, [0, 0, 1, 1, 2, 2], w_router, experts, [0, 1, 2, 0, 1, 2, 0, 1], 2, 3)


@case("eight devices with one expert each: 32 tokens, k = 3")
def one_expert_per_device(ctx):
    moe = TinyMoE(ctx.fw, seed=11, n_experts=8, top_k=3)
    x, w_router = layer_inputs(ctx, moe, 0, 2, 16)
    home = [int(h) for h in ctx.rng.integers(0, 8, size=32)]
    run(ctx, x, home, w_router, moe.experts(0), [3, 0, 7, 5, 1, 6, 2, 4], 3, 8)
