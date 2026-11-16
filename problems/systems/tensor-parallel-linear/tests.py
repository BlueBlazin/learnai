"""Tests for tensor-parallel-linear.

`shard_weights` is compared exactly with slices computed here. For `tp_mlp` the test
stores shards on a `SimDevices` whose budgets are exactly full and checks three
things: the output against the unsplit layer (numpy), the ledger (one all-reduce,
no all-gather, no send, `comm_bytes == n * nbytes(y)`), and what each device
contributed to the all-reduce.

`SimDevices` is a ledger and cannot see where a computation ran, so a solution could
compute the whole layer in one place and pass the result through `all_reduce`. The
local subclass `SpyDevices` therefore records the tensors handed to every collective,
and the test compares the tensor of device i with the partial output of device i's
shard. Nothing else is added to the kit object.
"""

from itertools import pairwise

import numpy as np

from learnai.judge import case, check
from learnai.kit import SimDevices
from learnai.kit.simdevices import OutOfDeviceMemory

NAMES = ("up", "up_bias", "down", "down_bias")
RTOL, ATOL = 1e-4, 1e-5


class SpyDevices(SimDevices):
    """SimDevices that remembers the parts handed to all_reduce and all_gather."""

    def __init__(self, fw, memory_bytes):
        super().__init__(fw, memory_bytes=memory_bytes)
        self.reduced = []
        self.gathered = []

    def all_reduce(self, parts):
        out = super().all_reduce(parts)
        self.reduced.append([np.array(self._numpy(p, "all_reduce")) for p in parts])
        return out

    def all_gather(self, parts, axis=-1):
        out = super().all_gather(parts, axis)
        self.gathered.append([np.array(self._numpy(p, "all_gather")) for p in parts])
        return out


def silu(z):
    return z / (1.0 + np.exp(-z))


def make_layer(ctx, d, h, d_out, *, bias_scale=1.0):
    w_up = ctx.randn(h, d, scale=1.0 / np.sqrt(d))
    b_up = ctx.randn(h, scale=0.5)
    w_down = ctx.randn(d_out, h, scale=1.0 / np.sqrt(h))
    b_down = ctx.randn(d_out, scale=bias_scale)
    return w_up, b_up, w_down, b_down


def true_shards(layer, n):
    w_up, b_up, w_down, b_down = layer
    h = w_up.shape[0]
    bounds = [i * h // n for i in range(n + 1)]
    return [
        {"up": w_up[lo:hi], "up_bias": b_up[lo:hi], "down": w_down[:, lo:hi], "down_bias": b_down}
        for lo, hi in pairwise(bounds)
    ]


def shard_bytes(shard):
    return sum(np.asarray(shard[name]).nbytes for name in NAMES)


def describe(layer, n, x=None):
    w_up, _, w_down, _ = layer
    text = f"d={w_up.shape[1]}, h={w_up.shape[0]}, d_out={w_down.shape[0]}, n={n}"
    return text + (f", x of shape {x.shape}" if x is not None else "") + "\n"


def check_shards(ctx, layer, n):
    """Call shard_weights and compare it with the slices the statement defines."""
    want = true_shards(layer, n)
    got = ctx.call("shard_weights", *layer, n)
    setup = describe(layer, n)
    check.that(
        isinstance(got, (list, tuple)) and all(isinstance(s, dict) for s in got),
        "shard_weights: must return a list of dicts",
        f"got: {type(got).__name__}",
    )
    check.equal(len(got), n, label="number of shards")
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        check.equal(sorted(g), sorted(NAMES), label=f"keys of shard {i}")
        for name in NAMES:
            check.that(
                isinstance(g[name], np.ndarray) and g[name].dtype == np.float32,
                f"shard {i}: {name!r} is not a float32 tensor",
            )
            check.shape(g[name], w[name].shape, label=f"shard {i}: {name!r}")
            check.that(
                np.array_equal(g[name], w[name]),
                f"shard {i}: {name!r} is not the slice of the weights this device owns",
                f"{setup}expected =\n{w[name]}\ngot =\n{g[name]}\n",
            )
    return got


def place(ctx, shards, budgets):
    devs = SpyDevices(ctx.fw, budgets)
    for i, shard in enumerate(shards):
        for name in NAMES:
            try:
                devs.put(i, name, np.asarray(shard[name], dtype=np.float32))
            except OutOfDeviceMemory as exc:
                check.that(
                    False,
                    f"shard {i} does not fit on its device: it is larger than the balanced "
                    "split allows",
                    str(exc),
                )
    return devs


def check_tp(ctx, layer, n, x, *, shards=None):
    """Run tp_mlp on budgeted devices; check the output, the ledger, and the parts."""
    w_up, b_up, w_down, b_down = layer
    truth = true_shards(layer, n)
    budgets = [shard_bytes(s) for s in truth]
    devs = place(ctx, truth if shards is None else shards, budgets)
    placed = devs.counters.bytes_placed
    setup = describe(layer, n, x)

    y = ctx.call("tp_mlp", x, devs)
    want = silu(x @ w_up.T + b_up) @ w_down.T + b_down
    check.allclose(y, want, rtol=RTOL, atol=ATOL, label="tp_mlp", dtype=True)

    check.that(
        devs.counters.bytes_placed == placed,
        "tp_mlp stored tensors on the devices; it must only read the shards",
        setup,
    )
    check.cost(devs.counters, "all_gathers", exactly=0)
    check.cost(devs.counters, "sends", exactly=0)
    check.cost(devs.counters, "all_reduces", exactly=1)
    check.cost(devs.counters, "comm_bytes", exactly=n * want.nbytes, unit="bytes")

    parts = devs.reduced[0]
    for i, (part, shard) in enumerate(zip(parts, truth, strict=True)):
        partial = silu(x @ shard["up"].T + shard["up_bias"]) @ shard["down"].T
        ok = part.shape == partial.shape and np.allclose(part, partial, rtol=RTOL, atol=ATOL)
        check.that(
            ok,
            f"the tensor device {i} contributed to the all-reduce is not the partial output "
            "of its own shard",
            f"{setup}device {i} holds hidden units of shapes up {shard['up'].shape}, down "
            f"{shard['down'].shape}\nexpected partial output =\n{partial}\ngot =\n{part}\n",
        )
    return y


EX_LAYER = (
    np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.float32),
    np.array([0, -1, 0, 3], dtype=np.float32),
    np.array([[1, 0, 1, 0], [0, 2, 0, -1]], dtype=np.float32),
    np.array([10, 20], dtype=np.float32),
)
EX_X = np.array([1, 2], dtype=np.float32)


# Rejects bias_on_every_device ([23.59, 39.70]), down_split_by_rows, and every mutant
# that changes the ledger.
@case("the worked example: d=2, h=4, d_out=2 on two devices", public=True)
def worked_example(ctx):
    shards = check_shards(ctx, EX_LAYER, 2)
    check.equal(shards[1]["down"], np.array([[1, 0], [0, -1]], dtype=np.float32), label="down")
    y = check_tp(ctx, EX_LAYER, 2, EX_X)
    check.allclose(y, np.array([13.588781, 19.700523], dtype=np.float32), label="tp_mlp")


# Rejects remainder_on_last (2, 2, 2, 4) and overlapping_shards.
@case("shard_weights: h=10 on four devices gives blocks of 2, 3, 2, 3 hidden units",
      public=True)  # fmt: skip
def uneven_shards(ctx):
    layer = make_layer(ctx, 6, 10, 3)
    shards = check_shards(ctx, layer, 4)
    check.equal([s["up"].shape[0] for s in shards], [2, 3, 2, 3], label="hidden units per shard")


# Rejects gather_between_layers, single_device, fake_all_reduce, sends_to_device_zero,
# reduce_per_row.
@case("tp_mlp on four devices, a batch of 5: one all-reduce of the output's size, and "
      "each device contributes its own partial output", public=True)  # fmt: skip
def batch_on_four_devices(ctx):
    layer = make_layer(ctx, 6, 16, 4)
    check_tp(ctx, layer, 4, ctx.randn(5, 6))


# Rejects overlapping_shards and remainder_on_last through the memory budgets.
@case("end to end: your shards stored on devices with exact budgets, then tp_mlp",
      public=True)  # fmt: skip
def end_to_end(ctx):
    layer = make_layer(ctx, 6, 10, 3)
    shards = ctx.call("shard_weights", *layer, 4)
    check.that(
        isinstance(shards, (list, tuple)) and len(shards) == 4,
        "shard_weights: must return one shard per device",
    )
    check_tp(ctx, layer, 4, ctx.randn(5, 6), shards=shards)


# Rejects bias_on_every_device even when the weights are small.
@case("a large output bias is added exactly once")
def bias_once(ctx):
    layer = make_layer(ctx, 6, 16, 4, bias_scale=50.0)
    check_tp(ctx, layer, 4, ctx.randn(5, 6))
    check_tp(ctx, layer, 2, ctx.randn(5, 6))


# Rejects down_split_by_rows where the two axes of w_down have the same length.
@case("d_out equal to h: the second layer is cut along its columns")
def square_down(ctx):
    layer = make_layer(ctx, 6, 8, 8)
    check_shards(ctx, layer, 2)
    check_shards(ctx, layer, 3)
    check_tp(ctx, layer, 2, ctx.randn(5, 6))


# Rejects reduce_per_row and anything that flattens the batch axes.
@case("a 3-D input on three devices; a single vector")
def batch_axes(ctx):
    layer = make_layer(ctx, 6, 16, 4)
    check_shards(ctx, layer, 3)
    check_tp(ctx, layer, 3, ctx.randn(2, 5, 6))
    check_tp(ctx, layer, 3, ctx.randn(6))


@case("one hidden unit per device, and n that does not divide h")
def extreme_splits(ctx):
    layer = make_layer(ctx, 6, 8, 8)
    check_shards(ctx, layer, 8)
    check_tp(ctx, layer, 8, ctx.randn(5, 6))
    layer = make_layer(ctx, 6, 16, 4)
    for n in (5, 7, 16):
        shards = check_shards(ctx, layer, n)
        check.equal(sum(s["up"].shape[0] for s in shards), 16, label="hidden units in all shards")
    check_tp(ctx, layer, 5, ctx.randn(5, 6))


@case("random layers and inputs on two to four devices, with saturated activations")
def random_layers(ctx):
    for trial in range(6):
        n = (2, 3, 4)[trial % 3]
        layer = make_layer(ctx, 6, 16, 4, bias_scale=float(10.0 ** ctx.rng.uniform(-1, 1)))
        x = ctx.randn(5, 6, scale=float(10.0 ** ctx.rng.uniform(-1, 1.2)))
        check_tp(ctx, layer, n, x)
