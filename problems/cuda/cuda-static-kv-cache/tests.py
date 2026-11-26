"""Tests for cuda-static-kv-cache.

A case plays a request through the cache: a prefill block and decode steps, layer by
layer. At every write it checks, on the CPU:

- **constant shapes**: the tensors `update` returns are always `(B, H, max_len, D)` and
  the mask is always `(n_new, max_len)`;
- **constant memory**: `data_ptr()` of every buffer is what it was after construction,
  and a reference to the buffer taken before the first write shows every later write
  (a rebound or reallocated buffer fails both);
- **the contents**: slots `pos .. pos + T - 1` hold what was written and every other
  slot is untouched;
- **the mask**, exactly; and **attention over the full buffers with that mask** against
  attention over a cache grown with `torch.cat`. Before the first write the judge
  fills every buffer with large finite garbage, so a mask that lets one unused slot
  through changes the result by orders of magnitude. The garbage is finite on
  purpose: a NaN in an unused *value* slot would poison any attention (0 * NaN), and a
  NaN key survives an additive mask (NaN - inf), so no mask could be correct;
- **no host reads and no CPU tensors**: the whole request runs under `SyncSpy` and
  `DeviceSpy`, and one case runs it on `torch.device("meta")`, where a tensor used as
  a slice bound (which `SyncSpy` cannot see) raises.

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import math

import numpy as np
import torch

from learnai.judge import case, check
from learnai.kit import DeviceSpy, SyncSpy

CPU = torch.device("cpu")
META = torch.device("meta")
NAME = "StaticCache"


def attend(q, k_buf, v_buf, mask):
    """Scaled dot-product attention; `mask` is bool, True where a query may attend."""
    q, k_buf, v_buf = q.float(), k_buf.float(), v_buf.float()
    scores = q @ k_buf.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v_buf


def visible(pos, n_new, width):
    """The mask of the statement, in NumPy: slot j <= pos + i."""
    return np.arange(width)[None, :] <= (pos + np.arange(n_new))[:, None]


def as_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


class Request:
    """Builds a cache and plays writes through it, checking every step."""

    def __init__(self, ctx, n_layers, batch, heads, max_len, dim, dtype, device=CPU, rng=None):
        self.ctx, self.rng = ctx, rng if rng is not None else ctx.rng
        self.n_layers, self.max_len, self.dtype, self.device = n_layers, max_len, dtype, device
        self.shape = (batch, heads, max_len, dim)
        self.cache = ctx.solution.StaticCache(n_layers, batch, heads, max_len, dim, device, dtype)
        self.buffers = self.check_buffers()
        self.pointers = [buffer.data_ptr() for buffer in self.buffers]
        self.expected = [torch.zeros(self.shape, dtype=dtype, device=device) for _ in self.buffers]
        self.start()

    # -- helpers ---------------------------------------------------------------------

    def random(self, *shape, scale=1.0):
        values = self.rng.standard_normal(shape).astype(np.float32) * scale
        return torch.from_numpy(values).to(device=self.device, dtype=self.dtype)

    def check_buffers(self):
        """cache.k and cache.v: n_layers tensors each, right shape, dtype, device; all distinct."""
        cache, buffers = self.cache, []
        for attr in ("k", "v"):
            value = getattr(cache, attr, None)
            check.that(
                isinstance(value, list)
                and len(value) == self.n_layers
                and all(isinstance(t, torch.Tensor) for t in value),
                f"{NAME}: `{attr}` must be a list with one tensor per layer",
            )
            for layer, tensor in enumerate(value):
                label = f"{attr}[{layer}]"
                check.that(
                    tuple(tensor.shape) == self.shape,
                    f"{label}: got shape {tuple(tensor.shape)}, expected shape {self.shape}",
                    f"expected (batch, n_kv_heads, max_len, head_dim) = {self.shape}",
                )
                check.that(
                    tensor.dtype == self.dtype,
                    f"{label}: got dtype {dtype_name(tensor.dtype)}, expected dtype "
                    f"{dtype_name(self.dtype)}",
                )
                check.that(
                    tensor.device.type == self.device.type,
                    f"{label}: the buffer is on {tensor.device.type}, not on the device",
                )
                buffers.append(tensor)
        if self.device.type != "meta":
            check.that(
                len({buffer.data_ptr() for buffer in buffers}) == len(buffers),
                f"{NAME}: two of the 2 * n_layers buffers are the same memory",
                "every layer needs a key buffer and a value buffer of its own",
            )
            for buffer in buffers:
                check.that(
                    not bool(buffer.count_nonzero()),
                    f"{NAME}: a buffer is not zero-filled after construction or reset",
                )
        return buffers

    def start(self, garbage=True):
        """Begin a request at slot 0; optionally fill every unused slot with garbage."""
        self.length = 0
        self.grown = [None] * (2 * self.n_layers)
        for buffer, expected in zip(self.buffers, self.expected, strict=True):
            expected.zero_()
            if garbage:
                expected.copy_(self.random(*self.shape, scale=30.0))
            buffer.copy_(expected)  # the judge writes into the solution's buffer

    def unchanged_memory(self, when):
        current = [*self.cache.k, *self.cache.v]
        check.that(
            len(current) == len(self.buffers)
            and all(now.data_ptr() == ptr for now, ptr in zip(current, self.pointers, strict=True)),
            f"{NAME}: a buffer was reallocated or rebound {when}; data_ptr() changed",
            "the buffers allocated by the constructor must be written in place and kept",
        )

    # -- one write -------------------------------------------------------------------

    def write(self, n_new):
        """Write a block of n_new tokens into every layer and check everything."""
        pos, end = self.length, self.length + n_new
        batch, heads, max_len, dim = self.shape
        want_mask = visible(pos, n_new, max_len)
        for layer in range(self.n_layers):
            k_new = self.random(batch, heads, n_new, dim)
            v_new = self.random(batch, heads, n_new, dim)
            q = self.random(batch, heads, n_new, dim)
            pos_tensor = torch.tensor(pos, device=self.device)
            out = self.cache.update(layer, k_new, v_new, pos_tensor)
            mask = self.cache.mask(pos_tensor, n_new)

            check.that(
                isinstance(out, tuple)
                and len(out) == 2
                and all(isinstance(t, torch.Tensor) for t in out),
                "update: must return a tuple (k, v) of two tensors",
            )
            for label, tensor in zip(("update k", "update v"), out, strict=True):
                check.that(
                    tuple(tensor.shape) == self.shape,
                    f"{label}: got shape {tuple(tensor.shape)}, expected shape {self.shape}",
                    f"expected the full buffer, {self.shape}, at every step",
                )
            self.unchanged_memory("by update")
            k_index, v_index = layer, self.n_layers + layer
            for label, tensor, index in (("k", out[0], k_index), ("v", out[1], v_index)):
                check.that(
                    tensor.data_ptr() == self.pointers[index],
                    f"update: the returned {label} is not the buffer of layer {layer}",
                )

            check.that(isinstance(mask, torch.Tensor), "mask: must return a tensor")
            check.that(
                mask.dtype == torch.bool and tuple(mask.shape) == (n_new, max_len),
                f"mask: got dtype {dtype_name(mask.dtype)} and shape {tuple(mask.shape)}, "
                f"expected bool and shape {(n_new, max_len)}",
                f"expected a bool tensor of shape (n_new, max_len) = {(n_new, max_len)}",
            )
            check.that(
                mask.device.type == self.device.type,
                f"mask: the mask is on {mask.device.type}, not on the device",
            )
            check.equal(mask.cpu().numpy(), want_mask, label="mask")

            # Contents, read through the references taken before the first write.
            self.expected[k_index][:, :, pos:end] = k_new
            self.expected[v_index][:, :, pos:end] = v_new
            for label, index in ((f"k[{layer}]", k_index), (f"v[{layer}]", v_index)):
                check.allclose(
                    as_numpy(self.buffers[index]),
                    as_numpy(self.expected[index]),
                    rtol=0.0,
                    atol=0.0,
                    label=f"buffer {label} after the write",
                )

            # Attention over the full buffers equals attention over a grown cache.
            for index, new in ((k_index, k_new), (v_index, v_new)):
                old = self.grown[index]
                self.grown[index] = new if old is None else torch.cat([old, new], dim=2)
            causal = torch.from_numpy(visible(pos, n_new, end)).to(self.device)
            want = attend(q, self.grown[k_index], self.grown[v_index], causal)
            got = attend(q, out[0], out[1], mask)
            check.allclose(
                as_numpy(got),
                as_numpy(want),
                rtol=1e-4,
                atol=1e-5,
                label="attention over the static cache",
            )
        self.length = end

    def reset(self):
        result = self.cache.reset()
        check.that(result is None, "reset: must return None")
        self.unchanged_memory("by reset")
        self.check_buffers()
        for buffer in self.buffers:
            check.that(
                not bool(buffer.count_nonzero()),
                "reset: the buffers allocated by the constructor were not zeroed in place",
            )


def play(
    ctx, blocks, *, n_layers=2, batch=2, heads=2, max_len=12, dim=4, dtype=torch.float32, **kw
):
    """One request under the spies: construct, write `blocks`, report host reads."""
    with DeviceSpy() as factories, SyncSpy() as syncs:
        request = Request(ctx, n_layers, batch, heads, max_len, dim, dtype, **kw)
        for n_new in blocks:
            request.write(n_new)
    check_spies(factories, syncs)
    return request


def check_spies(factories, syncs):
    check.that(
        not factories.implicit,
        f"{NAME}: {len(factories.implicit)} tensors were created without device= "
        "(they land on the default device, the CPU)",
        factories.describe(),
    )
    check.that(
        syncs.syncs == 0,
        f"{NAME}: {syncs.syncs} host synchronizations (a value was read on the host)",
        syncs.describe(),
    )


def solution_line(ctx, exc):
    """' (line N of your file)' for the innermost frame of the solution on the traceback."""
    line, tb = None, exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == ctx.solution.__file__:
            line = tb.tb_lineno
        tb = tb.tb_next
    return f" (line {line} of your file)" if line is not None else ""


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: three tokens in a cache of four slots", public=True)
def worked_example(ctx):
    def rows(values):
        return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 2)

    q = rows([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    k = rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with DeviceSpy() as factories, SyncSpy() as syncs:
        cache = ctx.solution.StaticCache(1, 1, 1, 4, 2, CPU, torch.float32)
        k_buf, v_buf = cache.update(0, k[:, :, :2], v[:, :, :2], torch.tensor(0))
        mask = cache.mask(torch.tensor(0), 2)
        check.equal(tuple(k_buf.shape), (1, 1, 4, 2), label="shape of the returned k")
        check.equal(mask.numpy(), np.array([[1, 0, 0, 0], [1, 1, 0, 0]], bool), label="mask")
        check.allclose(
            k_buf.numpy().reshape(4, 2), [[1, 0], [0, 1], [0, 0], [0, 0]], label="k after 2 tokens"
        )
        check.allclose(
            attend(q[:, :, :2], k_buf, v_buf, mask).numpy().reshape(2, 2),
            [[1.0, 2.0], [2.608860, 3.608860]],
            atol=1e-5,
            label="attention after the prefill",
        )
        pointer = k_buf.data_ptr()
        k_buf2, v_buf2 = cache.update(0, k[:, :, 2:], v[:, :, 2:], torch.tensor(2))
        mask = cache.mask(torch.tensor(2), 1)
        check.equal(mask.numpy(), np.array([[1, 1, 1, 0]], bool), label="mask")
        check.that(
            k_buf2.data_ptr() == pointer and tuple(k_buf2.shape) == (1, 1, 4, 2),
            "update: the second call returned a different buffer, or a different shape",
        )
        check.allclose(
            k_buf.numpy().reshape(4, 2),
            [[1, 0], [0, 1], [1, 1], [0, 0]],
            label="the k returned by the first call, after the second",
        )
        check.allclose(
            attend(q[:, :, 2:], k_buf2, v_buf2, mask).numpy().reshape(1, 2),
            [[3.510469, 4.510469]],
            atol=1e-5,
            label="attention after the decode step",
        )
    check_spies(factories, syncs)


@case("a prefill of four tokens, then six decode steps, two layers", public=True)
def prefill_and_decode(ctx):
    # Rejects grows_with_cat, rebinds_buffers, returns_filled_prefix (shapes, memory),
    # the four mask mutants, shared_layer_buffers, host_read_position, and
    # arange_default_device.
    play(ctx, [4, 1, 1, 1, 1, 1, 1])


@case("reset zeroes the same buffers, and a second request reuses them", public=True)
def reset_and_reuse(ctx):
    # Rejects reset_reallocates and reset_does_nothing.
    with DeviceSpy() as factories, SyncSpy() as syncs:
        request = Request(ctx, 2, 1, 2, 10, 4, torch.float32)
        for n_new in (3, 1, 1):
            request.write(n_new)
        request.reset()
        request.start(garbage=False)
        for n_new in (2, 1, 1, 1):
            request.write(n_new)
        request.reset()
        request.start()
        request.write(5)
    check_spies(factories, syncs)


@case("half-precision caches: the buffers have the requested dtype")
def half_precision(ctx):
    # Rejects ignores_dtype.
    play(ctx, [3, 1, 1], dtype=torch.float16)
    play(ctx, [2, 1], dtype=torch.bfloat16, n_layers=1)


@case("the cache is filled to its last slot, by decode steps and by one block")
def filled_to_capacity(ctx):
    play(ctx, [5, 1, 1, 1], max_len=8)
    play(ctx, [6], max_len=6, n_layers=1)
    play(ctx, [1] * 5, max_len=5, batch=1, heads=1)


@case("prefill blocks of several tokens: each query sees up to its own slot")
def multi_token_blocks(ctx):
    # Rejects mask_not_causal_in_block: within a block, later tokens are not visible.
    play(ctx, [3, 4, 2, 1], max_len=12)
    play(ctx, [7, 5], max_len=16, n_layers=3, batch=1)


@case("one layer, one head, one row; and four layers")
def edge_sizes(ctx):
    play(ctx, [1, 1, 1], n_layers=1, batch=1, heads=1, max_len=4, dim=2)
    play(ctx, [2, 1, 1], n_layers=4, batch=3, heads=1, max_len=6, dim=8)


@case("the same request on a data-free device: nothing is read on the host")
def on_meta_device(ctx):
    # Rejects tensor_slice_bound, which SyncSpy cannot see: a tensor used as a slice
    # bound is read in C++. On the meta device there is nothing to read, so it raises.
    try:
        cache = ctx.solution.StaticCache(2, 1, 2, 16, 4, META, torch.float32)
        pos = torch.tensor(3, device=META)
        new = torch.zeros(1, 2, 2, 4, device=META)
        k_buf, v_buf = cache.update(1, new, new, pos)
        mask = cache.mask(pos, 2)
        cache.reset()
    except (RuntimeError, NotImplementedError) as exc:
        text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        if "device" not in text and "meta tensor" not in text:
            raise
        line = solution_line(ctx, exc)
        check.that(
            False,
            f"{NAME}: a value was read on the host, or a tensor was not on `device`{line}: {text}",
            "The cache, the new keys and values, and `pos` were on torch.device('meta'), "
            "whose tensors have a shape and no data. int(pos), pos.item(), and a tensor "
            "used as a slice bound all read the value of `pos`.",
        )
    for label, tensor, shape in (
        ("update k", k_buf, (1, 2, 16, 4)),
        ("update v", v_buf, (1, 2, 16, 4)),
        ("mask", mask, (2, 16)),
    ):
        check.that(
            tensor.device.type == "meta" and tuple(tensor.shape) == shape,
            f"{label}: got shape {tuple(tensor.shape)} on {tensor.device.type}, expected "
            f"shape {shape} on meta",
        )


@case("random requests: shapes, block sizes, and lengths")
def random_requests(ctx):
    configs = [(1, 1, 2, 8, 4), (2, 2, 2, 12, 4), (3, 1, 4, 16, 8)]

    def prop(rng):
        n_layers, batch, heads, max_len, dim = configs[int(rng.integers(0, len(configs)))]
        blocks, total = [int(rng.integers(1, 6))], 0
        total = blocks[0]
        target = int(rng.integers(total, max_len + 1))
        while total < target:
            step = 1 if rng.random() < 0.7 else int(rng.integers(1, 4))
            step = min(step, target - total)
            blocks.append(step)
            total += step
        play(
            ctx,
            blocks,
            n_layers=n_layers,
            batch=batch,
            heads=heads,
            max_len=max_len,
            dim=dim,
            rng=rng,
        )

    check.forall(prop, trials=10)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: a request on the CUDA device, float32 and float16", requires="cuda")
def gpu_request(ctx):
    device = torch.device("cuda")
    play(ctx, [4, 1, 1, 1, 1], device=device)
    play(ctx, [3, 1, 1], device=device, dtype=torch.float16)


@case("on the GPU: 256 decode steps do not grow the reserved memory", requires="cuda")
def gpu_no_growth(ctx):
    device = torch.device("cuda")
    n_layers, batch, heads, max_len, dim = 2, 1, 8, 512, 128
    cache = ctx.solution.StaticCache(n_layers, batch, heads, max_len, dim, device, torch.float16)
    new = torch.randn(batch, heads, 1, dim, device=device, dtype=torch.float16)
    positions = [torch.tensor(p, device=device) for p in range(8 + 256)]
    pointers = [buffer.data_ptr() for buffer in (*cache.k, *cache.v)]

    def step(pos):
        for layer in range(n_layers):
            k_buf, v_buf = cache.update(layer, new, new, pos)
            mask = cache.mask(pos, 1)
        return k_buf, v_buf, mask

    for pos in positions[:8]:  # warm-up: the allocator's pools reach their steady state
        step(pos)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_reserved(device)
    for pos in positions[8:]:
        k_buf, _, mask = step(pos)
    torch.cuda.synchronize(device)
    growth = torch.cuda.memory_reserved(device) - before
    ctx.metric("reserved_growth", growth, "bytes")
    check.that(
        tuple(k_buf.shape) == (batch, heads, max_len, dim) and tuple(mask.shape) == (1, max_len),
        f"{NAME}: the shapes after 264 steps are not the shapes of the first step",
    )
    check.that(
        [buffer.data_ptr() for buffer in (*cache.k, *cache.v)] == pointers,
        f"{NAME}: a buffer was reallocated during the decode steps",
    )
    check.that(
        growth <= 2 * 2**20,
        f"{NAME}: reserved GPU memory grew by {growth} bytes over 256 decode steps",
        "a static cache allocates nothing per step beyond a few small temporaries; the "
        "limit is 2 MiB, one segment of the allocator's small pool",
    )
