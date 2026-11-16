"""Tests for mlx-fast-attention.

The numbers are compared with scaled dot-product attention computed here in float64
NumPy, from the definition: scores, an explicit mask over the buffer's slots, softmax,
weighted sum. Unused slots of a buffer are filled with large values, so a solution
that reads them is far off.

Every call also checks that the solution really used the kernel: a `sys.monitoring`
hook (the mechanism of the kit's `EvalSpy`) counts the calls of
`mx.fast.scaled_dot_product_attention` made from the solution file, whatever name the
solution imported it under. Exactly one call per `attend` is required.
"""

import math
import sys

import mlx.core as mx
import numpy as np

from learnai.judge import case, check

FN = "attend"
KERNEL = mx.fast.scaled_dot_product_attention


class KernelCalls:
    """Counts calls of the fused attention kernel made by code in `file`."""

    def __init__(self, file):
        self.file = str(file)
        self.count = 0
        self._tool = None

    def __enter__(self):
        monitoring = sys.monitoring
        for tool in (4, 3, 2, 1):
            try:
                monitoring.use_tool_id(tool, "learnai-kernel-calls")
            except ValueError:
                continue
            self._tool = tool
            break
        else:
            raise RuntimeError("KernelCalls: no free sys.monitoring tool id")
        monitoring.register_callback(self._tool, monitoring.events.CALL, self._on_call)
        monitoring.set_events(self._tool, monitoring.events.CALL)
        monitoring.restart_events()
        return self

    def __exit__(self, *exc):
        monitoring = sys.monitoring
        monitoring.set_events(self._tool, 0)
        monitoring.register_callback(self._tool, monitoring.events.CALL, None)
        monitoring.free_tool_id(self._tool)
        self._tool = None

    def _on_call(self, code, offset, function, arg0):
        if code.co_filename != self.file:
            return sys.monitoring.DISABLE
        if function is KERNEL:
            self.count += 1
        return None


def oracle(q, k, v, n_cached, causal):
    """Attention from the definition, in float64. Shapes as in the statement."""
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    steps, slots = q.shape[2], k.shape[2]
    groups = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
    last = n_cached + np.arange(steps)[:, None] if causal else n_cached + steps - 1
    visible = np.arange(slots)[None, :] <= last
    scores = np.where(visible, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def make(ctx, batch, q_heads, kv_heads, steps, slots, dim, n_cached, dtype=np.float32):
    """Random q, k, v. Slots after n_cached + steps hold large values, not zeros."""
    q = ctx.randn(batch, q_heads, steps, dim)
    k = ctx.randn(batch, kv_heads, slots, dim)
    v = ctx.randn(batch, kv_heads, slots, dim)
    used = n_cached + steps
    if used > slots:
        raise RuntimeError("authoring error: the buffer is too small")
    k[:, :, used:] = ctx.randn(batch, kv_heads, slots - used, dim, scale=6.0)
    v[:, :, used:] = ctx.randn(batch, kv_heads, slots - used, dim, loc=40.0, scale=10.0)
    return tuple(a.astype(dtype) for a in (q, k, v))


def run(ctx, q, k, v, n_cached, causal):
    """Call the solution; check the result, then that the kernel was called once."""
    dtype = q.dtype
    arrays = [mx.array(a) for a in (q, k, v)]
    with KernelCalls(ctx.solution.__file__) as kernel:
        out = ctx.solution.attend(*arrays, n_cached, causal)
    check.that(isinstance(out, mx.array), f"{FN}: must return an mx.array")
    got_dtype = str(out.dtype).rsplit(".", 1)[-1]
    check.that(
        got_dtype == dtype.name,
        f"{FN}: got dtype {got_dtype}, which is not the expected dtype",
        f"the result must have the dtype of q ({dtype.name})",
    )
    half = dtype == np.float16
    check.allclose(
        np.array(out).astype(np.float32),
        oracle(q, k, v, n_cached, causal).astype(np.float32),
        rtol=1e-2 if half else 1e-4,
        atol=2e-2 if half else 1e-5,
        label=FN,
    )
    check.that(
        kernel.count == 1,
        f"{FN}: called mx.fast.scaled_dot_product_attention {kernel.count} time(s); "
        "exactly one call is required",
        "compute the attention with one call of the fused kernel for the whole batch "
        "and all heads, not with matmul and softmax",
    )


@case("the worked example: two new tokens after one cached token", public=True)
def worked_example(ctx):
    q = np.array([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]], dtype=np.float32)
    expected = oracle(q, k, v, 1, True)
    if not np.allclose(expected, [[[[0.6698, 0.3302], [1.0000, 1.3374]]]], atol=5e-5):
        raise RuntimeError("authoring error: the worked example changed")
    run(ctx, q, k, v, 1, True)
    # The same tokens in a buffer of five slots: the two unused slots change nothing.
    spare_k = np.array([[[[9.0, 9.0], [-7.0, 3.0]]]], dtype=np.float32)
    spare_v = np.array([[[[50.0, -50.0], [80.0, 80.0]]]], dtype=np.float32)
    run(ctx, q, np.concatenate([k, spare_k], axis=2), np.concatenate([v, spare_v], axis=2), 1, True)


@case("a square prefill: 6 tokens, no cache, 4 heads", public=True)
def square_prefill(ctx):
    run(ctx, *make(ctx, 2, 4, 4, 6, 6, 8, 0), 0, True)


@case("a full buffer: 3 new tokens after 5 cached ones, then a single-token step", public=True)
def full_buffer(ctx):
    run(ctx, *make(ctx, 2, 4, 4, 3, 8, 8, 5), 5, True)
    run(ctx, *make(ctx, 2, 4, 4, 1, 9, 8, 8), 8, True)


@case("unused capacity: a 16-slot buffer holding 5 cached and 3 new tokens", public=True)
def unused_capacity(ctx):
    # Rejects causal_string_always: "causal" would align the last query with slot 15.
    run(ctx, *make(ctx, 2, 4, 4, 3, 16, 8, 5), 5, True)


@case("causal = False: every new token reads all valid slots, and no unused one")
def not_causal(ctx):
    run(ctx, *make(ctx, 2, 2, 2, 4, 4, 8, 0), 0, False)
    run(ctx, *make(ctx, 1, 2, 2, 3, 7, 8, 4), 4, False)
    # Rejects no_mask_when_not_causal.
    run(ctx, *make(ctx, 2, 2, 2, 3, 12, 8, 4), 4, False)


@case("grouped-query attention: 8 query heads share 2 key/value heads")
def grouped_query(ctx):
    # Rejects tiles_kv_heads: query head h belongs to key/value head h // 4.
    run(ctx, *make(ctx, 2, 8, 2, 5, 5, 8, 0), 0, True)
    run(ctx, *make(ctx, 1, 8, 2, 2, 12, 8, 6), 6, True)
    run(ctx, *make(ctx, 1, 6, 1, 3, 3, 4, 0), 0, True)


@case("float16 queries, keys, and values, with and without unused capacity")
def half_precision(ctx):
    # Rejects float32_additive_mask: the kernel refuses a float32 mask for float16 inputs.
    run(ctx, *make(ctx, 2, 4, 4, 4, 10, 8, 6, np.float16), 6, True)
    run(ctx, *make(ctx, 2, 4, 2, 3, 16, 8, 5, np.float16), 5, True)
    run(ctx, *make(ctx, 1, 4, 4, 2, 9, 8, 3, np.float16), 3, False)


@case("a single-token step in a buffer with unused capacity")
def single_token_spare(ctx):
    run(ctx, *make(ctx, 2, 4, 4, 1, 32, 8, 9), 9, True)
    run(ctx, *make(ctx, 2, 4, 2, 1, 32, 8, 9), 9, False)
    run(ctx, *make(ctx, 1, 2, 2, 1, 8, 8, 0), 0, True)  # the first token of a sequence


@case("head dimension 64, where MLX has a fused Metal kernel: 16 new tokens, 48 slots")
def head_dim_64(ctx):
    run(ctx, *make(ctx, 1, 4, 4, 16, 48, 64, 20), 20, True)
    run(ctx, *make(ctx, 1, 4, 2, 16, 36, 64, 20), 20, True)
    run(ctx, *make(ctx, 2, 4, 4, 24, 24, 64, 0), 0, True)


@case("an empty cache in a buffer with capacity, and other layouts")
def layouts(ctx):
    # Rejects ignores_n_cached only where n_cached > 0; with n_cached = 0 and capacity,
    # an explicit mask is still needed.
    for batch, heads, steps, slots, n_cached, causal in [
        (1, 2, 4, 10, 0, True),
        (3, 1, 2, 5, 2, True),
        (2, 2, 7, 20, 13, True),
        (1, 4, 5, 6, 0, False),
        (2, 2, 2, 3, 1, True),
    ]:
        run(ctx, *make(ctx, batch, heads, heads, steps, slots, 4, n_cached), n_cached, causal)
