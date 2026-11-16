"""Tests for cuda-sdpa-attention.

The numbers are compared with attention computed here in float64 NumPy from the
definition: scores, an explicit visibility mask, softmax, weighted sum. Padding slots
hold large keys and values, so a solution that attends to them is far off.

Every call of `attend` also runs under `KernelSpy`, which replaces
`torch.nn.functional.scaled_dot_product_attention` (under every name the solution
holds it) by a recorder that passes the call on. From the recorded arguments the tests
check the rules of the statement, which no CPU result could show:

- the kernel is called exactly once;
- a square prefill without padding uses `is_causal=True` and no mask;
- every other layout passes a boolean `attn_mask` and leaves `is_causal` off (a
  single new token without padding may also pass no mask: every slot is visible);
- the keys and values reach the kernel with their N_kv heads (`enable_gqa=True`).

`DeviceSpy` checks that every factory call in the solution passes `device=`, and one
case runs `attend` on `torch.device("meta")`, where a mask left on the CPU raises as
it would next to CUDA tensors.

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from learnai.judge import case, check
from learnai.kit import DeviceSpy

FN = "attend"
KERNEL = F.scaled_dot_product_attention
KERNEL_ARGS = (
    "query",
    "key",
    "value",
    "attn_mask",
    "dropout_p",
    "is_causal",
    "scale",
    "enable_gqa",
)


class KernelSpy:
    """Records the calls of the fused attention kernel made while the block runs."""

    def __init__(self, solution):
        self.solution = solution
        self.calls = []
        self._undo = []

    def _recorder(self, *args, **kwargs):
        self.calls.append({**dict(zip(KERNEL_ARGS, args, strict=False)), **kwargs})
        return KERNEL(*args, **kwargs)

    def __enter__(self):
        holders = [
            (F, "scaled_dot_product_attention"),
            (torch._C._nn, "scaled_dot_product_attention"),
        ]
        # `from torch.nn.functional import scaled_dot_product_attention [as name]`
        holders += [(self.solution, name) for name, v in vars(self.solution).items() if v is KERNEL]
        for owner, name in holders:
            self._undo.append((owner, name, getattr(owner, name)))
            setattr(owner, name, self._recorder)
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self._undo):
            setattr(owner, name, original)
        self._undo.clear()


def oracle(q, k, v, n_cached, key_padding):
    """Attention from the definition, in float64. Shapes as in the statement."""
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    steps, slots = q.shape[2], k.shape[2]
    groups = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
    visible = np.arange(slots)[None, :] <= n_cached + np.arange(steps)[:, None]
    visible = np.broadcast_to(visible, (q.shape[0], 1, steps, slots))
    if key_padding is not None:
        visible = visible & ~key_padding[:, None, None, :]
    if not visible.any(axis=-1).all():
        raise RuntimeError("authoring error: a query has no visible slot")
    scores = np.where(visible, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def make(ctx, batch, q_heads, kv_heads, steps, n_cached, dim, pad=None, dtype=np.float32):
    """Random q, k, v and a padding mask. `pad` lists the padded slots of each row."""
    slots = n_cached + steps
    q = ctx.randn(batch, q_heads, steps, dim)
    k = ctx.randn(batch, kv_heads, slots, dim)
    v = ctx.randn(batch, kv_heads, slots, dim)
    key_padding = None
    if pad is not None:
        key_padding = np.zeros((batch, slots), dtype=bool)
        for row, padded in enumerate(pad):
            key_padding[row, list(padded)] = True
        # Padding holds loud numbers: attending to it cannot go unnoticed.
        k[key_padding.nonzero()[0], :, key_padding.nonzero()[1]] *= 4.0
        v[key_padding.nonzero()[0], :, key_padding.nonzero()[1]] += 30.0
    return q.astype(dtype), k.astype(dtype), v.astype(dtype), n_cached, key_padding


def check_output(out, q, k, v, n_cached, key_padding):
    check.that(
        isinstance(out, torch.Tensor), f"{FN}: returned a {type(out).__name__}, not a tensor"
    )
    check.that(
        tuple(out.shape) == q.shape,
        f"{FN}: got shape {tuple(out.shape)}, which is not the expected shape",
        f"expected the shape of q, {q.shape}",
    )
    got_dtype = str(out.dtype).removeprefix("torch.")
    check.that(
        got_dtype == q.dtype.name,
        f"{FN}: got dtype {got_dtype}, which is not the expected dtype",
        f"the result must have the dtype of q ({q.dtype.name})",
    )
    half = q.dtype == np.float16
    check.allclose(
        out.detach().cpu().float().numpy(),
        oracle(q, k, v, n_cached, key_padding).astype(np.float32),
        rtol=1e-2 if half else 1e-4,
        atol=3e-2 if half else 1e-5,
        label=FN,
    )


def check_kernel_call(spy, q, k, n_cached, key_padding):
    """The rules of the statement, read from the arguments the kernel received."""
    check.that(
        len(spy.calls) == 1,
        f"{FN}: called F.scaled_dot_product_attention {len(spy.calls)} time(s); "
        "exactly one call is required",
        "compute the attention with one call of the fused kernel for the whole batch "
        "and all heads, not with matmul and softmax",
    )
    call = spy.calls[0]
    mask, is_causal = call.get("attn_mask"), bool(call.get("is_causal", False))
    key = call.get("key")
    check.that(
        isinstance(key, torch.Tensor) and key.shape[1] == k.shape[1],
        f"{FN}: the keys reached the kernel with a different number of heads than k has",
        f"k has {k.shape[1]} heads; pass k and v as they are with enable_gqa=True instead "
        "of repeating them once per query head",
    )
    if n_cached == 0 and key_padding is None:
        check.that(
            is_causal and mask is None,
            f"{FN}: a square prefill without padding must use is_causal=True and no attn_mask",
            f"the kernel received is_causal={is_causal} and "
            f"attn_mask={'None' if mask is None else 'a tensor'}",
        )
        return
    check.that(
        not is_causal,
        f"{FN}: is_causal=True was passed for a layout that is not a square prefill "
        "without padding",
    )
    if key_padding is None and q.shape[2] == 1 and mask is None:
        return  # one new token, no padding: every slot is visible, no mask is needed
    check.that(
        isinstance(mask, torch.Tensor) and mask.dtype == torch.bool,
        f"{FN}: this layout needs a boolean attn_mask; the kernel received "
        f"{'none' if mask is None else f'one of dtype {mask.dtype}'}",
        "pass attn_mask as a bool tensor in which True means: may attend",
    )


def run(ctx, q, k, v, n_cached, key_padding, device=None):
    """Call the solution; check the numbers, then how the kernel was called."""
    device = device or torch.device("cpu")
    tensors = [torch.from_numpy(a).to(device) for a in (q, k, v)]
    padding = None if key_padding is None else torch.from_numpy(key_padding).to(device)
    with KernelSpy(ctx.solution) as spy, DeviceSpy() as factories:
        out = ctx.solution.attend(*tensors, n_cached, padding)
    check_output(out, q, k, v, n_cached, key_padding)
    check_kernel_call(spy, q, k, n_cached, key_padding)
    check.that(
        not factories.implicit,
        f"{FN}: {len(factories.implicit)} tensors were created without device= "
        "(they land on the default device, the CPU)",
        factories.describe(),
    )
    return out


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: two new tokens after one cached token, then with padding", public=True)
def worked_example(ctx):
    q = np.array([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]], dtype=np.float32)
    padding = np.array([[False, True, False]])
    plain, padded = oracle(q, k, v, 1, None), oracle(q, k, v, 1, padding)
    if not np.allclose(plain, [[[[0.6698, 0.3302], [1.0000, 1.3374]]]], atol=5e-5):
        raise RuntimeError("authoring error: the worked example changed")
    if not np.allclose(padded, [[[[1.0, 0.0], [1.8044, 1.6088]]]], atol=5e-5):
        raise RuntimeError("authoring error: the worked example with padding changed")
    run(ctx, q, k, v, 1, None)
    run(ctx, q, k, v, 1, padding)


@case("a square prefill without padding: is_causal=True and no mask", public=True)
def square_prefill(ctx):
    # Rejects always_builds_mask and manual_attention.
    run(ctx, *make(ctx, 2, 4, 4, 6, 0, 8))
    run(ctx, *make(ctx, 1, 2, 2, 1, 0, 8))  # the first token of a sequence


@case("new tokens after a cache: 3 after 5, then a single decode step", public=True)
def with_cache(ctx):
    # Rejects is_causal_with_cache (query i would see slots 0..i only) and
    # mask_excludes_self.
    run(ctx, *make(ctx, 2, 4, 4, 3, 5, 8))
    run(ctx, *make(ctx, 2, 4, 4, 1, 8, 8))


@case("a padded batch: prefill with right padding, and a cache with left padding", public=True)
def with_padding(ctx):
    # Rejects padding_ignored_in_prefill, padding_polarity_inverted, mask_and_is_causal.
    run(ctx, *make(ctx, 2, 2, 2, 5, 0, 8, pad=[[], [3, 4]]))
    run(ctx, *make(ctx, 3, 2, 2, 2, 6, 8, pad=[[0, 1, 2], [], [0]]))


@case("grouped-query attention: 8 query heads share 2 key/value heads")
def grouped_query(ctx):
    # Rejects tiles_kv_heads (query head h belongs to key/value head h // 4) and
    # repeats_kv_heads (the kernel must receive the 2 heads).
    run(ctx, *make(ctx, 2, 8, 2, 5, 0, 8))
    run(ctx, *make(ctx, 1, 8, 2, 2, 6, 8))
    run(ctx, *make(ctx, 2, 6, 3, 3, 4, 4, pad=[[1], [0, 2]]))
    run(ctx, *make(ctx, 1, 6, 1, 3, 0, 4))


@case("float16 queries, keys, and values")
def half_precision(ctx):
    run(ctx, *make(ctx, 2, 4, 4, 4, 6, 8, dtype=np.float16))
    run(ctx, *make(ctx, 2, 4, 2, 5, 0, 8, dtype=np.float16))
    run(ctx, *make(ctx, 2, 4, 2, 3, 5, 8, pad=[[0, 1], [4]], dtype=np.float16))


@case("key_padding given but nothing is padded, and padding among the new tokens")
def padding_edges(ctx):
    # A mask of all False is still a mask: this is not the is_causal case.
    run(ctx, *make(ctx, 2, 2, 2, 4, 0, 8, pad=[[], []]))
    run(ctx, *make(ctx, 2, 2, 2, 4, 3, 8, pad=[[], []]))
    # Row 1 is shorter: its last two new slots are padding.
    run(ctx, *make(ctx, 2, 2, 2, 4, 3, 8, pad=[[], [5, 6]]))


@case("the mask is built on the device of q (checked on the data-free meta device)")
def on_meta_device(ctx):
    # Rejects mask_default_device, however the mask was created.
    meta = torch.device("meta")
    q, k, v, n_cached, key_padding = make(ctx, 2, 4, 2, 3, 5, 8, pad=[[0], [2, 3]])
    tensors = [torch.from_numpy(a).to(meta) for a in (q, k, v)]
    for padding in (None, torch.from_numpy(key_padding).to(meta)):
        try:
            out = ctx.solution.attend(*tensors, n_cached, padding)
        except RuntimeError as exc:
            text = str(exc).strip().splitlines()[0]
            if "device" not in text:
                raise
            check.that(
                False,
                f"{FN}: a tensor that is not on the device of q reached an operation: {text}",
                "q, k, v, and key_padding were on torch.device('meta'), which like CUDA "
                "does not mix with CPU tensors. Create the mask with device=q.device.",
            )
        check.that(
            isinstance(out, torch.Tensor) and out.device.type == "meta",
            f"{FN}: the result is not on the device of q",
        )
        check.that(
            tuple(out.shape) == q.shape,
            f"{FN}: got shape {tuple(out.shape)}, which is not the expected shape",
        )


@case("head dimension 64 and longer sequences")
def larger(ctx):
    run(ctx, *make(ctx, 1, 4, 4, 16, 20, 64))
    run(ctx, *make(ctx, 2, 4, 2, 24, 0, 64))
    run(ctx, *make(ctx, 2, 4, 2, 12, 20, 64, pad=[list(range(7)), []]))


@case("many layouts")
def layouts(ctx):
    for batch, q_heads, kv_heads, steps, n_cached, pad in [
        (1, 1, 1, 1, 0, None),
        (3, 1, 1, 2, 3, None),
        (2, 2, 1, 7, 13, None),
        (2, 3, 3, 4, 0, [[3], []]),
        (1, 2, 2, 1, 11, [[0, 1, 2, 3]]),
        (4, 4, 2, 5, 2, [[], [0], [0, 1], [6]]),
    ]:
        run(ctx, *make(ctx, batch, q_heads, kv_heads, steps, n_cached, 8, pad=pad))


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: float32 prefill, cache, padding, and grouped-query layouts", requires="cuda")
def gpu_float32(ctx):
    device = torch.device("cuda")
    for args in [
        make(ctx, 2, 4, 4, 6, 0, 64),
        make(ctx, 2, 4, 4, 3, 5, 64),
        make(ctx, 2, 8, 2, 4, 6, 64, pad=[[0, 1], []]),
        make(ctx, 2, 4, 2, 5, 0, 64, pad=[[], [3, 4]]),
    ]:
        out = run(ctx, *args, device=device)
        check.that(out.is_cuda, f"{FN}: the result is on {out.device}, not on the CUDA device")


@case("on the GPU: the same layouts in float16", requires="cuda")
def gpu_float16(ctx):
    device = torch.device("cuda")
    for args in [
        make(ctx, 2, 4, 4, 6, 0, 64, dtype=np.float16),
        make(ctx, 2, 4, 4, 3, 5, 64, dtype=np.float16),
        make(ctx, 2, 8, 2, 4, 6, 64, pad=[[0, 1], []], dtype=np.float16),
        make(ctx, 2, 4, 2, 5, 0, 64, pad=[[], [3, 4]], dtype=np.float16),
    ]:
        out = run(ctx, *args, device=device)
        check.that(out.is_cuda, f"{FN}: the result is on {out.device}, not on the CUDA device")
