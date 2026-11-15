## The idea

Two decisions define a quantized KV cache: **which tokens** are stored as codes, and
**what shares a grid**.

*Which tokens.* The newest `keep_recent` tokens stay float. The next token attends
most strongly to its immediate predecessors, and those few vectors cost almost
nothing to keep exact, so the loss is confined to the long tail of old tokens. The
cache is then determined by the token count alone: with $S$ tokens appended, the
last $\min(S, \text{keep\_recent})$ are float and the rest are codes. An append
therefore has one job. Put the old window and the new chunk in a row, cut off
everything but the newest `keep_recent`, and quantize what was cut off:

$$
\text{leaving} = \max(0,\ S_r + T - \text{keep\_recent}).
$$

No case analysis is needed for "the chunk is longer than the window" or "nothing
leaves yet"; both are values of `leaving`.

*What shares a grid.* A grid is a minimum and a step. If one grid covers values of
very different magnitude, the small ones share one or two codes. Keys and values
vary strongly from channel to channel and from token to token, so each token's
vector gets several grids, one per run of `group_size` channels. Because a group
never spans two tokens, a token's codes depend on that token alone. That is what
makes the cache append-only: a token is quantized once, from its float values, and
never again.

That last property is not automatic. If a group spans tokens (one grid per channel
over the whole sequence, say), every new token can widen the range, and the only way
to keep one grid is to dequantize the old codes and quantize them again on the new,
coarser grid. Each such pass adds rounding error to tokens that were already
rounded. With per-token groups, requantizing a dequantized group would give back the
same codes (its minimum and maximum are grid points), so there is nothing to gain and
nothing to lose, and the right thing is to not touch old codes at all.

The half-step bound follows from the grid: the minimum maps to code 0 and the
maximum to code $L$, every value in between is at most half a step from a grid
point, and `clamp` never acts on in-range values. So
$|\hat x - x| \le \text{scale} / 2 = (\max - \min) / (2L)$ per element: with 8 bits a
group is resolved to 1 part in 510 of its range, with 4 bits to 1 part in 30.

## Reference solution (PyTorch)

```python
def _quantize(x, bits, group_size):
    b, h, s, d = x.shape
    levels = 2**bits - 1
    groups = x.reshape(b, h, s, d // group_size, group_size)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(b, h, s, d), scale, lo


def _dequantize(quant):
    codes, scale, bias = quant
    b, h, s, d = codes.shape
    groups = codes.reshape(b, h, s, scale.shape[-1], d // scale.shape[-1]).to(torch.float32)
    return (groups * scale[..., None] + bias[..., None]).reshape(b, h, s, d)


def _extend(old, new):
    if old is None:
        return new
    return tuple(torch.cat((a, b), dim=2) for a, b in zip(old, new, strict=True))


def quantized_append(cache, k_new, v_new, bits, group_size, keep_recent):
    if cache is None:
        k_quant = v_quant = None
        k_float, v_float = k_new, v_new
    else:
        k_quant, v_quant, k_recent, v_recent = cache
        k_float = torch.cat((k_recent, k_new), dim=2)
        v_float = torch.cat((v_recent, v_new), dim=2)
    leaving = max(0, k_float.shape[2] - keep_recent)
    k_quant = _extend(k_quant, _quantize(k_float[:, :, :leaving], bits, group_size))
    v_quant = _extend(v_quant, _quantize(v_float[:, :, :leaving], bits, group_size))
    return k_quant, v_quant, k_float[:, :, leaving:], v_float[:, :, leaving:]


def read_cache(cache):
    k_quant, v_quant, k_recent, v_recent = cache
    keys = torch.cat((_dequantize(k_quant), k_recent), dim=2)
    values = torch.cat((_dequantize(v_quant), v_recent), dim=2)
    return keys, values
```

A slab of zero tokens goes through `_quantize` like any other and yields tensors with
an axis of length 0, so "nothing leaves the window" needs no branch. The JAX and MLX
versions are the same lines with `concatenate`, `min`, `max`, `clip`, and `astype`.

## Common mistakes

- **Quantizing tokens on arrival.** Then the float window holds values that went
  through the codes and back: the tokens the next step attends to most are the lossy
  ones, and the window buys nothing.
- **One range for everything quantized in a call** (or per token, or per head). The
  shapes can be made to fit by broadcasting, the codes are valid, and small groups
  collapse onto a few codes. On the model cases this is five to six times the
  reference's divergence.
- **`keep_recent` off by one.** `n - keep_recent - 1` keeps one token too many in
  float; `>=` for `>` in a hand-written condition quantizes one too early.
- **Moving only the old window.** Code that pops tokens from the stored window and
  then stores the new chunk as the window is right for one token per call and leaves
  a 16-token prefill entirely in float.
- **Dividing the range by $2^{\text{bits}}$.** There are $2^{\text{bits}}$ codes and
  $2^{\text{bits}} - 1$ steps between them. With the wrong divisor the top code is
  never produced by rounding and the maximum of every group is clamped short.
- **Truncating instead of rounding.** The cast to an integer type truncates. The
  error doubles to a whole step and is biased downward, which a mean over many
  elements does not average away.
- **No guard for a constant group.** `(max - min) / L` is 0, and `0 / 0` is NaN. A
  padded position or a zero vector is enough to trigger it.
- **Reading back out of order.** The dequantized tokens are the old ones and go
  first.

## Complexity

An append costs $O(B\,H\,D \cdot \text{leaving})$ for the quantization plus the
concatenations, which copy the cache as in `kv-cache-append` (a production cache
writes into a preallocated buffer instead). `read_cache` dequantizes all $S_q$
tokens, $O(B\,H\,S\,D)$ per step: that is the price of the format, and the reason
engines dequantize inside the attention kernel or keep the cache quantized only
when memory, not speed, is the constraint.

Memory per quantized token and head is $D$ codes plus $2 D / \text{group\_size}$
floats. Here a code occupies a whole byte; packed, an 8-bit code with float16
metadata and groups of 64 costs $8 + 32/64 = 8.5$ bits per element against 16 for a
float16 cache, and a 4-bit code $4.5$.

## In the wild

- **mlx-lm** (`mlx_lm/models/cache.py`, read in version 0.32.2):
  `QuantizedKVCache.update_and_fetch` calls
  `mx.quantize(keys, group_size=self.group_size, bits=self.bits)` on the incoming
  keys and values and writes the packed codes, scales, and biases into buffers that
  grow in steps of 256 tokens. `mx.quantize` groups along the last axis, so its
  groups are this problem's: runs of channels within one token. A float `KVCache` is
  converted with `to_quantized` once its offset reaches `quantized_kv_start` (5000 by
  default); from then on every token is quantized on arrival, with no float window.
- **Hugging Face Transformers** (`QuantizedLayer` in `cache_utils.py`, read in
  version 5.23.0; the docstring credits the KIVI paper, "KIVI: A Tuning-Free
  Asymmetric 2bit Quantization for KV Cache", arXiv 2402.02750) keeps a
  full-precision residual of up to `residual_length` tokens (128 by default) next to
  the quantized states. When the residual fills, `update` dequantizes the stored
  states, concatenates them with the residual and the new tokens, and quantizes the
  whole sequence again. Its backends quantize per channel (`axis_key`, `axis_value`,
  `q_group_size`), a layout in which that requantization is not free of error, which
  is the contrast drawn in "The idea".
- **llama.cpp** selects the storage type of the K and V tensors with
  `-ctk, --cache-type-k` and `-ctv, --cache-type-v`; the server documentation lists
  `f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1` with `f16` as the
  default.
