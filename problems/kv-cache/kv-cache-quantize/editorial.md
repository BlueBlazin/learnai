## The idea

Two design choices define a quantized KV cache: which tokens stay in floating point, and which elements share a quantization grid.

**Which tokens stay in floating point.** The newest `keep_recent` tokens remain in `float32`. The next token attends most strongly to its immediate predecessors, and keeping those few vectors exact costs almost nothing while confining quantization error to older tokens. After $S$ tokens have been appended, the last $\min(S, \text{keep\_recent})$ tokens are float and the rest are quantized. Every append therefore follows one rule. Concatenate the old float window (length $S_r$) with the new chunk (length $T$) along the sequence axis, determine how many oldest tokens must leave the window, and quantize that slice:

$$
\text{leaving} = \max(0,\ S_r + T - \text{keep\_recent}).
$$

This single formula handles every case, including chunks longer than `keep_recent` and appends where nothing leaves the window yet.

**What shares a grid.** A quantization grid is defined by a minimum (`bias`) and a step size (`scale`). If one grid spans values of very different magnitudes, the smaller values collapse onto one or two codes. Because keys and values vary sharply across both channels and tokens, each token is split into independent grids of `group_size` channels.

Keeping groups inside a single token makes the cache append-only. Each token's codes depend only on that token's float values, so it is quantized once and never modified again. By contrast, if a grid spanned multiple tokens (such as one range per channel across the sequence), every new token could widen the range and force you to dequantize and requantize old tokens on a coarser grid, compounding rounding error. With per-token groups, old codes never need to be touched.

The half-step error bound follows directly from the grid definition. The group minimum maps to code `0`, the maximum maps to code $L$, and every value between them lies within half a step of a grid point so `clamp` never clips in-range values. Thus $|\hat x - x| \le \text{scale} / 2 = (\max - \min) / (2L)$ per element, giving 1 part in 510 of the group range at 8 bits and 1 part in 30 at 4 bits.

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

A slice of zero tokens passes through `_quantize` cleanly and produces tensors with length `0` along axis `2`, so no separate branch is needed when nothing leaves the window. The JAX and MLX implementations are identical using `concatenate`, `min`, `max`, `clip`, and `astype`.

## Common mistakes

- **Quantizing tokens on arrival.** If tokens are quantized and dequantized before entering the float window, the most recent tokens become lossy and the window provides no benefit.
- **Sharing one range across groups** (per token, per head, or per call). Broadcasting can hide the shape mismatch and produce valid codes, but smaller-magnitude groups collapse onto a few levels. On the model test cases, this increases KL divergence by five to six times.
- **Off-by-one window size.** Writing `n - keep_recent - 1` keeps one extra token in float, while using `>=` instead of `>` in a manual condition quantizes one token too early.
- **Evicting only from the old float window.** Popping tokens from `k_recent` and storing `k_new` directly as the new window works when `T = 1`, but fails on multi-token prefills where `T > keep_recent`.
- **Dividing the range by $2^{\text{bits}}$ instead of $2^{\text{bits}} - 1$.** There are $2^{\text{bits}}$ codes and $2^{\text{bits}} - 1$ intervals between them. Dividing by $2^{\text{bits}}$ wastes the top code or shifts every level.
- **Truncating instead of rounding.** Casting directly to an integer type truncates toward zero. This doubles the maximum error to a full step and introduces downward bias that does not average out.
- **Missing constant-group guard.** When `hi == lo`, `(hi - lo) / L` is `0`, turning `(x - lo) / scale` into `0 / 0 = NaN`. A zero vector or padded token triggers this immediately.
- **Concatenating in the wrong order on read.** The dequantized tokens are older and must come before `k_recent` and `v_recent`.

## Complexity

Appending costs $O(B\,H\,D \cdot \text{leaving})$ for quantization plus the concatenation copies (a production cache writes into a preallocated buffer instead). `read_cache` dequantizes all $S_q$ quantized tokens in $O(B\,H\,S\,D)$ time per step. Because materializing the full float cache every step is expensive, inference engines either dequantize inside the attention kernel or enable KV quantization only when memory bandwidth or capacity is the bottleneck.

Each quantized token and head stores `D` integer codes plus `2 * D / group_size` floats for scales and biases. Here each code uses a full `uint8` byte. When bit-packed with `float16` metadata and `group_size = 64`, an 8-bit cache costs $8 + 32 / 64 = 8.5$ bits per element (versus 16 for `float16`), and a 4-bit cache costs $4.5$ bits per element.

## In the wild

- **mlx-lm** (`mlx_lm/models/cache.py`, version 0.32.2): `QuantizedKVCache.update_and_fetch` calls `mx.quantize(keys, group_size=self.group_size, bits=self.bits)` on incoming keys and values and writes packed codes, scales, and biases into buffers that grow in steps of 256 tokens. Because `mx.quantize` groups along the last axis, its layout matches this problem's per-token channel groups. A float `KVCache` converts via `to_quantized` once its length reaches `quantized_kv_start` (default 5000), after which every incoming token is quantized immediately with no float window.
- **Hugging Face Transformers** (`QuantizedLayer` in `cache_utils.py`, version 5.23.0, crediting "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", arXiv 2402.02750) maintains a full-precision residual window of up to `residual_length` tokens (default 128) alongside the quantized cache. When the residual fills, `update` dequantizes the stored cache, concatenates the residual and new tokens, and requantizes the whole sequence. Its backends quantize per channel (`axis_key`, `axis_value`, `q_group_size`), where requantization can introduce additional rounding error.
- **llama.cpp** selects storage formats for K and V via `-ctk, --cache-type-k` and `-ctv, --cache-type-v`. Supported types include `f32`, `f16` (default), `bf16`, `f8_e4m3`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, and `q5_1`.
