import mlx.core as mx

Quant = tuple[mx.array, mx.array, mx.array]  # (codes, scale, bias)
Cache = tuple[Quant, Quant, mx.array, mx.array]  # (k_quant, v_quant, k_recent, v_recent)


def _quantize(x: mx.array, bits: int, group_size: int) -> Quant:
    b, h, s, d = x.shape
    levels = 2**bits - 1  # the largest code: 2**bits codes have 2**bits - 1 steps between them
    # Every token of every head is cut into runs of group_size channels: (B, H, S, G, group_size).
    groups = x.reshape(b, h, s, d // group_size, group_size)
    lo = groups.min(axis=-1)
    hi = groups.max(axis=-1)
    # A constant group has hi == lo: scale 1 makes the division below 0 / 1.
    scale = mx.where(hi > lo, (hi - lo) / levels, mx.ones_like(hi))
    codes = mx.round((groups - lo[..., None]) / scale[..., None])
    codes = mx.clip(codes, 0, levels).astype(mx.uint8)
    return codes.reshape(b, h, s, d), scale, lo


def _dequantize(quant: Quant) -> mx.array:
    codes, scale, bias = quant
    b, h, s, d = codes.shape
    groups = codes.reshape(b, h, s, scale.shape[-1], d // scale.shape[-1]).astype(mx.float32)
    return (groups * scale[..., None] + bias[..., None]).reshape(b, h, s, d)


def _extend(old: Quant | None, new: Quant) -> Quant:
    if old is None:
        return new
    return tuple(mx.concatenate((a, b), axis=2) for a, b in zip(old, new, strict=True))


def quantized_append(
    cache: Cache | None,
    k_new: mx.array,
    v_new: mx.array,
    bits: int,
    group_size: int,
    keep_recent: int,
) -> Cache:
    if cache is None:
        k_quant = v_quant = None
        k_float, v_float = k_new, v_new
    else:
        k_quant, v_quant, k_recent, v_recent = cache
        k_float = mx.concatenate((k_recent, k_new), axis=2)
        v_float = mx.concatenate((v_recent, v_new), axis=2)
    # Everything but the newest keep_recent tokens leaves the float window now. Those
    # tokens are quantized here, once, from their float values; older codes are not touched.
    leaving = max(0, k_float.shape[2] - keep_recent)
    k_quant = _extend(k_quant, _quantize(k_float[:, :, :leaving], bits, group_size))
    v_quant = _extend(v_quant, _quantize(v_float[:, :, :leaving], bits, group_size))
    return k_quant, v_quant, k_float[:, :, leaving:], v_float[:, :, leaving:]


def read_cache(cache: Cache) -> tuple[mx.array, mx.array]:
    k_quant, v_quant, k_recent, v_recent = cache
    keys = mx.concatenate((_dequantize(k_quant), k_recent), axis=2)
    values = mx.concatenate((_dequantize(v_quant), v_recent), axis=2)
    return keys, values
