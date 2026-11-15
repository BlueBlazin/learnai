# BUG: divides the range by 2**bits (the number of codes) instead of 2**bits - 1 (the number of steps), so the top code is never used and the grid stops short of the maximum.
import torch

Quant = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (codes, scale, bias)
Cache = tuple[Quant, Quant, torch.Tensor, torch.Tensor]  # (k_quant, v_quant, k_recent, v_recent)


def _quantize(x: torch.Tensor, bits: int, group_size: int) -> Quant:
    b, h, s, d = x.shape
    levels = 2**bits - 1  # the largest code: 2**bits codes have 2**bits - 1 steps between them
    # Every token of every head is cut into runs of group_size channels: (B, H, S, G, group_size).
    groups = x.reshape(b, h, s, d // group_size, group_size)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    # A constant group has hi == lo: scale 1 makes the division below 0 / 1.
    scale = torch.where(hi > lo, (hi - lo) / (levels + 1), torch.ones_like(hi))
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(b, h, s, d), scale, lo


def _dequantize(quant: Quant) -> torch.Tensor:
    codes, scale, bias = quant
    b, h, s, d = codes.shape
    groups = codes.reshape(b, h, s, scale.shape[-1], d // scale.shape[-1]).to(torch.float32)
    return (groups * scale[..., None] + bias[..., None]).reshape(b, h, s, d)


def _extend(old: Quant | None, new: Quant) -> Quant:
    if old is None:
        return new
    return tuple(torch.cat((a, b), dim=2) for a, b in zip(old, new, strict=True))


def quantized_append(
    cache: Cache | None,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    bits: int,
    group_size: int,
    keep_recent: int,
) -> Cache:
    if cache is None:
        k_quant = v_quant = None
        k_float, v_float = k_new, v_new
    else:
        k_quant, v_quant, k_recent, v_recent = cache
        k_float = torch.cat((k_recent, k_new), dim=2)
        v_float = torch.cat((v_recent, v_new), dim=2)
    # Everything but the newest keep_recent tokens leaves the float window now. Those
    # tokens are quantized here, once, from their float values; older codes are not touched.
    leaving = max(0, k_float.shape[2] - keep_recent)
    k_quant = _extend(k_quant, _quantize(k_float[:, :, :leaving], bits, group_size))
    v_quant = _extend(v_quant, _quantize(v_float[:, :, :leaving], bits, group_size))
    return k_quant, v_quant, k_float[:, :, leaving:], v_float[:, :, leaving:]


def read_cache(cache: Cache) -> tuple[torch.Tensor, torch.Tensor]:
    k_quant, v_quant, k_recent, v_recent = cache
    keys = torch.cat((_dequantize(k_quant), k_recent), dim=2)
    values = torch.cat((_dequantize(v_quant), v_recent), dim=2)
    return keys, values
