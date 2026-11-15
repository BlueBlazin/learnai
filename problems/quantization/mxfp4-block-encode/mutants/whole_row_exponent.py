# BUG: takes the maximum over the whole row instead of over each block of 32, so every block of a row gets the scale of the row's largest block.
import torch

BLOCK = 32  # elements that share one scale
EMAX = 2  # the largest exponent of an E2M1 element: its largest power of two is 2**2
BIAS = 127  # exponent bias of the E8M0 scale, the same as float32's

MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _exponent_field(m: torch.Tensor) -> torch.Tensor:
    return m.view(torch.int32) >> 23


def _power_of_two(field: torch.Tensor) -> torch.Tensor:
    return (field << 23).view(torch.float32)


def mxfp4_encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_dim, in_dim = x.shape
    blocks = x.reshape(out_dim, in_dim // BLOCK, BLOCK)
    largest = blocks.abs().amax(dim=(-2, -1), keepdim=True).squeeze(-1).expand(blocks.shape[:2])
    field = torch.where(largest > 0, _exponent_field(largest) - EMAX, BIAS)
    scale = _power_of_two(field)
    a = (blocks.abs() / scale[..., None]).clamp(max=6.0)
    magnitude = torch.where(
        a < 2.0,
        torch.round(a * 2.0),
        torch.where(a < 4.0, torch.round(a) + 2.0, torch.round(a / 2.0) + 4.0),
    )
    codes = magnitude + 8.0 * (blocks < 0)
    return codes.reshape(out_dim, in_dim).to(torch.uint8), field.to(torch.uint8)


def mxfp4_decode(codes: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.int32)
    magnitude = torch.tensor(MAGNITUDES)[codes & 7]
    sign = 1.0 - 2.0 * (codes >> 3)
    scale = _power_of_two(exps.to(torch.int32))
    return sign * magnitude * scale.repeat_interleave(BLOCK, dim=1)
