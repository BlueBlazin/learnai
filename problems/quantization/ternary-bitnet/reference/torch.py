import torch


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The absmean: one scale for the matrix, the mean magnitude of all elements.
    gamma = w.abs().mean()
    # An all-zero matrix gets scale 1, so that nothing divides by zero.
    scale = torch.where(gamma > 0, gamma, torch.ones_like(gamma))
    # round() rounds half to even, so +-0.5 become 0; the clamp folds +-2, +-3, ... to +-1.
    t = torch.clamp(torch.round(w / scale), -1, 1)
    return t.to(torch.int8), scale


def ternary_matmul(x: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # (..., 1, in) against t of shape (out, in): every output row selects its inputs.
    xe = x.unsqueeze(-2)
    zero = torch.zeros_like(xe)
    # Compare with 1 and -1 explicitly: inputs whose weight is 0 are in neither sum.
    plus = torch.where(t == 1, xe, zero).sum(dim=-1)
    minus = torch.where(t == -1, xe, zero).sum(dim=-1)
    # The only multiplication of the layer: one per output.
    return scale * (plus - minus)
