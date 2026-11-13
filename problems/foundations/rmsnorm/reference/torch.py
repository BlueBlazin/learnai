import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Divide each vector by s = max(1, max|x|) before squaring. The squares are then at
    # most 1, so they cannot overflow, and small vectors (s = 1) are left untouched.
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=-1, keepdim=True)
    # x / sqrt(mean(x^2) + eps) == u / sqrt(mean(u^2) + eps / s^2). For huge s the term
    # eps / s^2 rounds to 0, which is exact to float32 precision: mean(u^2) >= 1 / d.
    return u / torch.sqrt(mean_sq + eps / (s * s)) * weight
