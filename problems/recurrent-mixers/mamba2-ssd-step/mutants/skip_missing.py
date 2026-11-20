# BUG: the skip connection D * x is left out of the output.
import torch
import torch.nn.functional as F


def ssd_step(
    h: torch.Tensor,
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    dt_limit: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    # The step size of each head: positive, then clamped to the configured range.
    dt = torch.clamp(F.softplus(dt_raw + dt_bias), min=dt_limit[0], max=dt_limit[1])
    # Group j serves the heads j * r .. j * r + r - 1: repeat, do not tile.
    reps = h.shape[0] // B.shape[0]
    B = B.repeat_interleave(reps, dim=0)  # (H, N)
    C = C.repeat_interleave(reps, dim=0)
    # h <- h * exp(dt * A) + dt * x B^T, with A = -exp(A_log) < 0.
    decay = torch.exp(-dt * torch.exp(A_log))
    h = h * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]
    # Read the updated state, plus the skip connection.
    y = torch.einsum("hpn,hn->hp", h, C)
    return y, h


def ssd_matrix(
    dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor, A_log: torch.Tensor
) -> torch.Tensor:
    steps, heads = dt.shape
    reps = heads // B.shape[1]
    B = B.repeat_interleave(reps, dim=1)  # (T, H, N)
    C = C.repeat_interleave(reps, dim=1)
    scores = torch.einsum("thn,shn->hts", C, B)  # C_t . B_s
    # exp(sum of dt_r * A for s < r <= t) = exp(c_t - c_s), c the running sum of dt * A.
    c = torch.cumsum(-dt * torch.exp(A_log), dim=0).T  # (H, T)
    lower = torch.ones(steps, steps, dtype=torch.bool).tril()
    decay = torch.exp((c[:, :, None] - c[:, None, :]).masked_fill(~lower, -torch.inf))
    return scores * decay * dt.T[:, None, :]  # the input at s enters scaled by dt_s
