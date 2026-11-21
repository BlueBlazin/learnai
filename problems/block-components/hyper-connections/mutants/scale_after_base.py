# BUG: the scale multiplies raw + base instead of the raw value alone.
import torch

NORM_EPS = 1e-6  # rms_norm_eps of DeepSeek-V4; not the `eps` argument


def hc_weights(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps, n, dim = streams.shape
    # One vector per token: all streams end to end, normalized as a whole, no gain.
    flat = streams.reshape(steps, n * dim)
    flat = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + NORM_EPS)
    # One linear map gives the raw values of all three weights: N, N, and N * N of them.
    raw = flat @ fn.T
    pre = torch.sigmoid((raw[:, :n] + base[:n]) * scale[0]) + eps
    post = 2.0 * torch.sigmoid((raw[:, n : 2 * n] + base[n : 2 * n]) * scale[1])
    logits = ((raw[:, 2 * n :] + base[2 * n :]) * scale[2]).reshape(steps, n, n)
    comb = torch.softmax(logits, dim=-1) + eps

    # Sinkhorn-Knopp: columns once, then (iters - 1) times rows and columns again.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre[:, :, None] * streams).sum(dim=1)
    return collapsed, post, comb


def hc_merge(
    streams: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    # Output stream n = post[n] * y + sum_j comb[j, n] * streams[j]: comb is used transposed.
    return post[:, :, None] * y[:, None, :] + comb.transpose(-1, -2) @ streams
