import torch


def _quantize_groups(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    # Group-wise affine quantization followed by dequantization (as in groupwise-4bit):
    # every run of group_size inputs in a row gets its own minimum and step.
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1, keepdim=True)
    hi = groups.amax(dim=-1, keepdim=True)
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.clamp(torch.round((groups - lo) / scale), 0, levels)
    return (codes * scale + lo).reshape(out_dim, in_dim)


def awq_scales(
    w: torch.Tensor, x_calib: torch.Tensor, bits: int, group_size: int, grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # How large each input channel is on the calibration data: the mean magnitude.
    act = x_calib.abs().mean(dim=0)
    target = x_calib @ w.T  # what the layer computes before quantization
    candidates, losses = [], []
    for alpha in grid:
        # alpha = 0 is no scaling; alpha = 1 scales each channel by its magnitude.
        s = torch.clamp(act**alpha, min=1e-4)
        # Normalize so that the largest and the smallest scale are reciprocal.
        s = s / torch.sqrt(s.max() * s.min())
        # Scale the columns up, quantize, and scale back down. Dividing the weights by
        # s is the same as dividing the activations by s: x @ (q / s).T == (x / s) @ q.T.
        w_hat = _quantize_groups(w * s, bits, group_size) / s
        # The objective is the error of the layer's output, not of its weights.
        losses.append(((target - x_calib @ w_hat.T) ** 2).mean())
        candidates.append(s)
    # argmin returns the first minimum: ties go to the earlier grid entry.
    best = int(torch.argmin(torch.stack(losses)))
    return candidates[best], grid[best]
