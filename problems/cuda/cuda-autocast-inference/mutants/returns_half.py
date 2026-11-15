# BUG: the logits are never cast back to float32; outside the autocast region the statistics are computed, and returned, in the 16-bit type.
import torch


@torch.inference_mode()
def forward_amp(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=device.type, dtype=dtype):
        logits = model(x)

    logits = logits / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs, entropy
