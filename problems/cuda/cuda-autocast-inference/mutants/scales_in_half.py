# BUG: the logits are divided by the temperature before the cast to float32; the quotient is rounded to the 16-bit type, and in float16 it overflows to inf above 65504.
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

    logits = (logits / temperature).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs, entropy
