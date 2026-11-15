# BUG: only the log-softmax is computed in float32; the entropy is computed on a 16-bit copy of the log-probabilities, where small terms of the sum are rounded away.
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

    logits = logits.float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    half = log_probs.to(dtype)
    entropy = -(half.exp() * half).sum(dim=-1)
    return log_probs, entropy.float()
