# BUG: instead of an autocast region the model itself is converted with model.to(dtype); the caller's parameters are 16-bit from then on.
import torch


@torch.inference_mode()
def forward_amp(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.to(dtype)(x.to(dtype))

    logits = logits.float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs, entropy
