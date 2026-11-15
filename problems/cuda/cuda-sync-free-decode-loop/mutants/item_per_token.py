# BUG: reads every sampled token with .item(), collects the tokens in a Python list, and tests for EOS in Python; one host synchronization per token.
import torch


def decode(
    model,
    prompt: torch.Tensor,
    n: int,
    uniforms: torch.Tensor,
    device: torch.device,
    eos_id: int,
    check_every: int,
) -> torch.Tensor:
    prompt = prompt.to(device)
    uniforms = uniforms.to(device)
    sampled = []
    logits, cache = model(prompt)
    for step in range(n):
        cdf = torch.softmax(logits[-1], dim=-1).cumsum(dim=-1)
        token = (cdf <= uniforms[step] * cdf[-1]).sum().clamp(max=cdf.shape[0] - 1).item()
        sampled.append(token)
        if token == eos_id or step + 1 == n:
            break
        logits, cache = model(torch.tensor([token], device=device), cache)
    sampled += [eos_id] * (n - len(sampled))
    return torch.tensor(sampled, device=device).cpu()
