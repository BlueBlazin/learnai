# BUG: tracks EOS with a flag on the device and checks it on schedule, but appends int(token) to a Python list every step; that conversion is a host read per token.
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
    done = torch.zeros((), dtype=torch.bool, device=device)

    logits, cache = model(prompt)
    for step in range(n):
        cdf = torch.softmax(logits[-1], dim=-1).cumsum(dim=-1)
        token = (cdf <= uniforms[step] * cdf[-1]).sum().clamp(max=cdf.shape[0] - 1)
        token = torch.where(done, eos_id, token)
        sampled.append(int(token))
        done = done | (token == eos_id)
        if step + 1 == n:
            break
        if (step + 1) % check_every == 0 and done.item():
            break
        logits, cache = model(token.reshape(1), cache)
    sampled += [eos_id] * (n - len(sampled))
    return torch.tensor(sampled, device=device).cpu()
