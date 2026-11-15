# BUG: checks the flag when the zero-based step index is a multiple of check_every, that is after tokens 1, check_every + 1, ...; an EOS found at token check_every costs one more model call than allowed.
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
    tokens = torch.full((n,), eos_id, dtype=torch.long, device=device)
    done = torch.zeros((), dtype=torch.bool, device=device)

    logits, cache = model(prompt)
    for step in range(n):
        cdf = torch.softmax(logits[-1], dim=-1).cumsum(dim=-1)
        token = (cdf <= uniforms[step] * cdf[-1]).sum().clamp(max=cdf.shape[0] - 1)
        token = torch.where(done, eos_id, token)
        tokens[step] = token
        done = done | (token == eos_id)
        if step + 1 == n:
            break
        if step % check_every == 0 and done.item():
            break
        logits, cache = model(token.reshape(1), cache)
    return tokens.cpu()
