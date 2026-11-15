import torch


@torch.inference_mode()
def decode(
    model,
    prompt: torch.Tensor,
    n: int,
    uniforms: torch.Tensor,
    device: torch.device,
    eos_id: int,
    check_every: int,
) -> torch.Tensor:
    # Everything the loop touches lives on the device before the loop starts.
    prompt = prompt.to(device)
    uniforms = uniforms.to(device)
    # A fixed-size buffer, prefilled with what follows an EOS. Its shape never depends
    # on a value the device has computed, so the host never has to ask for one.
    tokens = torch.full((n,), eos_id, dtype=torch.long, device=device)
    done = torch.zeros((), dtype=torch.bool, device=device)

    logits, cache = model(prompt)
    for step in range(n):
        # Inverse-CDF sampling on the device: the number of cumulative sums that do
        # not exceed the threshold is the index of the first one that does.
        cdf = torch.softmax(logits[-1], dim=-1).cumsum(dim=-1)
        token = (cdf <= uniforms[step] * cdf[-1]).sum().clamp(max=cdf.shape[0] - 1)
        # After an EOS the loop may run on until the next check; those steps write EOS.
        token = torch.where(done, eos_id, token)
        tokens[step] = token
        done = done | (token == eos_id)

        if step + 1 == n:
            break  # the last token is returned, never fed
        # The only host read in the loop: once per check_every tokens.
        if (step + 1) % check_every == 0 and done.item():
            break
        logits, cache = model(token.reshape(1), cache)

    return tokens.cpu()  # one transfer for the whole result
