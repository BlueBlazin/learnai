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
    """Sample up to n tokens on `device`, reading from the device as rarely as allowed.

    model: called as `logits, cache = model(tokens)` and `model(tokens, cache)` with a
        1-D int64 tensor on `device`; logits is (T, V) float32 on `device`.
    prompt: int64 tensor (P,), on the CPU.
    uniforms: float32 tensor (n,), on the CPU; uniforms[i] samples token i.
    Returns an int64 tensor of shape (n,) on the CPU: the sampled tokens up to and
    including the first eos_id, then eos_id in every later position.
    At most ceil(n / check_every) + 1 host synchronizations: one read of the "done"
    flag after every check_every-th token, and one `.cpu()` at the end.
    """
    raise NotImplementedError
