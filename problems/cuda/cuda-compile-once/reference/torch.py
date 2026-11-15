from collections.abc import Callable

import torch

Step = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_step(
    model: torch.nn.Module, cache: object, backend: str = "inductor", mode: str | None = None
) -> Step:
    def step(tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Nothing here depends on a Python number that changes or on the value of a
        # tensor: the trace of one call is valid for every call with the same shapes.
        return model(tokens, pos, cache)

    # One call of torch.compile, here, and nowhere else. fullgraph=True: a graph break
    # is an error, not a silent split. dynamic=False: every guard failure costs a full
    # compilation, so the number of graphs is the number of distinct input shapes.
    return torch.compile(step, fullgraph=True, dynamic=False, backend=backend, mode=mode)


def generate(
    step: Step, prompt: torch.Tensor, n_new: int, device: torch.device, bucket: int = 16
) -> torch.Tensor:
    batch, n_prompt = prompt.shape
    # Pad on the right to the next multiple of `bucket`: prompts of 1 to 16 tokens
    # share one shape, and so one graph. The padding sits after every real token, so
    # the causal mask keeps it out of their attention, and the slots it fills in the
    # cache are overwritten by the decode steps before anything attends to them.
    padded = -(-n_prompt // bucket) * bucket
    tokens = torch.zeros((batch, padded), dtype=prompt.dtype, device=device)
    tokens[:, :n_prompt] = prompt
    out = torch.empty((batch, n_new), dtype=prompt.dtype, device=device)

    # Positions are 0-d tensors on the device. Their values change at every step, and
    # a tensor's value is never part of a guard; a Python int's value is.
    logits = step(tokens, torch.tensor(0, device=device))
    # The first new token comes from the last real token, not from the last slot.
    token = logits[:, n_prompt - 1].argmax(dim=-1, keepdim=True)
    out[:, 0] = token[:, 0]
    pos = torch.tensor(n_prompt, device=device)
    for i in range(1, n_new):
        # Always (batch, 1) tokens and a 0-d position: the second and last graph.
        token = step(token, pos)[:, 0].argmax(dim=-1, keepdim=True)
        out[:, i] = token[:, 0]
        pos += 1
    return out
