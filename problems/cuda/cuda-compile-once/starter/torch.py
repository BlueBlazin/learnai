from collections.abc import Callable

import torch

Step = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_step(
    model: torch.nn.Module, cache: object, backend: str = "inductor", mode: str | None = None
) -> Step:
    """Compile one step of the model, once.

    model(tokens, pos, cache) -> float32 logits of shape (batch, T, vocab), where tokens
    is int64 of shape (batch, T) and pos is the slot of tokens[:, 0].
    Returns step(tokens, pos) -> the same logits, where step is the result of a single
    torch.compile(..., fullgraph=True, dynamic=False, backend=backend, mode=mode) call.
    """
    raise NotImplementedError


def generate(
    step: Step, prompt: torch.Tensor, n_new: int, device: torch.device, bucket: int = 16
) -> torch.Tensor:
    """Greedy decoding through `step` with a bounded number of input shapes.

    prompt: int64 tokens of shape (batch, P), on `device`.
    Calls step once on the prompt, padded on the right with zeros to a multiple of
    `bucket`, at position 0, and then once per new token except the last, on a
    (batch, 1) block. Every position is a 0-d int64 tensor on `device`.
    Returns the n_new greedy tokens: int64, shape (batch, n_new), on `device`.
    """
    raise NotImplementedError
