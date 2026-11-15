import torch


def schedule(n_masked: int, steps: int) -> list[int]:
    """How many positions to commit in each of `steps` steps.

    n_masked: the number of masked positions at the start, an int >= 0.
    steps: the number of reverse steps, an int >= 1.
    Returns a list of `steps` Python ints that add up to n_masked: n_masked // steps
    each, plus one for each of the first n_masked % steps steps.
    """
    raise NotImplementedError


def unmask_step(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    mask_id: int,
    n_unmask: int,
    temperature: float,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """One reverse step of masked-diffusion decoding.

    tokens: integer tensor, shape (L,). Masked positions hold mask_id.
    logits: float32 tensor, shape (L, V): the model's prediction for every position.
    mask_id: the id of the mask token, 0 <= mask_id < V. It is never predicted.
    n_unmask: how many masked positions to commit in this step, an int >= 0; if
        fewer are masked, all of them are committed.
    temperature: 0 for the most probable token, otherwise candidates are drawn from
        softmax(logits / temperature).
    uniforms: float32 tensor, shape (L,), in [0, 1): uniforms[i] draws the candidate
        of position i by inverse-CDF sampling (unused when temperature is 0).
    Confidence of a masked position: the probability softmax(logits) (temperature 1)
    gives its candidate. The n_unmask most confident masked positions (ties: lowest
    index first) take their candidates; every other position is unchanged.
    Returns the new tokens, an integer tensor of shape (L,).
    """
    raise NotImplementedError
