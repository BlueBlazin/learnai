# BUG: loads the model with mlx-lm, which the statement rules out; the model is not written.
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model as mlx_lm_load_model

from mlx import nn


class Wrapped(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self._inner = model

    def __call__(self, tokens: mx.array, cache: list | None = None) -> tuple[mx.array, list]:
        cache = make_prompt_cache(self._inner) if cache is None else cache
        return self._inner(tokens, cache=cache), cache


def load_model(path: str) -> nn.Module:
    model, _ = mlx_lm_load_model(path)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return Wrapped(model)


def generate(
    model, prompt_ids: list[int], max_new_tokens: int, eos_id: int | None = None
) -> list[int]:
    logits, cache = model(mx.array([prompt_ids]))
    out: list[int] = []
    for index in range(max_new_tokens):
        token = mx.argmax(logits[0, -1])
        mx.async_eval(token)
        out.append(token.item())
        if out[-1] == eos_id or index + 1 == max_new_tokens:
            break
        logits, cache = model(token.reshape(1, 1), cache)
    return out
