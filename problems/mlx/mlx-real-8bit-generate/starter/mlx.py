import json
from pathlib import Path

import mlx.core as mx

from learnai.kit import RealConfig
from mlx import nn


def load_model(path: str) -> nn.Module:
    """Build a quantized Granite model from a checkpoint directory, without ever
    holding its float weights.

    path: a directory with config.json (model_type "granite", with a "quantization"
        entry {"group_size", "bits"}) and model.safetensors in the MLX layout: for
        every linear layer and the embedding, `<name>.weight` (uint32 packed codes),
        `<name>.scales`, and `<name>.biases` (bfloat16); norm gains in bfloat16.
        RealConfig.from_json(json.loads(...)) turns the config into the fields of
        the previous problem, plus cfg.quantization = (bits, group_size).
    Returns an evaluated nn.Module whose parameter names are the tensor names of the
    checkpoint: packed codes still uint32, every floating parameter float32. Called as
        logits, cache = model(tokens)          # tokens: integer array (1, T)
        logits, cache = model(tokens, cache)   # continue after the cached tokens
    it returns float32 logits (1, T, vocab_size) and a cache of your own design.
    """
    raise NotImplementedError


def generate(
    model, prompt_ids: list[int], max_new_tokens: int, eos_id: int | None = None
) -> list[int]:
    """Greedy generation with the KV cache and asynchronous evaluation.

    model: what load_model returns; call it only as documented there.
    prompt_ids: list of P >= 1 ints. max_new_tokens >= 1.
    eos_id: stop after generating this token (it is kept); None: never.
    Returns the generated tokens as Python ints. Step t + 1 is scheduled with
    mx.async_eval before token t is read with .item(); mx.eval is not used.
    """
    raise NotImplementedError
