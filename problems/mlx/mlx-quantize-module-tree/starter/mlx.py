from collections.abc import Callable

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map_with_path

from mlx import nn


def quantize_model(model: nn.Module, recipe: Callable[[str, nn.Module], dict | None]) -> dict:
    """Quantize, in place, the nn.Linear and nn.Embedding layers that `recipe` selects.

    model: an nn.Module tree; child modules may sit in attributes, lists, and dicts.
    recipe: called once as recipe(path, module) for every nn.Linear and nn.Embedding,
        with the dotted path of the module ("layers.0.q_proj"). It returns None (leave
        the layer as it is) or {"bits": b, "group_size": g} (replace it by an
        nn.QuantizedLinear or nn.QuantizedEmbedding in affine mode).
    Returns {path: {"bits": b, "group_size": g}} for every layer that was replaced.
    On return, every parameter of the model is evaluated.
    """
    raise NotImplementedError


def bits_per_weight(model: nn.Module) -> float:
    """8 * (bytes of all parameter arrays) / (number of weights they stand for).

    A quantized layer's packed `weight` counts as its out * in weights; its `scales`
    and `biases` count as bytes only. Returns a Python float.
    """
    raise NotImplementedError
