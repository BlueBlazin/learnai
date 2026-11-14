# BUG: the hand-written walk follows child modules that are attributes and skips the ones inside lists and dicts, so the blocks of a transformer are never visited.
from collections.abc import Callable

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx import nn

QUANTIZED = (nn.QuantizedLinear, nn.QuantizedEmbedding)


def quantize_model(model: nn.Module, recipe: Callable[[str, nn.Module], dict | None]) -> dict:
    quantized = {}

    def walk(module: nn.Module, prefix: str) -> None:
        for name, child in module.children().items():
            if not isinstance(child, nn.Module):
                continue
            path = f"{prefix}{name}"
            if isinstance(child, (nn.Linear, nn.Embedding)):
                params = recipe(path, child)
                if params is not None:
                    bits, group_size = params["bits"], params["group_size"]
                    quantized[path] = {"bits": bits, "group_size": group_size}
                    module[name] = child.to_quantized(group_size=group_size, bits=bits)
            else:
                walk(child, f"{path}.")

    walk(model, "")
    mx.eval(model.parameters())
    return quantized


def bits_per_weight(model: nn.Module) -> float:
    arrays = [value for _, value in tree_flatten(model.parameters())]
    stored_bytes = sum(value.nbytes for value in arrays)
    weights = sum(value.size for value in arrays)
    for _, module in model.named_modules():
        if isinstance(module, QUANTIZED):
            stored = module.weight.size + module.scales.size + module.biases.size
            weights += module.weight.size * 32 // module.bits - stored
    return 8 * stored_bytes / weights
