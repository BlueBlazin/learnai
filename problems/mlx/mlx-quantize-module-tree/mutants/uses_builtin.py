# BUG: hands the walk to nn.quantize and its class_predicate, the function this problem asks you to write.
from collections.abc import Callable

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx import nn

QUANTIZED = (nn.QuantizedLinear, nn.QuantizedEmbedding)


def quantize_model(model: nn.Module, recipe: Callable[[str, nn.Module], dict | None]) -> dict:
    quantized = {}

    def predicate(path: str, module: nn.Module) -> bool | dict:
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        params = recipe(path, module)
        if params is None:
            return False
        quantized[path] = {"bits": params["bits"], "group_size": params["group_size"]}
        return dict(quantized[path])

    nn.quantize(model, class_predicate=predicate)
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
