# BUG: only nn.Linear layers are offered to the recipe; an nn.Embedding is never quantized, whatever the recipe would say.
from collections.abc import Callable

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map_with_path

from mlx import nn

QUANTIZED = (nn.QuantizedLinear, nn.QuantizedEmbedding)


def quantize_model(model: nn.Module, recipe: Callable[[str, nn.Module], dict | None]) -> dict:
    quantized = {}

    def convert(path: str, module: nn.Module) -> nn.Module:
        if not isinstance(module, nn.Linear):
            return module
        params = recipe(path, module)
        if params is None:
            return module
        bits, group_size = params["bits"], params["group_size"]
        quantized[path] = {"bits": bits, "group_size": group_size}
        return module.to_quantized(group_size=group_size, bits=bits)

    leaves = tree_map_with_path(convert, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
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
