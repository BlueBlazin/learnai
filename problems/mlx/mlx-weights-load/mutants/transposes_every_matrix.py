# BUG: transpose_linear transposes every 2-D tensor, including embedding tables, which are not Linear weights.
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx import nn


def load_into(
    module: nn.Module,
    path: str,
    *,
    prefix: str = "",
    transpose_linear: bool = False,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    tensors = mx.load(path)
    current = dict(tree_flatten(module.parameters()))

    loaded, unexpected = {}, []
    for key, value in tensors.items():
        name = key[len(prefix) :] if key.startswith(prefix) else None
        if name not in current:
            unexpected.append(key)
            continue
        if transpose_linear and value.ndim == 2:
            value = value.T
        if value.shape != current[name].shape:
            raise ValueError(
                f"{key}: shape {value.shape} does not fit {name}, {current[name].shape}"
            )
        loaded[name] = value.astype(current[name].dtype)
    missing = sorted(set(current) - set(loaded))
    unexpected.sort()
    if strict and (missing or unexpected):
        raise ValueError(f"missing parameters: {missing}; unexpected tensors: {unexpected}")

    module.update(tree_unflatten(list(loaded.items())))
    mx.eval(module.parameters())
    return missing, unexpected
