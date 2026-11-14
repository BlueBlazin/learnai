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
    """Load the tensors of a safetensors file into the parameters of `module`.

    module: any mlx.nn.Module; its parameters are named by their dotted path
        ("layers.0.q_proj.weight").
    path: a .safetensors file. The tensor named `prefix + <parameter path>` belongs
        to that parameter; every other tensor of the file is unexpected.
    transpose_linear: the file stores the weight of every nn.Linear as (in, out);
        transpose it to MLX's (out, in) before the shape check.
    strict: raise ValueError if a parameter has no tensor or a tensor has no
        parameter. A tensor of the wrong shape raises ValueError whatever `strict` is.
    Each tensor is cast to the dtype of the parameter it replaces. On return the
    module's parameters are evaluated; tensors that were not loaded are never read.
    Returns (missing, unexpected): sorted parameter paths without a tensor, and sorted
    file names without a parameter. On ValueError the module is unchanged.
    """
    raise NotImplementedError
