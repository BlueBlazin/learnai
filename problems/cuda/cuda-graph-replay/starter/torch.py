import contextlib
from collections.abc import Callable, Iterator, Sequence

import torch


@contextlib.contextmanager
def side_stream(device: torch.device) -> Iterator[None]:
    """Run the block on a CUDA stream of its own; on any other device, just run it.

    Provided: PyTorch's CUDA notes require the warm-up before a capture to run on a
    side stream. Use it as `with side_stream(device): ...` around the warm-up calls.
    """
    if device.type != "cuda":
        yield
        return
    torch.cuda.synchronize(device)
    with torch.cuda.stream(torch.cuda.Stream(device)):
        yield
    torch.cuda.synchronize(device)


class GraphedStep:
    """step_fn, captured once into `graph` and replayed on every call."""

    def __init__(
        self,
        step_fn: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]],
        example_inputs: Sequence[torch.Tensor],
        graph: torch.cuda.CUDAGraph,
        warmup: int = 3,
    ) -> None:
        """Allocate placeholders holding the example values, call step_fn on them
        `warmup` times, then once more inside `with torch.cuda.graph(graph):`.

        example_inputs: the tensors step_fn takes; they fix the shapes, dtypes, and device.
        graph: a torch.cuda.CUDAGraph, or the judge's CPU stand-in with the same methods.
        """
        raise NotImplementedError

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Copy `inputs` into the placeholders, replay the graph, and return copies of
        the outputs: a tensor if step_fn returned a tensor, a tuple if it returned a tuple.

        Raises ValueError for a wrong number of inputs or an input of the wrong shape.
        """
        raise NotImplementedError
