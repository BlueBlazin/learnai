# BUG: checks the outputs for NaN inside the capture block; the check reads a value on the host, which a capturing stream does not permit.
import contextlib
from collections.abc import Callable, Iterator, Sequence

import torch


@contextlib.contextmanager
def side_stream(device: torch.device) -> Iterator[None]:
    """Run the block on a CUDA stream of its own; on any other device, just run it."""
    if device.type != "cuda":
        yield
        return
    torch.cuda.synchronize(device)
    with torch.cuda.stream(torch.cuda.Stream(device)):
        yield
    torch.cuda.synchronize(device)


class GraphedStep:
    def __init__(
        self,
        step_fn: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]],
        example_inputs: Sequence[torch.Tensor],
        graph: torch.cuda.CUDAGraph,
        warmup: int = 3,
    ) -> None:
        self.graph = graph
        # The placeholders: memory of our own, allocated before the capture and filled
        # with the example values. The graph will read its inputs from these addresses
        # for as long as it lives.
        self.static_inputs = tuple(x.clone() for x in example_inputs)
        # Warm-up: eager runs that get every one-time cost (lazy initialization, the
        # allocator's first blocks) out of the way, so that none of it is recorded.
        with side_stream(self.static_inputs[0].device):
            for _ in range(warmup):
                step_fn(*self.static_inputs)
        # Capture: one more run, recorded. Only kernels are recorded, with the addresses
        # they read and write; the Python of step_fn never runs again.
        with torch.cuda.graph(graph):
            self.static_outputs = step_fn(*self.static_inputs)
            outputs = self.static_outputs
            for out in (outputs,) if isinstance(outputs, torch.Tensor) else outputs:
                if torch.isnan(out).any():
                    raise ValueError("step_fn returned NaN for the example inputs")

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.static_inputs):
            raise ValueError(f"expected {len(self.static_inputs)} inputs, got {len(inputs)}")
        for static, new in zip(self.static_inputs, inputs, strict=True):
            if new.shape != static.shape:
                raise ValueError(f"expected shape {tuple(static.shape)}, got {tuple(new.shape)}")
        # Copy INTO the placeholders. Rebinding a name would change nothing the graph
        # can see: it reads the captured addresses.
        for static, new in zip(self.static_inputs, inputs, strict=True):
            static.copy_(new)
        self.graph.replay()
        # The static outputs are overwritten by the next replay: hand out copies.
        if isinstance(self.static_outputs, torch.Tensor):
            return self.static_outputs.clone()
        return tuple(out.clone() for out in self.static_outputs)
