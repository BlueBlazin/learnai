## The idea

Two independent switches control how a PyTorch module executes, and inference requires both.

**The training flag** (`module.training`, toggled by `train()` and `eval()`) changes what certain layers compute. Dropout zeroes activations only when `training` is `True`. Batch normalization uses batch statistics and updates its running averages only when `training` is `True`. This flag is completely orthogonal to autograd, so `model.eval()` still builds the autograd graph.

**The autograd mode** controls what PyTorch records. With gradients enabled, every operation on a tensor that requires gradients adds a node to the graph and retains the activations needed for a backward pass. In a deep network, that keeps every layer's activations alive until the output is released. `torch.no_grad()` disables graph recording. `torch.inference_mode()` goes further by creating *inference tensors* that omit version counters and view-tracking metadata, reducing per-operation overhead. The trade-off is that inference tensors cannot later enter an autograd-tracked computation, which is fine for serving. `Tensor.is_inference()` distinguishes inference tensors from `no_grad` tensors, and the tests check this flag directly.

Because `model.training` is mutable state on the caller's model, `predict` saves every submodule's flag before calling `model.eval()` and restores them inside a `finally` block. The autograd mode is thread-local state, so a `with torch.inference_mode():` block automatically restores it on both normal return and exception exit.

## Reference solution (PyTorch)

```python
def predict(model, x, device):
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            return model(x.to(device))
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn):
    return torch.inference_mode()(fn)
```

`flags` stores one entry per submodule because `model.train(mode)` is recursive and sets the flag on every descendant. Saving only `model.training` and calling `model.train(was_training)` afterward only works when every submodule started with the same flag.

`as_inference` relies on how PyTorch's context-manager decorators handle generator functions. Instead of wrapping the initial call, `torch.inference_mode()(fn)` wraps each resumption of the generator. It enters inference mode right before `next()` advances the generator and exits as soon as the generator yields. As a result, the generator body always runs under inference mode while the caller's loop body does not. Writing this manually requires a loop that enters inference mode around each `next(gen)` call and yields outside the `with` block.

## Common mistakes

- **Using `torch.no_grad()` instead of `torch.inference_mode()`.** The numerical output matches and no graph is recorded, but the returned tensors are ordinary tensors with version counters and view tracking. Use `no_grad()` when outputs must flow back into autograd later, and `inference_mode()` for pure inference.
- **Running with autograd enabled and calling `.detach()` at the end.** `.detach()` only cuts the final tensor loose after the forward pass finishes. Every intermediate activation was still retained during execution, which fails the GPU memory test.
- **Leaving the model in training mode.** Inference mode does not imply evaluation mode. Dropout stays active, and batch normalization overwrites its running statistics with live request data.
- **Calling `eval()` without restoring the original state.** The caller's next training step silently runs without dropout.
- **Restoring flags without a `finally` block.** A single exception during inference leaves the model stuck in evaluation mode for the rest of the process.
- **Restoring with `model.train(was_training)`.** This propagates the root module's flag to every submodule and unfreezes layers that were intentionally kept in evaluation mode.
- **Calling `torch.set_grad_enabled(False)` as a bare statement.** This mutates thread-local state without restoring it, leaving the caller with gradients disabled.
- **Forgetting to move `x` to `device`.** This passes when tests happen to put `x` on the model's device, then fails with `"Expected all tensors to be on the same device"` when batches arrive on the CPU.
- **Wrapping a generator with `with torch.inference_mode(): return fn(...)`.** Calling a generator function only constructs the iterator without running its body. The `with` block exits before the first step executes, so the entire stream runs with autograd enabled.
- **Wrapping a generator with `with torch.inference_mode(): yield from fn(...)`.** The `with` block stays open while the generator is suspended. Because inference mode is thread-local, the caller's code between yielded tokens also runs in inference mode.

## Complexity

One forward pass. Under inference mode, live activation memory holds only the input and output of the current layer. With autograd enabled, live memory grows linearly with depth because each layer saves activations for the backward pass.

## In the wild

Inference servers and token-generation loops wrap forward passes with `torch.inference_mode()`, while libraries whose outputs might be differentiated by user code use `torch.no_grad()`. On a GPU, the memory savings appear directly in `torch.cuda.max_memory_allocated()`. The memory figures quoted for the GPU test are derived from the tensor sizes and the PyTorch 2.15 documentation, as the GPU case was not run on hardware by the author.
