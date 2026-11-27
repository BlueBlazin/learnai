Implement two inference wrappers for PyTorch models. The `predict` function runs a module in evaluation mode under `torch.inference_mode()` and restores the module's exact training state afterward. The `as_inference` decorator makes any function, including a token-streaming generator, execute its body under inference mode.

## Specification

```python
def predict(model, x, device): ...
def as_inference(fn): ...
```

### `predict(model, x, device)`

Run `model(x)` for inference and return the output tensor on `device`.

- `model` is a `torch.nn.Module` whose parameters are already on `device`. Its submodules may be in training mode, in evaluation mode, or in a mix of both (for example, a training model with a frozen normalization layer in evaluation mode).
- `x` is an input tensor on any device. Move it to `device` before passing it to `model`.
- `device` is a `torch.device`. Tests use `torch.device("cpu")`, a CUDA device, and `torch.device("meta")` (a device whose tensors track shape and dtype without storing data).

Your implementation must satisfy four requirements:

1. **Evaluation mode.** During the forward pass, `model` and all of its submodules must be in evaluation mode. Dropout acts as the identity, and batch normalization uses its running statistics without updating them.
2. **Inference mode.** The forward pass must run under `torch.inference_mode()`. The returned tensor must be an *inference tensor*, meaning `out.is_inference()` is `True`, `out.requires_grad` is `False`, and `out.grad_fn` is `None`, even when `x` and the model parameters require gradients. Using `torch.no_grad()` or `.detach()` does not produce inference tensors and is not accepted.
3. **Exact state restoration.** After `predict` returns, every module in `model.modules()` must have the exact `training` flag it had before the call, regardless of how submodules were mixed.
4. **Exception safety.** If `model` raises an exception, let it propagate unchanged while still restoring every submodule's `training` flag.

In both normal and error paths, the caller's autograd state must remain unchanged (gradients enabled and inference mode off).

### `as_inference(fn)`

Return a callable that accepts the same positional and keyword arguments as `fn` and returns the same result, while running the body of `fn` under `torch.inference_mode()`.

- If `fn` is a regular function, tensors created inside it are inference tensors. If `fn` raises an exception, the exception propagates and the caller's autograd mode is restored.
- If `fn` is a **generator function**, calling the wrapper returns an iterator that yields the same values. Each step of the generator body must run under inference mode, so every yielded tensor is an inference tensor. Between steps, while the generator is suspended, the caller's code must run in the caller's own mode, where `torch.is_inference_mode_enabled()` is `False`.

## Shapes

Here $B$ is the batch size, $F$ is the input feature dimension, and $C$ is the output feature dimension.

| Name | Shape | Dtype |
|---|---|---|
| `x` | `(B, F)` | `float32` |
| `predict` output | `(B, C)` (or whatever `model` returns) | `float32` |

## Example

```python
model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
# weight [[1, 0], [1, 1]], bias [0.5, -0.5]
model.train()

out = predict(model, torch.tensor([[2.0, 4.0]]), torch.device("cpu"))
# out: [[2.5, 5.5]]        2*1 + 4*0 + 0.5 = 2.5,   2*1 + 4*1 - 0.5 = 5.5
# out.is_inference(): True
# model.training: True     (restored to its pre-call value)
```

In training mode, the same input would produce one of `[[5, 11]]`, `[[5, 0]]`, `[[0, 11]]`, or `[[0, 0]]`. Dropout with $p = 0.5$ zeroes each activation with probability $0.5$ and doubles the surviving values.

```python
@as_inference
def stream(x, steps):
    for _ in range(steps):
        x = torch.tanh(layer(x))
        yield x


for h in stream(x, 3):
    # h.is_inference(): True
    # torch.is_inference_mode_enabled(): False   (caller code between steps)
    ...
```

## Constraints

- Outputs are compared against the model run in evaluation mode with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- CPU test models are small multilayer perceptrons with dropout and, in some cases, `BatchNorm1d`, with $B \le 8$. One GPU test case uses a 12-layer network with a batch size of 4096.
- `predict` must not move `model`, copy it, or modify its parameters or buffers.
- **CPU vs. GPU tests.** Tests without a hardware mark run on the CPU and verify all rules above. Two tests run only on a machine with an NVIDIA GPU. The first checks that the output is an inference tensor on the CUDA device. The second checks that peak `torch.cuda.max_memory_allocated()` during `predict` is at most 60% of the peak of a forward pass that records the autograd graph. The GPU tests were written from the PyTorch 2.15 documentation and could not be executed when this problem was authored.

## Why this matters

A forward pass with autograd enabled retains every intermediate activation until the output is freed so that a backward pass can compute gradients. On a serving GPU, those saved activations consume memory that could otherwise hold a larger batch. For the 12-layer network in the GPU test, peak memory is roughly four times higher with autograd enabled than without it. `torch.inference_mode()` is the lowest-overhead way to disable recording because it skips the autograd graph, version counters, and view tracking.

The `training` flag is a separate source of silent bugs. A model left in training mode returns noisy dropout outputs and overwrites its batch-normalization running statistics with live traffic. Conversely, restoring a model with `model.train()` overwrites submodules that a fine-tuning setup intentionally froze in evaluation mode. Streaming endpoints add a third trap, because a naive decorator around a generator exits inference mode before the first token is even computed.
