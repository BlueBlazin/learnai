Write the two wrappers every inference service puts around a model: `predict`, which
runs a module in evaluation mode under `torch.inference_mode()` and hands the module
back exactly as it found it, and `as_inference`, which makes any function, including a
token-streaming generator, run under inference mode. Each is a few lines; what makes
them non-trivial is that every wrong version returns the right shape, and most return
the right numbers.

## Specification

```python
def predict(model, x, device): ...
def as_inference(fn): ...
```

**`predict(model, x, device)`** returns `model(x)` computed for inference:

- `model`: a `torch.nn.Module` whose parameters are already on `device`. It may be in
  training mode, in evaluation mode, or mixed: a model that is training can contain
  submodules that were put in evaluation mode on purpose (a frozen normalization
  layer), and the reverse.
- `x`: a float32 tensor on any device. It is moved to `device` before the model sees
  it.
- `device`: a `torch.device`. The tests use `torch.device("cpu")`, a CUDA device, and
  `torch.device("meta")`, a device whose tensors have a shape and no data.
- Returns the model's output, a tensor on `device`.

Four things must hold:

1. **Evaluation mode.** During the forward pass the model and all its submodules are
   in evaluation mode: dropout is the identity, and batch normalization uses its
   running statistics and does not update them.
2. **Inference mode.** The forward pass runs under `torch.inference_mode()`. The
   result is an *inference tensor*: `out.is_inference()` is `True`,
   `out.requires_grad` is `False`, and `out.grad_fn` is `None`, even when the
   parameters and `x` require gradients. `torch.no_grad()` and `.detach()` do not
   produce inference tensors and are not accepted.
3. **The model is handed back as it was.** After the call, **every** module of
   `model.modules()` has the `training` flag it had before the call, whatever mix
   that was.
4. **Also on error.** If the model raises, the exception propagates unchanged, and
   rule 3 still holds.

In both cases the caller's autograd mode is what it was before the call: gradients
enabled, inference mode off.

**`as_inference(fn)`** returns a callable that takes the same positional and keyword
arguments as `fn` and returns what `fn` returns, with the body of `fn` running under
inference mode:

- If `fn` is a plain function, the tensors it creates are inference tensors. If it
  raises, the exception propagates and the caller's mode is restored.
- If `fn` is a **generator function**, calling the wrapper returns an iterator that
  yields what the generator yields. Every step of the generator body runs under
  inference mode, so the yielded tensors are inference tensors. Between two steps,
  while the generator is suspended, the caller's own code runs in the caller's own
  mode: `torch.is_inference_mode_enabled()` is `False` there.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, F)` | float32 |
| result of `predict` | whatever the model returns, here `(B, C)` | float32 |

## Example

```python
model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
# weight [[1, 0], [1, 1]], bias [0.5, -0.5]
model.train()

out = predict(model, torch.tensor([[2.0, 4.0]]), torch.device("cpu"))
# out: [[2.5, 5.5]]        2*1 + 4*0 + 0.5 = 2.5,   2*1 + 4*1 - 0.5 = 5.5
# out.is_inference(): True
# model.training: True     (as before the call)
```

In training mode the same call would return one of `[[5, 11]]`, `[[5, 0]]`,
`[[0, 11]]`, `[[0, 0]]`: dropout with $p = 0.5$ zeroes each activation with
probability one half and doubles the survivors.

```python
@as_inference
def stream(x, steps):
    for _ in range(steps):
        x = torch.tanh(layer(x))
        yield x


for h in stream(x, 3):
    # h.is_inference(): True
    # torch.is_inference_mode_enabled(): False   (the caller's code, between steps)
    ...
```

## Constraints

- Outputs are compared with the model run in evaluation mode to a relative tolerance
  of `1e-4` and an absolute tolerance of `1e-5`.
- The test models are small multilayer perceptrons with dropout and, in some cases,
  `BatchNorm1d`; $B \le 8$. On the GPU one case uses a 12-layer network with a batch
  of 4096.
- `predict` must not move the model, copy it, or change its parameters and buffers.
- **What runs where.** The cases without a hardware mark run on the CPU and check all
  of the above. Two cases run only on a machine with an NVIDIA GPU: one checks that
  the result is an inference tensor on the CUDA device, the other that the peak of
  `torch.cuda.max_memory_allocated()` during `predict` is at most 60% of the peak of a
  forward pass that records the autograd graph. The GPU cases were written from the
  PyTorch 2.15 documentation and could not be run when the problem was authored.

## Why this matters

A forward pass that records the autograd graph keeps every intermediate activation
alive until the output is released, because a backward pass might need it. On a GPU
serving requests that is memory taken from the batch size: the 12-layer network of the
GPU case holds about four times as much at its peak with the graph as without it.
Inference mode is the cheapest of the ways to turn recording off (no graph, no version
counters, no view tracking), which is why serving code uses it and not `no_grad`. The
training flag is the other half: a model left in training mode answers with dropout
noise and silently overwrites its batch-normalization statistics with those of live
traffic, and a helper that "restores" training mode with `model.train()` thaws layers
that a fine-tuning run had frozen. Streaming endpoints add the generator trap: a
decorator that looks right runs the whole generation loop with autograd on.
