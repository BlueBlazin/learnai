## The idea

Two independent switches decide how a module runs, and inference needs both.

**The training flag** (`module.training`, set by `train()` and `eval()`) changes what
some layers *compute*: dropout zeroes activations only in training mode, and batch
normalization normalizes with the statistics of the current batch and updates its
running averages only in training mode. The PyTorch notes call it "completely
orthogonal" to the autograd modes: `model.eval()` records the graph as before.

**The autograd mode** changes what is *recorded*. With gradients enabled, every
operation on a tensor that requires grad adds a node to the graph, and the node keeps
the tensors that the backward pass would need. For a stack of layers that is every
activation, alive until the output is dropped. `torch.no_grad()` stops the recording.
`torch.inference_mode()` goes further: tensors created under it are *inference
tensors*, which carry no version counter and no view-tracking metadata, so there is
less bookkeeping per operation. The price is that an inference tensor can never be
used in a computation that autograd records ("Inference tensors cannot be saved for
backward"), which is no restriction for a serving path. `Tensor.is_inference()` tells
the two apart, and that is what the tests look at.

The rest of `predict` is being a good guest. The flags are state on the caller's
model, so they are saved before `eval()` and restored in a `finally` block. The mode
is thread-local state; the `with` statement restores it on every exit path.

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

`flags` holds one entry per module because `train(mode)` is recursive: it sets the
flag of the module and of every descendant. Saving only `model.training` and calling
`model.train(was_training)` afterward is correct exactly when all flags were equal.

`as_inference` leans on a detail of torch's context-manager decorators: for a
generator function they do not wrap the call, they wrap each resumption. The mode is
entered before the generator's `next` runs and left when it yields, so the body always
runs under inference mode and the caller never does. Writing it by hand means a loop
that enters the mode around every `next(gen)` and yields outside it.

## Common mistakes

- **`torch.no_grad()` where inference mode was asked for.** The numbers are equal
  and no graph is recorded; the results are ordinary tensors with version counters
  and view tracking. It is the right tool when the results must flow back into
  autograd later, and the slower one otherwise.
- **Running the pass with the graph and detaching the result.** `.detach()` cuts the
  result loose after the fact. During the pass every activation was kept alive, which
  is the memory the GPU case measures.
- **Leaving the model in training mode.** Inference mode does not imply evaluation
  mode (the `inference_mode` documentation carries a warning about exactly this).
  Dropout stays active, and batch normalization overwrites its running statistics
  with the statistics of the request.
- **`eval()` without restoring.** The caller's next training step runs without
  dropout and nobody notices.
- **Restoring without `finally`.** One request that makes the model raise leaves it
  in evaluation mode for the rest of the process.
- **`model.train(was_training)`.** It writes the root's flag into every submodule and
  thaws layers that were deliberately frozen in evaluation mode.
- **`torch.set_grad_enabled(False)` as a statement.** It is a global switch for the
  thread, not a scope; the caller is left with gradients off.
- **Not moving `x`.** It works in every test that happens to pass an input on the
  model's device and fails with "Expected all tensors to be on the same device" in
  production, where the batch arrives on the CPU.
- **`with torch.inference_mode(): return fn(...)` around a generator function.** The
  call only creates the generator; the `with` block has ended before the first step
  runs, and the whole stream is computed with autograd on.
- **`with torch.inference_mode(): yield from fn(...)`.** Now the block stays open
  while the generator is suspended, and since the mode is thread-local, the caller's
  code between two tokens runs in inference mode too.

## Complexity

One forward pass. Under inference mode the live memory at any point is the input and
output of the layer being computed; with the graph recorded it grows with depth, one
saved activation per layer that needs it for backward.

## In the wild

Inference servers and `generate` loops are decorated with `torch.inference_mode()` or
`torch.no_grad()` for this reason; libraries that must hand tensors back to user code
that might train on them use `no_grad`. On a GPU the saving is visible as
`torch.cuda.max_memory_allocated()`. The numbers quoted for the GPU case come from the
sizes of the tensors involved and the PyTorch documentation; the case itself has not
been run on a GPU by the author.
