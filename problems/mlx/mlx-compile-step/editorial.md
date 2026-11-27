## The idea

`mx.compile(fn)` returns a wrapper that behaves like `fn` while running its Python body far less often. On each call, MLX inspects the *signature* of the arguments, consisting of the shape and dtype of every array and the value of every Python scalar. When it sees a new signature, it **traces** `fn` by calling it with placeholder arrays, recording the resulting graph, simplifying and fusing operations, and caching the compiled graph. On later calls with the same signature, MLX skips Python completely and replays the cached graph on the new arrays.

Three consequences follow directly from how tracing works.

1. **Any change to the signature triggers a retrace.** The KV-cache buffers in `state` have fixed size `max_len`, so their shapes never change across steps (unlike a growing cache, which changes shape on every step). The subtle trap is the token dtype. `mx.argmax` returns `uint32`. If `step` receives `int32` and returns `uint32`, the second call has a different input dtype and triggers a second trace. Casting the argmax result back to `mx.int32` keeps the signature identical on every call.
2. **Array values do not exist during tracing.** While tracing, `token` is a placeholder with no concrete data. Writing `if token == eos_id:` asks for a Python `bool`, which forces evaluation of the placeholder and raises *"Attempting to eval an array during function transformations like compile or vmap is not allowed."* Any value-dependent choice must stay inside the graph using `mx.where(token == eos_id, eos_id, best)`, which computes both branches and selects per row. Because `eos_id` is a fixed Python `int` closed over by `step`, recording it as a constant is fine.
3. **Arrays read outside the arguments are frozen as constants.** `model.step` reads weight arrays directly from `model`. During tracing, MLX sees those arrays as non-placeholder values and bakes their current contents into the graph. If the caller later updates the weights, the compiled graph keeps using the old ones with no error and no retrace. Passing `inputs=model.state` to `mx.compile` declares those implicit inputs. Before every call, MLX reads the current arrays from `model.state` and feeds them into the graph alongside the explicit arguments. Because `model.state` is the module's live dict, weight updates take effect immediately.

## Reference solution (MLX)

```python
def make_step(model, eos_id):
    def step(token, state):
        logits, state = model.step(token, state)
        best = mx.argmax(logits, axis=-1).astype(mx.int32)
        next_token = mx.where(token == eos_id, eos_id, best)
        return next_token, state

    return mx.compile(step, inputs=model.state)
```

`mx.compile` accepts nested trees as inputs and outputs, so `state` passes through unchanged. In `mx.where`, combining an `int32` array with a Python `int` preserves `int32` because Python scalars are weakly typed in MLX. The compiled function still returns lazy arrays, leaving the single `mx.eval(token, state)` to the caller.

## What it buys

Measured on an M1 Max (MLX 0.32.4), per step after the first call, taking the best of four runs of 200 steps. "Tiny" is the default `MLXTinyLM`. "Larger" has 6 layers, width 256, and vocabulary size 8192.

| step function | tiny | larger | traces in 201 calls |
|---|---|---|---|
| plain Python | 0.69 ms | 2.11 ms | (201 runs of the body) |
| compiled once | 0.61 ms | 1.58 ms | 1 |
| `mx.compile` called inside the step | 1.20 ms | 3.33 ms | 201 |

Small models spend a large fraction of each step building graph nodes in Python. Replaying a compiled graph removes that overhead and fuses elementwise chains into fewer GPU kernels. As matrix sizes grow, GPU compute dominates and the relative gain shrinks. The last row shows why the trace count matters: tracing costs more than the Python overhead it removes, so retracing every step is slower than not compiling at all. The very first call also pays a one-time startup cost (13 ms on the tiny model, and about a third of a second the first time the GPU builds fused kernels).

## Common mistakes

- **Not compiling**, or returning the uncompiled inner function by accident. This produces the right tokens but runs the model body 50 times in 50 calls.
- **Calling `mx.compile` inside `step`.** Each call to `mx.compile` creates a new compiled function with an empty trace cache, so every step traces again. Rebuilding and compiling a lambda or `functools.partial` inside the loop has the same effect.
- **Letting the token dtype drift.** Returning the raw `mx.argmax` output yields `uint32`. The second call sees a new dtype and triggers a second trace, and downstream code receives the wrong dtype.
- **Compiling without `inputs=model.state`.** Model weights are frozen at trace time. When the test replaces the parameters after 10 steps, the compiled function keeps emitting tokens from the old weights.
- **Branching on `token` with Python `if` inside the compiled function.** This raises on the first call because placeholder arrays cannot be evaluated during tracing.
- **Moving the branch into an uncompiled Python wrapper with `if token.item() == eos_id`.** Calling `.item()` forces an implicit evaluation in the middle of the caller's loop (the issue from `mlx-lazy-eval-placement`), and it raises whenever batch size `B > 1`.
- **Ignoring `eos_id`, or checking `best` instead of the input `token`.** The step that first emits `eos_id` looks right, but the following step resumes generating instead of repeating `eos_id`.
- **Calling `mx.eval` inside `step`.** Evaluating internally hides synchronization from the caller, preventing it from batching evaluation with its own state or scheduling with `mx.async_eval`.

One near miss fails loudly rather than silently: passing `inputs=model.parameters()` raises *"Attempting to compile a function with uncaptured inputs is not allowed"* on the first call. `model.parameters()` constructs a new tree on each call rather than returning the live container, so future weight updates would not appear in it anyway. Always pass `model.state`.

## Complexity

One trace per distinct input signature. Each subsequent call replays a fixed graph that executes one `model.step` per row, taking $O(\text{max\_len})$ attention work per layer with no Python overhead proportional to model depth.

## In the wild

The `inputs=` and `outputs=` arguments exist specifically for mutable state outside function arguments. In the MLX training example, the compiled training step passes `inputs=state, outputs=state` with `state = [model.state, optimizer.state]`, because each step both reads and updates parameters and optimizer buffers. The next problem, `mlx-compile-shapes`, uses `outputs=` for the same reason. Keep in mind that Python scalars also belong to the trace signature: calling a compiled function with `temperature=0.7` and then `0.8` traces twice, so any scalar that changes across calls should be passed as an array.
