## The idea

`mx.compile(fn)` returns a function that behaves like `fn` and runs it far less often.
On a call it looks at the *signature* of the arguments: the shape and dtype of every
array, and the value of every Python scalar. For a signature it has not seen, it
**traces**: it calls `fn` with placeholder arrays, records the graph that `fn` builds,
simplifies and fuses it, and keeps the result. For a signature it has seen, it skips
Python entirely and replays the recorded graph on the new arrays.

So `fn` is ordinary Python that runs once per signature, and three things follow.

**1. Anything that changes the signature costs a trace.** The state's buffers have a
fixed `max_len`, so their shapes never change (that is what the fixed-shape interface
is for; the growing cache of `model(tokens, cache)` would have a new shape at every
step). The token is the trap: `mx.argmax` returns uint32. A step that receives int32
and returns uint32 is called with a new signature the second time, and traced twice.
Casting the result to the dtype of the input closes the loop.

**2. Values do not exist at trace time.** During the trace `token` is a placeholder.
`if token == eos_id:` needs a Python bool, so MLX would have to evaluate the
placeholder, and it raises: *"Attempting to eval an array during function
transformations like compile or vmap is not allowed."* A decision that depends on a
value must be part of the graph: `mx.where(token == eos_id, eos_id, best)` computes
both sides and selects per row. (`eos_id` itself is a Python int closed over by the
function. It is recorded as a constant, which is right: it never changes.)

**3. Arrays the function reads but does not receive are frozen.** `model.step` reads
the weights from the module. The trace sees them as arrays that are not placeholders,
and records their *current values* as constants of the graph. Replace the weights
later and the compiled function keeps computing with the old ones: no error, no
retrace, wrong tokens. `mx.compile(fn, inputs=tree)` declares such implicit inputs:
before every call the compiled function collects the arrays that `tree` holds *now*
and feeds them alongside the arguments. `model.state` is the module's own dict, so it
always holds the current parameters.

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

Arguments and results may be trees: the state dict goes in and comes out as it is.
`mx.where` with an int32 array and a Python int stays int32, because Python scalars
are weakly typed. The compiled function returns lazy arrays, like any MLX function;
the caller's `mx.eval(token, state)` is still the one evaluation of the step.

## What it buys

Measured on this machine (M1 Max, MLX 0.32.4), per step, after the first call, best of
four runs of 200 steps. "Tiny" is the default `MLXTinyLM`; "larger" has 6 layers of
width 256 and a vocabulary of 8192.

| step function | tiny | larger | traces in 201 calls |
|---|---|---|---|
| plain Python | 0.69 ms | 2.11 ms | (201 runs of the body) |
| compiled once | 0.61 ms | 1.58 ms | 1 |
| `mx.compile` called inside the step | 1.20 ms | 3.33 ms | 201 |

A model this small spends a large share of each step in Python, building a few
hundred graph nodes; replaying a recorded graph removes that, and the compiler fuses
runs of elementwise operations into single kernels. The gain shrinks as the matrices
grow, because the GPU time grows and the Python time does not. The last row is why
the trace count is the thing to watch: tracing costs more than the Python it saves,
so a compiled function that is retraced on every call is the slowest of the three.
The very first call is slower still (13 ms for the tiny model, and a third of a
second the first time the GPU has to build the fused kernels), which is the price of
admission, paid once.

## Common mistakes

- **Not compiling**, or returning the inner function by accident. Right tokens, 50
  runs of the body in 50 calls.
- **Calling `mx.compile` inside the step.** Each call of `mx.compile` creates a new
  compiled function with an empty cache, so every step traces. The same happens when
  a lambda or a `functools.partial` is rebuilt and compiled in the loop.
- **Letting the dtype drift.** Returning the raw `mx.argmax` gives uint32. The
  second call is a new signature: 2 traces, and every later consumer of the token
  sees a dtype it did not expect.
- **Compiling without `inputs=model.state`.** The weights become constants. The test
  swaps the parameters after 10 steps; this version keeps emitting the old model's
  tokens.
- **A Python branch on the token inside the compiled function.** It raises on the
  first call, with the message quoted above.
- **Moving the branch outside**, into a Python wrapper that does
  `if token.item() == eos_id`. Now every step is evaluated implicitly in the middle of
  the caller's loop (the lesson of `mlx-lazy-eval-placement`), and `.item()` on a
  batch of more than one row raises.
- **Ignoring `eos_id`**, or testing the *new* token instead of the one fed in. The
  step that emits the end of sequence looks right either way; the following step
  starts generating again.
- **Evaluating inside `step`.** A wrapper that calls `mx.eval` on the outputs hides
  an evaluation from the caller, who can then no longer evaluate the step together
  with its own bookkeeping, or schedule it with `mx.async_eval`.

One near miss is an error and not a silent bug: `inputs=model.parameters()` raises on
the first call on this MLX version (*"Attempting to compile a function with uncaptured
inputs is not allowed"*). `parameters()` builds a new tree on every call, a snapshot
that later updates of the module would not reach in any case. Give `mx.compile` the
live container: `model.state` is the module itself.

## Complexity

One trace per input signature. Every other call costs one replay of a fixed graph:
the work of `model.step` for one token per row, $O(\text{max\_len})$ per layer for the
attention over the fixed buffer, and no Python proportional to the size of the model.

## In the wild

`inputs=` and `outputs=` exist for exactly this kind of state. The MLX documentation's
training example compiles its step with `inputs=state, outputs=state` where
`state = [model.state, optimizer.state]`, because the step both reads and replaces
the parameters; the next problem, `mlx-compile-shapes`, uses `outputs=`. Python
scalars are part of the signature too: a compiled function called with
`temperature=0.7` and then `0.8` is traced twice, so a value that varies per call
should be passed as an array.
