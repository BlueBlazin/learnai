## The idea

Two properties of `mx.compile` meet in this problem.

**The cache of traces is keyed by the signature of the arguments**: shapes and dtypes
of arrays, and the *values* of Python scalars. A prompt of length 17 after a prompt of
length 20 is a new signature, and the model's Python body runs again. The way out is
to make the signature take few values. Round the length up to a bucket, pad the
prompt to it, and pass where the prompt really ends as an **array**, whose value is
data and not part of the key:

| what varies per call | passed as | part of the signature? |
|---|---|---|
| the padded tokens | int32 array of shape `(16,)`, `(32,)`, or `(64,)` | the shape: three values |
| the position `T - 1` | int32 scalar array | no: only its shape `()` and dtype |
| the position `T - 1` | Python int | **yes**: one trace per distinct value |

Padding is free of side effects because the model is causal. Row `T - 1` of the
logits is computed from tokens `0` to `T - 1`; whatever sits to the right of it is
masked out of its attention. So pad on the right, with any token, and read row
`T - 1`. The result is the same as on the unpadded prompt, up to float rounding.

**A compiled function is meant to be pure**, and this one is not: it reads
`state["counts"]` and assigns to it. What the trace does with that:

- *Reading* an array that is not an argument records its current value as a constant.
- *Assigning* inside the function runs once, during the trace, and stores a
  placeholder: an array that belongs to the trace and has no value outside it.

`mx.compile(fn, inputs=state, outputs=state)` turns both into real inputs and outputs
of the compiled graph. Before each call the arrays in `state` are collected and fed
in; after each call the arrays that `fn` left in `state` during the trace are
replaced by the corresponding results of this call. The dict is the interface: the
caller can put a new array into it, and the next call reads that.

## Reference solution (MLX)

```python
BUCKETS = (16, 32, 64)


def make_sampler_step(model, state, penalty):
    def pick(tokens, last):
        logits, _ = model(tokens)
        adjusted = mx.take(logits, last, axis=0) - penalty * state["counts"]
        token = mx.argmax(adjusted).astype(mx.int32)
        state["counts"] = state["counts"] + (mx.arange(adjusted.shape[0]) == token)
        return token

    compiled = mx.compile(pick, inputs=[model.state, state], outputs=state)

    def step(prompt):
        length = prompt.shape[0]
        bucket = next(size for size in BUCKETS if size >= length)
        padded = mx.pad(prompt, (0, bucket - length))
        return compiled(padded, mx.array(length - 1))

    return step
```

The wrapper is ordinary Python and runs on every call; it only reads a shape, which
needs no evaluation. `adjusted.shape[0]` inside `pick` is the vocabulary size, a
constant, so baking it into the trace is harmless. `penalty` is a Python float in a
closure, also a constant, also harmless: it never changes. `model.state` is among the
inputs for the reason given in `mlx-compile-step`.

An equally good solution passes the counts explicitly: `pick(tokens, last, counts)`
returns `(token, new_counts)`, and the wrapper does `state["counts"] = new_counts`.
That function is pure, so it needs no `inputs=` or `outputs=` for the counts. Use the
declared form when the state is large or deeply nested (a model, an optimizer, the
random state) and the explicit form when it is one array.

## What a trace costs

Measured on this machine (M1 Max, MLX 0.32.4), one call on a prompt of 20 to 43
tokens, minimum over repeated runs. "Tiny" is the default `MLXTinyLM`; "larger" has 6
layers of width 256 and a vocabulary of 8192.

| call | tiny | larger |
|---|---|---|
| compiled, replaying a trace | 0.63 ms | 1.45 ms |
| not compiled | 0.75 ms | 1.73 ms |
| compiled, tracing (a new length) | 1.17 ms | 2.92 ms |

A call that traces costs about 1.6 times an uncompiled call, so a compiled function
that sees a new length on most calls loses to no compilation at all. With three
buckets the function traces three times in its life.

Buckets cost padded positions instead. With sizes 16, 32, 64 a prompt is at most
doubled (17 tokens run as 32), except below 16. One bucket of 64 would trace once and
make a 3-token prompt pay for 64 positions on every call; a bucket per length is the
unpadded function again. Powers of two are the usual compromise.

## Why not `shapeless=True`

`mx.compile(fn, shapeless=True)` keeps one trace for all shapes, and is right only if
the graph does not depend on the shapes it was traced with. Anything computed in
Python from `x.shape` is such a dependence. A two-line example gives wrong numbers
without any error:

```python
mean_rows = mx.compile(lambda x: x.sum(axis=0) / x.shape[0], shapeless=True)
mean_rows(mx.ones((4, 2)))  # [1, 1]
mean_rows(mx.ones((8, 2)))  # [2, 2]: still divides by 4
```

Written as `x.mean(axis=0)`, with the length left to the operation, it returns
`[1, 1]` both times. The model here is far from that: it builds `mx.arange(T)` for
the positions and a `T` by `T` mask, and reshapes with `T`. Compiled shapeless, the
first prompt of a different length raises `[Primitive::output_shapes] Slice cannot
infer output shapes`. Shapeless compilation suits bodies made of elementwise
operations, reductions, and matrix products; a sampler that takes logits of a fixed
vocabulary size has no varying shape to begin with.

## Common mistakes

- **Compiling without declaring the state.** The assignment in the body leaves a
  placeholder in the dict; evaluating it fails with *"Attempting to eval an array
  without a primitive. If you are compiling a function, make sure all the inputs and
  outputs are captured."*
- **`outputs=state` without `inputs=state`.** The quiet version. The counts are
  written back after every call, but read as the constant they were at trace time
  (all zeros). Every call therefore returns "zeros plus this call's token": the
  penalty never takes effect and the counts never accumulate. In the worked example
  call 4 returns 55 instead of 20.
- **Reading the counts once, when the sampler is made**, and threading a private copy
  through the calls. Correct until the caller assigns a new array to
  `state["counts"]`, which this version never looks at.
- **No padding.** One trace per distinct length: 12 in the 40-prompt case.
- **`shapeless=True` instead of padding.** Raises on the second length, as above.
- **Passing `T - 1` as a Python int.** The padding is in place and the shapes take
  three values, but the int is part of the signature: 12 traces again.
- **Reading `logits[-1]`.** That is the row of the last *padding* token. It is right
  exactly when the prompt fills its bucket (lengths 16, 32, 64).
- **Padding on the left.** Then the last row is the prompt's last token, but every
  prompt token has moved to a later position and attends to the padding before it.
  Left padding needs an attention mask and position offsets; right padding of a causal
  model needs neither.
- **One bucket of 64.** A single trace, and the wrong trade: the judge sees a 64-token
  trace for a run of short prompts.
- **Finding the length by counting nonzero tokens.** Token 0 is a token. A prompt
  that contains or ends in zeros is cut short.

## Complexity

At most three traces. Each call runs the model on at most $\max(16, 2T - 1)$
positions, plus $O(V)$ for the penalty, the argmax, and the count update.

## In the wild

The MLX documentation's own example of captured state is the training step compiled
with `inputs=state, outputs=state` for `state = [model.state, optimizer.state]`, and
it adds `mx.random.state` to that list whenever the function draws random numbers:
the random key is exactly this kind of state, read and replaced on every call. A
sampler compiled without it draws the same "random" numbers forever. Bucketed padding
is the standard answer to shape-keyed compilation everywhere, from JAX's `jit` to
static-shape inference engines.
