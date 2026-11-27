## The idea

This problem combines two core behaviors of `mx.compile`.

**1. Traces are keyed by argument signatures.** A signature consists of the shapes and dtypes of array arguments and the *values* of Python scalar arguments. Passing a length-17 prompt after a length-20 prompt creates a new signature and retraces the model. To bound the number of signatures, round each prompt length `T` up to a bucket (`16`, `32`, or `64`), pad the prompt to that bucket size, and pass the true end index `T - 1` as an **array**. Because an array's value is runtime data rather than part of the signature, all prompts in the same bucket share one trace.

| what varies per call | passed as | part of the signature? |
|---|---|---|
| padded tokens | `int32` array of shape `(16,)`, `(32,)`, or `(64,)` | shape only (3 values) |
| position `T - 1` | `int32` scalar array | no (shape `()` and dtype only) |
| position `T - 1` | Python `int` | **yes** (one trace per distinct value) |

Right-padding does not change the result because the model is causal. Row `T - 1` of `logits` depends only on tokens `0` through `T - 1`, and later positions are masked out of its attention. Padding on the right with any valid token id and reading row `T - 1` matches the unpadded prompt up to floating-point rounding.

**2. Compiled functions expect pure inputs and outputs.** Here, the body both reads `state["counts"]` and assigns a new array to it. Without declaring `state`, tracing breaks in two ways:

- Reading an undeclared array freezes its trace-time value as a constant.
- Assigning to an external container runs only during tracing and stores a trace placeholder that cannot be evaluated outside the trace.

Calling `mx.compile(fn, inputs=state, outputs=state)` turns both accesses into real graph inputs and outputs. Before each call, MLX reads the current arrays from `state` and feeds them into the graph. After each call, MLX writes the corresponding output arrays back into `state`. Because `state` itself is the live container, any new array the caller puts in `state["counts"]` between calls is picked up automatically.

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

The outer `step` wrapper is plain Python and runs on every call. Reading `prompt.shape[0]` inspects metadata only and does not evaluate the array. Inside `pick`, `adjusted.shape[0]` is the constant vocabulary size, and `penalty` is a fixed Python `float` in the closure, so baking both into the trace is safe. `model.state` is included in `inputs` for the same reason as in `mlx-compile-step`.

An equally valid design passes `counts` explicitly: `pick(tokens, last, counts)` returns `(token, new_counts)`, and the wrapper assigns `state["counts"] = new_counts`. That inner function is pure and needs neither `inputs=state` nor `outputs=state`. Explicit threading is clean for a single array, while `inputs=`/`outputs=` scales better to large nested trees like models, optimizers, and random state.

## What a trace costs

Measured on an M1 Max (MLX 0.32.4) for one call on a prompt of 20 to 43 tokens, taking the minimum over repeated runs. "Tiny" is the default `MLXTinyLM`. "Larger" has 6 layers, width 256, and vocabulary size 8192.

| call | tiny | larger |
|---|---|---|
| compiled, replaying a trace | 0.63 ms | 1.45 ms |
| not compiled | 0.75 ms | 1.73 ms |
| compiled, tracing (a new length) | 1.17 ms | 2.92 ms |

A tracing call costs about 1.6 times an uncompiled call. A compiled function that encounters new lengths on most calls is therefore slower than plain Python. With three buckets, tracing happens at most three times over the lifetime of `step`.

Bucketing trades traces for padded positions. With buckets `16`, `32`, and `64`, any prompt of length 16 or more runs on at most twice its length (for instance, 17 tokens pad to 32). A single bucket of 64 traces only once but forces a 3-token prompt to pay for 64 positions on every call. Powers of two strike a practical balance.

## Why not `shapeless=True`

`mx.compile(fn, shapeless=True)` reuses one trace across all input shapes, but it is valid only when the graph structure does not depend on concrete shape integers in Python. Any Python arithmetic on `x.shape` breaks it silently:

```python
mean_rows = mx.compile(lambda x: x.sum(axis=0) / x.shape[0], shapeless=True)
mean_rows(mx.ones((4, 2)))  # [1, 1]
mean_rows(mx.ones((8, 2)))  # [2, 2]: still divides by 4
```

Writing `x.mean(axis=0)` avoids the bug because the reduction handles the axis length inside the operation. Here, `MLXTinyLM` builds `mx.arange(T)` for positions, constructs a `T` by `T` causal mask, and reshapes using `T`. Under `shapeless=True`, the first prompt of a different length raises `[Primitive::output_shapes] Slice cannot infer output shapes`. Shapeless compilation works best for elementwise operations, reductions, and matrix multiplications without shape-dependent Python control flow.

## Common mistakes

- **Compiling without declaring `state`.** Assigning to `state["counts"]` inside the traced function leaves a placeholder in the dict. Evaluating it raises *"Attempting to eval an array without a primitive. If you are compiling a function, make sure all the inputs and outputs are captured."*
- **Passing `outputs=state` without `inputs=state`.** This fails silently. Updated counts are written back after each call, but every call reads the trace-time constant (all zeros). The penalty never accumulates, so in the worked example call 4 returns `55` instead of `20`.
- **Capturing `state["counts"]` once at setup time** and threading a private copy through calls. This works until the caller replaces `state["counts"]` with a new array, which the function ignores.
- **Skipping padding.** Each distinct prompt length triggers a new trace, producing 12 traces in the 40-prompt test.
- **Using `shapeless=True` instead of padding.** Raises on the second distinct prompt length, as shown above.
- **Passing `T - 1` as a Python `int`.** Even with padded shapes in three buckets, the Python `int` becomes part of the signature and causes 12 traces again.
- **Reading `logits[-1]` instead of `logits[T - 1]`.** This reads the last padding position and is only right when the prompt fills its bucket (lengths 16, 32, or 64).
- **Padding on the left.** Left-padding puts the prompt's last token at index `-1`, but it shifts every prompt token's position and lets prompt tokens attend to the padding. Left-padding a causal model requires an attention mask and position offsets, whereas right-padding needs neither.
- **Using a single bucket of 64.** This yields one trace, but fails the rule that short prompts must run in their smallest bucket.
- **Inferring length by counting nonzero tokens.** Token `0` is a valid token, so prompts containing or ending in `0` get truncated.

## Complexity

At most three traces over the lifetime of `step`. Each call runs the model on at most $\max(16, 2T - 1)$ positions, plus $O(\text{vocab\_size})$ work for the penalty subtraction, argmax, and count update.

## In the wild

The MLX documentation's training example compiles its step with `inputs=state, outputs=state` for `state = [model.state, optimizer.state]`, and adds `mx.random.state` whenever the step draws random numbers. A random key is read and replaced on every call, so compiling without it makes a sampler repeat the same "random" draws forever. Bucketed padding is the standard way to handle variable sequence lengths in shape-keyed compilers, from JAX `jit` to static-shape inference engines.
