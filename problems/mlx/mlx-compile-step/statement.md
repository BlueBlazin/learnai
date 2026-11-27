Compile the greedy decode step of a transformer with `mx.compile` so that repeated calls replay a single traced graph instead of running the model's Python code on every step. Your compiled step must handle end-of-sequence tokens without Python control flow, keep input and output dtypes stable across steps, and pick up updated model weights without retracing.

## Specification

```python
def make_step(model, eos_id): ...  # returns step


next_token, state = step(token, state)
```

`model` is an [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py) used through its fixed-shape decode interface:

```python
state = model.init_state(max_len, batch_size)  # created by the caller
logits, state = model.step(token, state)       # advances one token per row
```

- `token` holds the current token id for each of the `B` rows in the batch.
- `state` is a dict `{"keys": [...], "values": [...], "length": ...}`. It stores one key buffer and one value buffer per transformer layer, each allocated up to `max_len` positions, along with a scalar `length` counting how many tokens have been written so far. Calling `model.step` writes the new token at position `length`, increments `length` by 1, and returns a new dict with the same structure and shapes. Buffer shapes never depend on `length`.
- `logits` has shape `(B, vocab_size)` and holds the next-token scores for each row.

`make_step(model, eos_id)` returns a function `step`. The argument `eos_id` is a Python `int` that stays fixed for the lifetime of `step`.

Given `token` and `state`, `step(token, state)` runs `model.step(token, state)` and selects the next token for each row `b`. If row `b` received `eos_id`, it has finished and emits `eos_id` again. Otherwise, it emits the token `v` with the largest score in `logits[b]`:

$$
\text{next\_token}_b =
\begin{cases}
\text{eos\_id} & \text{if } \text{token}_b = \text{eos\_id}, \\
\arg\max_v \text{logits}_{b,v} & \text{otherwise}.
\end{cases}
$$

- `next_token` must always have dtype `int32`.
- Finished rows still pass through `model.step` so their state advances like every other row. Only the emitted token is replaced with `eos_id`.
- `step` returns `(next_token, state)`, where `state` is the dict returned by `model.step`.

### Rules

The judge counts executions of the model's Python body and checks for premature evaluation.

1. **Traced once per input shape.** Over any number of calls with the same batch size `B` and `max_len`, the Python body of `model.step` may run at most once (on the first call). Feeding `next_token` back as the next `token` must not trigger a second trace. A call with a new `B` or `max_len` may trigger one trace for that new shape. `make_step` itself must not run the model.
2. **Current parameters.** `step` must use whatever parameters `model` holds at call time. After `model.update(...)` or `model.load_weights(...)` replaces the weights, the next call must compute with the new values and still obey rule 1 with no new trace.
3. **No evaluation inside `step`.** Neither `make_step` nor `step` may evaluate any array. Do not call `mx.eval`, `.item()`, `int(x)`, or `bool(x)` on an array. The caller evaluates `next_token` and `state` together once per step.

## Shapes

Here `B` is the batch size, `n_kv_heads` is the number of key-value heads, `max_len` is the cache capacity, and `head_dim` is the per-head dimension.

| name | shape | dtype |
|---|---|---|
| `token` | `(B,)` | `int32` |
| `state["keys"][i]`, `state["values"][i]` | `(B, n_kv_heads, max_len, head_dim)` | `float32` |
| `state["length"]` | `()` | `int32` |
| `next_token` (result) | `(B,)` | `int32` |
| `state` (result) | same as input `state` | same as input `state` |

## Example

```python
model = MLXTinyLM(seed=1)  # vocab_size = 64
step = make_step(model, eos_id=32)

token = mx.array([3], dtype=mx.int32)
state = model.init_state(max_len=64, batch_size=1)
for _ in range(8):
    token, state = step(token, state)
    mx.eval(token, state)
```

| call | `token` in | argmax of `logits` | `next_token` | `state["length"]` after | runs of model body so far |
|---|---|---|---|---|---|
| 1 | 3 | 7 | 7 | 1 | 1 |
| 2 | 7 | 48 | 48 | 2 | 1 |
| 3 | 48 | 0 | 0 | 3 | 1 |
| 4 | 0 | 53 | 53 | 4 | 1 |
| 5 | 53 | 32 | 32 | 5 | 1 |
| 6 | 32 | (ignored) | 32 | 6 | 1 |
| 7 | 32 | (ignored) | 32 | 7 | 1 |
| 8 | 32 | (ignored) | 32 | 8 | 1 |

Call 5 emits token `32` because it is the argmax. Starting on call 6, the input token is `32` (`eos_id`), so `step` emits `32` regardless of the logits. Call 1 traces `step`, and calls 2 through 8 replay the compiled graph, so the model's Python body runs once in total. Every returned token has dtype `int32`, even though `mx.argmax` returns `uint32`.

## Constraints

- $1 \le B \le 4$. `max_len` is `16` or `64` and is never exceeded. Tests make up to 50 calls.
- Output tokens must match an uncompiled loop exactly. Test models keep the top two logits of every unfinished row more than `1e-3` apart. If two logits tie, argmax selects the lower token id (ties do not occur in the tests).
- The final `state` arrays are checked with relative tolerance `1e-4` and absolute tolerance `1e-5`. `state["length"]` is checked exactly.
- The trace count is exact. `model.counters.forward_calls` (reported as `traces`) must be at most 1 per input shape. A solution that returns the right tokens with 2 traces fails.
- `model.state` is the module's live dict of all parameters and buffers. By contrast, `model.parameters()` builds a fresh tree on every call containing the same arrays.

## Why this matters

A single decode step consists of a few hundred small operations. At small model sizes, building that graph in Python can take almost as long as running it on the GPU. On an M1 Max, a 6-layer, 256-wide model takes 2.1 ms per step uncompiled and 1.6 ms compiled.

Because `mx.compile` gets this speedup by tracing once and replaying, misuse fails silently. A step that retraces on every call (because of a drifting dtype, a growing cache, or a new compiled wrapper) runs at 3.3 ms, slower than uncompiled Python. A step that captures model weights as trace-time constants keeps generating from the old checkpoint even after you swap weights or load a LoRA adapter, with no error raised.
