Compile the greedy decode step of a transformer with `mx.compile`, so that 50 steps
run the model's Python code once instead of 50 times. Compiling is one line. Keeping
the compiled function *valid* is the work: it is traced again whenever an input
changes shape or dtype, it freezes every array it reads without being told, and a
Python `if` on a token's value cannot be traced at all.

## Specification

```python
def make_step(model, eos_id): ...  # returns step


next_token, state = step(token, state)
```

`model` is an [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py), used through
its fixed-shape interface:

```python
state = model.init_state(max_len, batch_size)  # made by the caller
logits, state = model.step(token, state)  # one token per row
```

- `token` has shape `(B,)`, int32: the newest token of each of the `B` rows.
- `state` is a dict `{"keys": [...], "values": [...], "length": ...}`: per layer one
  buffer of keys and one of values of shape `(B, n_kv_heads, max_len, head_dim)`, and
  the number of tokens stored so far as an int32 scalar array. `model.step` returns a
  new dict of the same structure and the same shapes with the token written at
  position `length` and `length + 1` stored. Shapes never depend on how many tokens
  have been fed.
- `logits` has shape `(B, vocab_size)`: the scores of each row's next token.

`make_step(model, eos_id)` returns a function `step`. A call
`step(token, state)` returns `(next_token, state)`:

$$
\text{next\_token}_b =
\begin{cases}
\text{eos\_id} & \text{if } \text{token}_b = \text{eos\_id}, \\
\arg\max_v \text{logits}_{b,v} & \text{otherwise},
\end{cases}
$$

- `next_token` has shape `(B,)` and dtype **int32**, on every call.
- A row that is fed `eos_id` has finished: it emits `eos_id` again, forever. Its
  state still advances like any other row's (the model is run on it; only the
  emitted token is replaced).
- The returned `state` is the one `model.step` returned.
- `eos_id` is a Python int, fixed for the lifetime of `step`.

**Rules.** The judge counts runs of the model's Python body and observes evaluation:

1. **Traced once per input shape.** Over any number of calls with the same batch size
   and the same `max_len`, the Python body of `model.step` runs at most once: on the
   first call. Feeding `next_token` back as the next `token` must not cause a second
   run. A call with a new batch size or a new `max_len` may run it once more.
   `make_step` itself does not run the model.
2. **Current parameters.** `step` uses the parameters `model` has when `step` is
   called. After `model.update(...)` or `model.load_weights(...)` replaces them, the
   next call computes with the new values, and rule 1 still holds: no new trace.
3. **Nothing is evaluated.** `step` only builds a graph: no `mx.eval`, no
   `.item()`, `int(x)`, or `bool(x)` on an array, in `make_step` or in `step`. The
   caller evaluates the token and the state together, once per step.

## Shapes

| name | shape | dtype |
|---|---|---|
| `token` | `(B,)` | int32 |
| `state["keys"][i]`, `state["values"][i]` | `(B, n_kv_heads, max_len, head_dim)` | float32 |
| `state["length"]` | `()` | int32 |
| `next_token` (result) | `(B,)` | int32 |
| `state` (result) | as the input state | |

## Example

```python
model = MLXTinyLM(seed=1)  # vocab_size 64
step = make_step(model, eos_id=32)

token = mx.array([3], dtype=mx.int32)
state = model.init_state(max_len=64, batch_size=1)
for _ in range(8):
    token, state = step(token, state)
    mx.eval(token, state)
```

| call | `token` in | argmax of the logits | `next_token` | `state["length"]` after | runs of the model's Python body so far |
|---|---|---|---|---|---|
| 1 | 3 | 7 | 7 | 1 | 1 |
| 2 | 7 | 48 | 48 | 2 | 1 |
| 3 | 48 | 0 | 0 | 3 | 1 |
| 4 | 0 | 53 | 53 | 4 | 1 |
| 5 | 53 | 32 | 32 | 5 | 1 |
| 6 | 32 | (ignored) | 32 | 6 | 1 |
| 7 | 32 | (ignored) | 32 | 7 | 1 |
| 8 | 32 | (ignored) | 32 | 8 | 1 |

Call 5 emits token 32 because it is the argmax. From call 6 on the row is fed 32, the
end-of-sequence token, and emits it whatever the logits say. The first call traced the
function; calls 2 to 8 replayed the recorded graph with new arrays, so the model's
Python code ran once. All eight tokens are int32, although `mx.argmax` returns uint32.

## Constraints

- $1 \le B \le 4$; `max_len` is 16 or 64 and is never exceeded; up to 50 calls.
- Tokens are compared exactly with the same loop run without `mx.compile`. The test
  models keep the two largest logits of every unfinished row more than `1e-3` apart.
  If two logits are equal, the argmax is the lower token id (this does not occur in
  the tests).
- The final state is compared to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-5`; its `length` exactly.
- The trace count is exact: `model.counters.forward_calls`, reported as `traces`, is
  at most 1 per input shape. A solution with the right tokens and 2 traces fails.
- `model.state` is the module's live dict of everything it holds, parameters
  included; `model.parameters()` is a new tree, built on each call, of the same
  arrays.

## Why this matters

A decode step is a few hundred small operations, and at this size the time goes to
building the graph in Python as much as to the GPU: on an M1 Max the step of a
6-layer, 256-wide model takes 2.1 ms uncompiled and 1.6 ms compiled. `mx.compile`
buys that by recording the graph once, which is also how it goes wrong. A compiled
step that is retraced on every call (a new function object each time, a dtype that
drifts, a cache that grows) is slower than no compilation: 3.3 ms. One that captured
the weights at trace time keeps generating from the old model after you load a LoRA
adapter or swap a checkpoint, without any error. Both mistakes produce a program that
runs and returns tokens.
