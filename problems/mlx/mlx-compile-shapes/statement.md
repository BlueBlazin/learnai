Compile a sampling step that picks the next token for variable-length prompts while updating an external repetition-penalty table, without retracing for every new prompt length. Your solution must bucket prompt lengths to bound traces and declare mutable external state so `mx.compile` reads and updates it on every call.

## Specification

```python
def make_sampler_step(model, state, penalty): ...  # returns step


token = step(prompt)
```

`model` is an [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py) invoked on a full prompt of length `T`:

```python
logits, _ = model(tokens)  # tokens: (T,) int array; logits: (T, vocab_size)
```

The model is causal, so row `i` of `logits` depends only on `tokens[0]` through `tokens[i]` and never on later positions. In particular, `logits[T - 1]` scores the token that follows the prompt.

`state` is a dict owned by the caller with one key, `"counts"`. Entry `counts[v]` records how many times token `v` has been emitted so far. `penalty` is a Python `float` with $\text{penalty} \ge 0$, fixed for the lifetime of `step`.

`make_sampler_step(model, state, penalty)` returns a function `step`. Given a prompt of length `T`, `step(prompt)` subtracts the repetition penalty from the last-position logits, picks the highest-scoring token `v`, and increments that token's count by 1:

$$
\text{token} = \arg\max_v \bigl( \text{logits}_{T-1,\,v} - \text{penalty} \cdot \text{counts}_v \bigr),
\qquad
\text{counts}_{\text{token}} \leftarrow \text{counts}_{\text{token}} + 1 .
$$

- `step` returns `token` as a scalar `int32` array.
- `step` replaces `state["counts"]` in the caller's dict with the updated `float32` counts array. The update must be visible in `state["counts"]` as soon as `step` returns.
- Each call must read whatever array `state["counts"]` currently holds when `step` is called. Between calls, the caller may assign a new array to `state["counts"]` (for example, to reset counts or ban a token), and the next call must use it.

### Rules

The judge records every execution of the model's Python body and checks for premature evaluation.

1. **Three buckets.** The model's Python body may run only on token arrays of length `16`, `32`, or `64`, and at most once per bucket over the lifetime of `step`. A prompt of length `T` must use the smallest bucket size that is at least `T`. Forty prompts spanning twelve lengths therefore trigger at most three traces, and prompts of length at most 16 trigger only one trace (on length 16).
2. **No evaluation inside `step`.** `step` only builds a graph. Do not call `mx.eval`, `.item()`, `int(x)`, or `bool(x)` on an array. The caller evaluates `token` and `state["counts"]` together after each call.
3. **Lazy setup.** `make_sampler_step` must not run the model.

![Three prompts and their buckets. The prompt occupies the first $T$ positions; the logits are read at position $T - 1$, and nothing after that position can influence them.](assets/buckets.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | `(T,)`, $1 \le T \le 64$ | `int32` |
| `state["counts"]` | `(vocab_size,)` | `float32` |
| `token` (result) | `()` | `int32` |

## Example

```python
model = MLXTinyLM(seed=1)  # vocab_size = 64
state = {"counts": mx.zeros(64)}
step = make_sampler_step(model, state, penalty=1.5)

short = mx.array([0, 17, 42, 5], dtype=mx.int32)  # T = 4
long = mx.array(list(range(20, 40)), dtype=mx.int32)  # T = 20
```

| call | prompt | two largest logits at $T - 1$ | counts before | token | runs of model body so far |
|---|---|---|---|---|---|
| 1 | `short` | token 55: 5.174, token 20: 3.391 | all zero | 55 | 1 (on 16 tokens) |
| 2 | `short` | same | `counts[55] = 1` | 55 | 1 |
| 3 | `long` | token 12: 4.762, token 26: 2.935 | `counts[55] = 2` | 12 | 2 (on 32 tokens) |
| 4 | `short` | token 55: 5.174, token 20: 3.391 | `counts[55] = 2`, `counts[12] = 1` | 20 | 2 |

On call 2, token `55` is penalized once, and $5.174 - 1.5 = 3.674$ still beats `3.391`. On call 4, token `55` has been emitted twice, so $5.174 - 3.0 = 2.174$ loses to token `20`. After all four calls, `state["counts"]` is `2` at index `55`, `1` at indices `12` and `20`, and `0` elsewhere. Calls 1, 2, and 4 share the 16-token bucket while call 3 uses the 32-token bucket, so the model's Python body runs twice in total.

## Constraints

- $1 \le T \le 64$. Prompts are `int32` arrays with token ids in `[0, vocab_size)`. Token `0` is a valid token and appears in test prompts.
- Output tokens must match an uncompiled run on the unpadded prompt exactly. Tests keep the top two adjusted logits more than `1e-3` apart. If two scores tie, argmax selects the lower token id (ties do not occur in the tests).
- `state["counts"]` is checked after every call with an absolute tolerance of `1e-6`.
- Both the trace count and the traced lengths are checked exactly. A solution that produces the right tokens with four traces, or that traces on 64 tokens for a 3-token prompt, fails.
- At most 40 calls per `step`. Two `step` functions created for the same model keep separate states and are traced separately.

> [!NOTE]
> `mx.compile(..., shapeless=True)` does not work here. The model computes position indices and its causal mask from the length of `tokens` in Python, which a shapeless trace cannot track. On this MLX version, the first prompt of a different length raises an error.

## Why this matters

Real prompts arrive at arbitrary lengths. A server that compiles prefill directly on raw prompts retraces on almost every request. Because a tracing call costs about 1.6 times as much as an uncompiled call on an M1 Max, retracing makes compilation slower than plain Python. Padding prompts to a small set of bucket sizes caps the number of traces at the cost of some extra positions, a technique used across tracing compilers including JAX `jit`.

Mutable external state is the other common pitfall. Repetition penalties, sampler statistics, and random keys all live outside a function's arguments. If they are captured implicitly during tracing, they freeze at their initial values and the sampler silently ignores all subsequent updates.
