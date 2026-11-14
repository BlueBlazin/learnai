Compile a function that picks the next token for prompts of many different lengths
and keeps a running penalty state, without retracing it for every length. A compiled
function is traced once per input shape, so a prompt of each new length would be a
new trace; and the state it reads and updates lives outside its arguments, where
`mx.compile` neither looks nor writes unless told to.

## Specification

```python
def make_sampler_step(model, state, penalty): ...  # returns step


token = step(prompt)
```

`model` is an [`MLXTinyLM`](../../../python/src/whetstone/kit/mlxlm.py), called on a
whole prompt:

```python
logits, _ = model(tokens)  # tokens: (T,) integer array; logits: (T, vocab_size)
```

The model is causal: row `i` of `logits` depends on `tokens[0]` to `tokens[i]` and on
nothing after them. `logits[T - 1]` scores the token that follows the prompt.

`state` is a dict owned by the caller, with one entry:

- `state["counts"]`: a float32 array of shape `(vocab_size,)`. `counts[v]` is how
  many times token `v` has been emitted so far.

`penalty` is a Python float, $\text{penalty} \ge 0$, fixed for the lifetime of `step`.

`make_sampler_step` returns a function `step`. A call `step(prompt)`, for an int32
array `prompt` of shape `(T,)` with $1 \le T \le 64$, does two things:

$$
\text{token} = \arg\max_v \bigl( \text{logits}_{T-1,\,v} - \text{penalty} \cdot \text{counts}_v \bigr),
\qquad
\text{counts}_{\text{token}} \leftarrow \text{counts}_{\text{token}} + 1 .
$$

- It returns `token`, an int32 array of shape `()`.
- It replaces `state["counts"]`, in the caller's dict, with the updated counts (a
  float32 array of shape `(vocab_size,)`). The update is visible to the caller as
  soon as `step` returns.
- `counts` on the right-hand side is whatever array `state["counts"]` holds **when
  `step` is called**. The caller may assign a different array to `state["counts"]`
  between calls (to reset the counts, or to ban a token), and the next call uses it.

**Rules.** The judge records every run of the model's Python body and observes
evaluation:

1. **Three buckets.** The model's Python body runs only on token arrays of length
   16, 32, or 64, and at most once for each of them over the lifetime of `step`. A
   prompt of length $T$ is processed at the smallest of the three sizes that is at
   least $T$. Forty prompts of twelve different lengths therefore cost at most three
   runs of the body; prompts of at most 16 tokens cost one, on 16 tokens.
2. **Nothing is evaluated.** `step` only builds a graph: no `mx.eval`, no `.item()`,
   `int(x)`, or `bool(x)` on an array. The caller evaluates the token and
   `state["counts"]` together after each call.
3. `make_sampler_step` does not run the model.

![Three prompts and their buckets. The prompt occupies the first $T$ positions; the logits are read at position $T - 1$, and nothing after that position can influence them.](assets/buckets.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | `(T,)`, $1 \le T \le 64$ | int32 |
| `state["counts"]` | `(vocab_size,)` | float32 |
| `token` (result) | `()` | int32 |

## Example

```python
model = MLXTinyLM(seed=1)  # vocab_size 64
state = {"counts": mx.zeros(64)}
step = make_sampler_step(model, state, penalty=1.5)

short = mx.array([0, 17, 42, 5], dtype=mx.int32)  # T = 4
long = mx.array(list(range(20, 40)), dtype=mx.int32)  # T = 20
```

| call | prompt | two largest logits at $T - 1$ | counts before | token | runs of the model's body so far |
|---|---|---|---|---|---|
| 1 | `short` | token 55: 5.174, token 20: 3.391 | all zero | 55 | 1 (on 16 tokens) |
| 2 | `short` | the same | `counts[55] = 1` | 55 | 1 |
| 3 | `long` | token 12: 4.762, token 26: 2.935 | `counts[55] = 2` | 12 | 2 (on 32 tokens) |
| 4 | `short` | token 55: 5.174, token 20: 3.391 | `counts[55] = 2`, `counts[12] = 1` | 20 | 2 |

In call 2 token 55 is penalized once: $5.174 - 1.5 = 3.674$ still beats $3.391$. In
call 4 it has been emitted twice: $5.174 - 3.0 = 2.174$ loses to token 20. After the
four calls `state["counts"]` is 2 at index 55, 1 at indices 12 and 20, and 0
elsewhere. Calls 1, 2, and 4 share the 16-token bucket and call 3 uses the 32-token
one: the model's Python code ran twice.

## Constraints

- $1 \le T \le 64$; prompts are int32 arrays of token ids in `[0, vocab_size)`. Token
  0 is an ordinary token and occurs in prompts.
- Tokens are compared exactly with the definition evaluated without compilation on
  the unpadded prompt. The tests keep the two largest adjusted logits more than
  `1e-3` apart. If two are equal, the argmax is the lower token id (this does not
  occur in the tests).
- `state["counts"]` is compared after every call, to an absolute tolerance of `1e-6`.
- The trace count is exact, and so are the traced lengths: a solution with the right
  tokens and four runs of the model's body, or one run on 64 tokens for a 3-token
  prompt, fails.
- At most 40 calls per `step`. Two `step` functions made for the same model keep
  separate states and are traced separately.

> [!NOTE]
> `mx.compile(..., shapeless=True)` does not help here: the model's body computes
> its positions and its mask from the length of `tokens` in Python, which a shapeless
> trace cannot follow. On this MLX version the first prompt of another length raises.

## Why this matters

Prompts never arrive at one length. A server that compiles its prefill and feeds it
raw prompts traces the model again for almost every request, and a call that traces
costs more than an uncompiled call (about 1.6 times, on an M1 Max), so the compiled
server is slower than the uncompiled one. Padding to a few bucket sizes bounds the
number of traces by the number of buckets, at the price of some padded positions; the
same trick serves every tracing compiler, JAX's `jit` included. The state is the
other half:
repetition penalties, sampler statistics, and random keys all live outside the
function's arguments. Captured silently, they are frozen at their trace-time values,
and the sampler runs without error while ignoring everything that happened since.
