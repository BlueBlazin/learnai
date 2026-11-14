Stream greedy generation from an MLX language model as a generator that yields each
token's text as soon as the token exists, and keep the GPU busy while it does. With
`mx.eval`, the GPU computes a token, then waits while Python detokenizes it and the
consumer handles it. With `mx.async_eval`, the next step is already running during
that time, provided it was scheduled **before** the current token was read.

## Specification

```python
def stream_generate(model, prompt, n, detok): ...
```

`model` is an [`MLXTinyLM`](../../../python/src/whetstone/kit/mlxlm.py), called as in
`mlx-lazy-eval-placement`; both forms return **lazy** arrays:

```python
logits, cache = model(tokens)  # no cache: `tokens` is the whole sequence so far
logits, cache = model(tokens, cache)  # with a cache: `tokens` continue the cached prefix
```

`tokens` is a 1-D integer array, `logits` has shape `(T, vocab_size)` with
`logits[-1]` scoring the next token, and `cache` is a list of `(keys, values)` pairs
that the model never modifies.

`stream_generate` is a **generator**. It decodes greedily: with $\ell$ the next-token
logits given the prompt and the tokens generated so far,

$$
\text{token} = \arg\max_v \ell_v ,
\qquad
\log p(\text{token}) = \ell_{\text{token}} - \log \sum_v e^{\ell_v} ,
$$

and for each of the `n` generated tokens, in order, it yields a pair
`(text, logprob)`:

- `text` is `detok(token)`, where `token` is the token id as a Python `int`. `detok`
  is called exactly once per token, in order, and returns a `str`.
- `logprob` is $\log p(\text{token})$ as a scalar float32 **array**. It is not read
  by the generator; when it is yielded it must already be scheduled or computed.

`prompt` is a list of $P \ge 1$ Python ints; $n \ge 1$; there is no stop token.

**Evaluation rules.** Number the steps and tokens from 0: step 0 is the model call on
the prompt and produces token 0; step $t + 1$ feeds token $t$ and produces token
$t + 1$. The judge observes the following.

1. Evaluation is requested with `mx.async_eval` only, exactly once per step, `n` times
   in all. `mx.eval` is never called.
2. Each token is read exactly once, with `.item()`, to obtain the int for `detok`.
   Nothing else is read (`.tolist()`, `int(x)`, `float(x)`, `np.array(x)`,
   `print(x)`): `n` reads in all.
3. **Step $t + 1$ is scheduled before token $t$ is read**, for every $t < n - 1$.
   The sequence of calls is therefore

   ```
   async_eval(step 0), async_eval(step 1), read token 0, async_eval(step 2), read token 1, ...
   ```

   ending with the reads of the last two tokens. No step is built or scheduled after
   the last token: when `detok` is called for token $t$, and when pair $t$ reaches
   the consumer, the model has been called exactly $\min(t + 2, n)$ times.
4. Every yielded `logprob` has 0 pending operations: it was covered by an
   `mx.async_eval`.

![The same three steps with blocking and with asynchronous evaluation. Scheduling step $t + 1$ before reading token $t$ lets the GPU compute it while Python turns token $t$ into text and the consumer uses it.](assets/overlap.svg)

**Cost limits**, as in `cached-greedy-decode`: `model.counters.forward_calls` is at
most $n$ and `model.counters.tokens_processed` is at most $P + n - 1$. If the consumer
stops after receiving $k < n$ pairs, the model has been called at most $k + 1$ times.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `model`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| argument of `detok` | scalar | Python `int` |
| `text` (yielded) | | Python `str` |
| `logprob` (yielded) | `()` | float32 |

## Example

```python
model = MLXTinyLM(seed=3)  # vocab_size 64
detok = lambda token: f"[{token}]"

for text, logprob in stream_generate(model, [0, 17, 42, 5], 8, detok):
    print(text, logprob.item())
# [28] -0.1615
# [34] -1.4226
# [31] -1.3849
# [15] -1.9084
# [1]  -1.9787
# [48] -1.8911
# [46] -2.1124
# [5]  -1.5798
```

What the judge sees, in order:

| call | model calls so far |
|---|---|
| `async_eval` of step 0 (the prompt, producing token 28) | 1 |
| `async_eval` of step 1 (feeds the array of token 28, producing token 34) | 2 |
| read token 0, `detok(28)`, yield `("[28]", -0.1615)` | 2 |
| `async_eval` of step 2 | 3 |
| read token 1, `detok(34)`, yield `("[34]", -1.4226)` | 3 |
| ... | |
| `async_eval` of step 7 (producing token 5) | 8 |
| read token 6, `detok(46)`, yield `("[46]", -2.1124)` | 8 |
| read token 7, `detok(5)`, yield `("[5]", -1.5798)` | 8 |

Eight `async_eval` calls, eight reads, eight model calls, 11 tokens processed. Step 1
can be built before token 0 is known as a number, because building a graph needs the
token only as an array.

## Constraints

- $1 \le P \le 64$, $1 \le n \le 64$.
- The texts and the tokens passed to `detok` are compared exactly with greedy decoding
  that uses no cache. The test models keep the two largest logits more than `1e-3`
  apart; if two logits were equal, the argmax would be the lower token id.
- Each `logprob` is compared with a float64 computation to a relative tolerance of
  `1e-4` and an absolute tolerance of `1e-3`.
- The evaluation rules and the cost limits are exact counts. A stream with the right
  texts that breaks one is rejected.
- "Pending operations" are counted with `mx.export_to_dot`. An array that has been
  passed to `mx.async_eval` has none from that moment on, whether or not its value is
  ready.

## Why this matters

A token is not finished when the GPU has computed it. Python still has to detokenize
it, check stop sequences, update the sampler's state, and hand the text to a socket or
a terminal, and on Apple silicon that work runs on the CPU while the GPU could be
doing something else. `mlx_lm.generate_step` is written exactly this way: it calls
`mx.async_eval` on step $t + 1$ and only then yields token $t$. The order is the whole
trick and it is easy to lose. Read the token first and the loop is correct, streams
fine, and runs with the GPU idle for the length of the Python work on every token: on
this machine that is the difference between 2.3 and 1.3 ms per token on a small model
with no detokenization cost at all. Schedule one step too many and every generation
ends with a forward pass nobody asked for.
