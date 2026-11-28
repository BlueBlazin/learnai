Implement a generator that streams greedy text generation from an MLX language model while keeping the GPU busy with `mx.async_eval`. Unlike blocking `mx.eval`, which leaves the GPU idle while Python detokenizes and yields each token, scheduling the next step with `mx.async_eval` before reading the current token overlaps GPU compute with Python work.

## Specification

```python
def stream_generate(model, prompt, n, detok): ...
```

`model` is an [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py), called as in `mlx-lazy-eval-placement`. Both call forms return **lazy** arrays:

```python
logits, cache = model(tokens)  # no cache: `tokens` is the whole sequence so far
logits, cache = model(tokens, cache)  # with a cache: `tokens` continue the cached prefix
```

Here `tokens` is a 1-D integer array of length `T`. The returned `logits` array has shape `(T, vocab_size)`, where `logits[-1]` scores the next token, and `cache` is a list of `(keys, values)` pairs that the model never modifies.

`stream_generate` is a **generator** that decodes greedily. Let $\ell$ be the next-token logits vector (of length `vocab_size`) produced from the prompt and all tokens generated so far, with entries $\ell_v$ for each vocabulary index $v$. At each step, the generator selects the highest-scoring token and its log-probability:

$$
\text{token} = \arg\max_v \ell_v ,
\qquad
\log p(\text{token}) = \ell_{\text{token}} - \log \sum_v e^{\ell_v} .
$$

For each of the `n` generated tokens, in order, yield a pair `(text, logprob)`:

- `text` is `detok(token)`, where `token` is the token id as a Python `int`. Call `detok` exactly once per token, in order. It returns a `str`.
- `logprob` is $\log p(\text{token})$ as a scalar float32 MLX array. Do not read its value inside the generator, and make sure it is already scheduled or computed before yielding it.

`prompt` is a list of $P \ge 1$ Python ints. `n` is a positive integer ($n \ge 1$). There is no stop token.

**Evaluation rules.** Number steps and tokens starting from 0. Step 0 runs the model on `prompt` and produces token 0. Step $t + 1$ feeds token $t$ into the model and produces token $t + 1$. The judge enforces four rules:

1. Request evaluation only with `mx.async_eval`, called exactly once per step (`n` times total). Never call `mx.eval`.
2. Read each token exactly once using `.item()` to get the Python `int` for `detok`. Do not read any other array (no `.tolist()`, `int(x)`, `float(x)`, `np.array(x)`, or `print(x)`), for `n` reads total.
3. **Schedule step $t + 1$ before reading token $t$**, for every $t < n - 1$. The required call order is:

   ```
   async_eval(step 0), async_eval(step 1), read token 0, async_eval(step 2), read token 1, ...
   ```

   This ends with reading the last two tokens. Do not build or schedule any step after the last token. When `detok` runs for token $t$, and when pair $t$ is yielded to the consumer, the model must have been called exactly $\min(t + 2, n)$ times.
4. Every yielded `logprob` must have 0 pending operations, meaning it was included in an `mx.async_eval` call.

![The same three steps with blocking and with asynchronous evaluation. Scheduling step $t + 1$ before reading token $t$ keeps the GPU busy while Python detokenizes.](assets/overlap.svg)

**Cost limits.** As in `cached-greedy-decode`, `model.counters.forward_calls` must be at most $n$ and `model.counters.tokens_processed` must be at most $P + n - 1$. If the consumer stops after receiving $k < n$ pairs, the model must have been called at most $k + 1$ times.

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

The judge observes these events in order:

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

In total, there are 8 `async_eval` calls, 8 reads, 8 model calls, and 11 tokens processed. Step 1 can be built before token 0's numeric value is known because constructing the next graph only needs the token as an MLX array.

## Constraints

- $1 \le P \le 64$, $1 \le n \le 64$.
- Yielded texts and token ids passed to `detok` must match uncached greedy decoding exactly. Test models keep the top two logits more than `1e-3` apart. If two logits were ever equal, argmax selects the lower token id.
- Each `logprob` is compared against a float64 reference within relative tolerance `1e-4` and absolute tolerance `1e-3`.
- Evaluation rules and cost limits are checked as exact counts. A stream that produces the right texts while breaking any rule is rejected.
- Pending operations are counted with `mx.export_to_dot`. Once an array is passed to `mx.async_eval`, it has 0 pending operations even before its value finishes computing.

## Why this matters

A token is not done as soon as the GPU finishes computing it. Python still has to detokenize the token, check stop sequences, update sampler state, and write the output to a socket or terminal. On Apple silicon, that CPU work can run while the GPU computes the next step.

`mlx_lm.generate_step` uses this exact pattern, calling `mx.async_eval` on step $t + 1$ before yielding token $t$. Getting the order wrong is a silent performance bug. Reading the token first still streams the exact right text, but leaves the GPU idle during Python work on every token. Even with zero detokenization cost, this increases per-token latency on a small model from 1.3 ms to 2.3 ms. Scheduling one extra step past the limit wastes a full forward pass at the end of every generation.
