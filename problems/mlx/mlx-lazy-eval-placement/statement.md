Implement greedy token generation with log-probability scoring for a transformer language model in MLX, calling `mx.eval` **exactly once per generated token**. Because MLX uses lazy evaluation, where you place `mx.eval` controls both how often the GPU runs and how much unevaluated graph accumulates between steps.

## Specification

```python
def generate(model, prompt, n): ...
```

`model` is an [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py), a small seeded decoder-only transformer built with `mlx.nn`. Both ways of calling it return **lazy** arrays, which represent a graph of operations that have not been computed yet.

```python
logits, cache = model(tokens)         # no cache: tokens is the full sequence so far
logits, cache = model(tokens, cache)  # with cache: tokens continue the cached prefix
```

- `tokens` is a 1-D integer array of length $T \ge 1$.
- `logits` contains one row of length `vocab_size` for each token passed in the call. The final row, `logits[-1]`, scores the next token in the sequence.
- `cache` is a list with one `(keys, values)` pair of arrays per transformer layer. The model never mutates a cache in place and always returns a new one.

`generate` takes `model`, `prompt` (a list of $P \ge 1$ Python ints), and `n` (the number of tokens to generate, $n \ge 1$, with no stop token). Greedy decoding repeats a single step $n$ times. At each step, let $\ell$ be the next-token logits (`logits[-1]`) given the prompt and all tokens generated so far. Select the highest-scoring vocabulary index $v$ and compute its log-probability via log-softmax:

$$
\text{token} = \arg\max_v \ell_v ,
\qquad
\log p(\text{token}) = \ell_{\text{token}} - \log \sum_v e^{\ell_v} .
$$

`generate` returns `(tokens, logprob)`:

- `tokens` is a Python list of $n$ scalar (0-dimensional) integer arrays in generation order. Any integer dtype is accepted (`mx.argmax` returns `uint32`).
- `logprob` is a scalar float32 array equal to the sum of $\log p(\text{token})$ over the $n$ generated tokens.

**Evaluation rules.** The judge monitors how and where your code evaluates arrays:

1. Call `mx.eval` exactly $n$ times, once per generated token. Do not use `mx.async_eval`.
2. Do not trigger implicit evaluations. Never call `.item()`, `.tolist()`, `int(x)`, `float(x)`, `np.array(x)`, or `print(x)` on an MLX array.
3. No lazy graph may cross a step. Whenever `model` is called, at most 8 pending operations may sit behind its arguments (the token array and all arrays in the cache combined). Reshaping an evaluated token adds 1 pending operation, whereas an unevaluated step leaves several hundred.
4. All returned arrays must already be evaluated, with 0 pending operations behind the $n$ tokens and `logprob`. The caller must be able to read them without triggering computation.

![Three steps of the loop. One `mx.eval` per token cuts the lazy graph at each dashed line, so only computed arrays cross into the next step.](assets/eval-cuts.svg)

**Cost limits.** As in `cached-greedy-decode`, `model.counters.forward_calls` must be at most $n$ (one call for the prompt, then one call per generated token except the last, which is returned and never fed back). `model.counters.tokens_processed` must be at most $P + n - 1$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `model`) | `(T, vocab_size)` for a call with $T$ tokens | float32 |
| `tokens` (result) | list of $n$ arrays of shape `()` | integer |
| `logprob` (result) | `()` | float32 |

## Example

```python
model = MLXTinyLM(seed=3)  # vocab_size 64

tokens, logprob = generate(model, [0, 17, 42, 5], 8)
# tokens:  [28, 34, 31, 15, 1, 48, 46, 5]     (eight scalar arrays)
# logprob: -12.4394
# 8 calls of mx.eval, 8 forward calls, 11 tokens processed
```

| step | tokens passed to `model` | argmax of `logits[-1]` | evaluated by this step's `mx.eval` |
|---|---|---|---|
| 1 | `[0, 17, 42, 5]` | 28 | token 28, the log-probability so far, a cache of 4 tokens |
| 2 | `[28]` | 34 | token 34, the log-probability so far, a cache of 5 tokens |
| ... | | | |
| 8 | `[46]` | 5 | token 5, the final log-probability, a cache of 11 tokens |

Step 8 feeds token 46 (generated in step 7) and produces token 5, which is returned without being fed back. On every call after the first, the model receives an evaluated cache and a 1-element array reshaped from an evaluated token, leaving 1 pending operation.

## Constraints

- $1 \le P \le 64$ and $1 \le n \le 64$.
- Generated tokens are compared exactly against greedy decoding without a cache. The test models keep the two largest logits more than `1e-3` apart so the argmax does not depend on how calls are split. If two logits tie, the argmax is the lower token id (this does not occur in the tests).
- `logprob` is compared against a float64 reference with relative tolerance `1e-4` and absolute tolerance `1e-3`.
- The evaluation rules and cost limits are enforced as exact counts. A solution that violates any of them fails even if its tokens and score match.
- Pending operations are counted with `mx.export_to_dot`, which exports the unevaluated graph behind an array. An evaluated array has 0 pending operations.

## Why this matters

Production generation loops on Apple silicon, including `mlx-lm`, evaluate exactly once per token for two competing reasons. Evaluating less often lets the computation graph grow across steps. An unevaluated accumulator (such as a running log-probability, repetition-penalty history, or token statistics) silently forms a chain across every step and keeps old logits alive in memory. Deferring all evaluation until the end also prevents token streaming, stop-token checks, and bounded memory usage.

Evaluating more often, or calling `.item()` mid-step, has the opposite cost. Each extra evaluation schedules a new graph and synchronizes with the GPU, adding fixed overhead that takes up a noticeable share of per-token latency on small and medium models.
