Implement block verification for speculative sampling. A draft model has sampled $K$ tokens in sequence, and a target model has scored all $K + 1$ resulting positions in one forward pass. Your function decides how many draft tokens to accept and samples one final token so that the emitted sequence has the **exact** distribution of the target model.

Each position follows the same accept-or-resample rule as a single speculative step. For a block, any draft after the first rejection is invalid, and accepting all $K$ drafts earns one extra bonus token from the target model.

![This problem is Verify, applied to a block of sampled draft tokens.](assets/map.svg)

## Specification

```python
def verify_block(p, q, draft_tokens, u_accept, u_resample): ...
```

Let $K$ be the number of drafted tokens and $V$ be the vocabulary size. Write $x_i$ for `draft_tokens[i]`. The draft model sampled $x_i$ from row `q[i]`, its distribution conditioned on the verified prefix and $x_0, \dots, x_{i-1}$. Row `p[i]` is the target model's distribution for that same position and context, and row `p[K]` is the target's distribution for the position after $x_{K-1}$.

Every row of `p` and `q` is non-negative and sums to 1. Entries of `p` may be exactly 0, and every drafted token has $q_i[x_i] > 0$. All random numbers are passed in as inputs (`u_accept` entries and `u_resample` lie in $[0, 1)$), so the function is deterministic.

**1. Accept a prefix.** Check the draft tokens in order from $i = 0$ to $K - 1$. Draft $i$ passes its acceptance test with probability $\min(1,\; p_i[x_i] / q_i[x_i])$. Multiplying both sides by $q_i[x_i]$ avoids division:

$$
\text{draft } i \text{ passes} \iff u_\text{accept}[i] \cdot q_i[x_i] < p_i[x_i].
$$

Stop at the first draft that fails. Write $n$ for `n_accepted`, the number of drafts accepted before that failure, or $K$ if all drafts pass:

$$
n = \min\big(\{\, i : 0 \le i < K,\; u_\text{accept}[i] \cdot q_i[x_i] \ge p_i[x_i] \,\} \cup \{K\}\big).
$$

Discard all drafts after position $n$ regardless of whether their own tests pass. Values `u_accept[j]` for $j > n$ have no effect on the result.

**2. Sample one final token.** Draw `next_token` by inverse-CDF sampling with `u_resample` from a weight vector $w$ of length $V$. The weights depend on whether a draft was rejected:

- **Rejection at position $n < K$.** Use the residual distribution $w = \max(0,\; p_n - q_n)$ at the rejected position.
- **All $K$ drafts accepted ($n = K$).** Use the target's distribution $w = p_K$ directly to draw a *bonus token*. There is no draft at position $K$, so nothing is subtracted.

Form the cumulative sums $c_j = \sum_{i \le j} w_i$ with total weight $W = c_{V-1}$, and pick the first token whose cumulative sum exceeds the threshold:

$$
\texttt{next\_token} = \min \{\, j : c_j > u_\text{resample} \cdot W \,\}.
$$

The total $W$ is always positive. When all drafts pass, $p_K$ sums to 1. When draft $n$ is rejected, we have $p_n[x_n] < q_n[x_n]$, so $p_n$ must exceed $q_n$ on at least one other token.

Return `(n_accepted, next_token)` as two Python `int`s. Together, `draft_tokens[:n_accepted]` followed by `next_token` form between 1 and $K + 1$ emitted tokens.

## Shapes

| name | shape | dtype |
|---|---|---|
| `p` | `(K + 1, V)` | float32 |
| `q` | `(K, V)` | float32 |
| `draft_tokens` | `(K,)` | integer |
| `u_accept` | `(K,)` | float32 |
| `u_resample` | scalar | Python `float` |
| result | `(n_accepted, next_token)` | Python `int`, Python `int` |

## Example

```python
p = [[0.10, 0.40, 0.20, 0.30],      # target, position of draft 0
     [0.25, 0.25, 0.25, 0.25],      # target, position of draft 1
     [0.50, 0.10, 0.10, 0.30]]      # target, position after the block
q = [[0.50, 0.20, 0.20, 0.10],      # draft, position 0
     [0.10, 0.60, 0.20, 0.10]]      # draft, position 1

verify_block(p, q, [1, 1], [0.9, 0.3], 0.65)   # (2, 2)
verify_block(p, q, [1, 1], [0.9, 0.7], 0.50)   # (1, 2)
verify_block(p, q, [0, 1], [0.6, 0.1], 0.70)   # (0, 3)
verify_block(p, q, [0, 3], [0.1, 0.99], 0.20)  # (2, 0)
```

1. **First call.** Draft 0 is token 1 with $p/q = 0.4 / 0.2 \ge 1$, so it is accepted for any uniform. Draft 1 is token 1 with $p/q = 0.25 / 0.6 \approx 0.417$, and $0.3 < 0.417$, so it is also accepted. Both drafts survive, and the bonus token is sampled from `p[2]`, whose cumulative sums are `[0.5, 0.6, 0.7, 1.0]`. The first sum above 0.65 is at index 2.
2. **Second call.** Draft 0 is accepted as before, but draft 1 is rejected because $0.7 \ge 0.417$, giving $n = 1$. The residual at position 1 is $\max(0, p_1 - q_1) = [0.15, 0, 0.05, 0.15]$, with total 0.35 and cumulative sums `[0.15, 0.15, 0.20, 0.35]`. The threshold is $0.5 \times 0.35 = 0.175$, and the first cumulative sum above it is at index 2.
3. **Third call.** Draft 0 is token 0 with $p/q = 0.1 / 0.5 = 0.2$. Since $0.6 \ge 0.2$, it is rejected immediately and $n = 0$. Draft 1 is ignored even though its uniform 0.1 would have passed. The residual at position 0 is `[0, 0.2, 0, 0.2]`, and the threshold $0.7 \times 0.4 = 0.28$ selects token 3.
4. **Fourth call.** Draft 0 passes because $0.1 < 0.2$, and draft 1 (token 3, $p/q = 2.5$) passes for any uniform. The bonus threshold 0.2 is below the first cumulative sum 0.5 of `p[2]`, selecting token 0.

## Constraints

- $1 \le K \le 8$ and $2 \le V \le 64$.
- **Exact cases.** Single calls are checked for exact equality. Inputs stay away from every decision boundary so float32 rounding cannot flip the result: $|u_\text{accept}[i] - p_i[x_i]/q_i[x_i]| \ge 10^{-3}$ for every examined draft with $0 < p_i[x_i]/q_i[x_i] < 1$, and $|u_\text{resample} - c_j / W| \ge 10^{-3}$ for every $j$ with $c_j > 0$.
- **Statistical cases.** The test samples each draft token from its row of `q`, draws independent uniforms, and makes up to 600 calls. Using finite-sample Chernoff bounds, it checks the distribution of `n_accepted`, the first emitted token (against `p[0]`), and the token emitted at position $j$ across calls where the first $j$ drafts were accepted (against `p[j]`, up to the bonus token against `p[K]`). A correct solution fails with probability below $10^{-9}$, and with fixed seeds it either always passes or always fails.
- The function is called a few thousand times per submission, so keep it to a handful of tensor operations per call.
- Do not draw random numbers inside the function.

## Why this matters

This verification step is the correctness core of every sampling-based speculative decoder, including Hugging Face `transformers` assisted generation and vLLM. The draft model is fast but imperfect, and this step guarantees that enabling speculation changes latency without altering the output distribution.

Bugs here fail silently. Keeping a draft token that passed after an earlier rejection shifts output toward the draft model. Sampling the bonus token from a residual, or drawing a correction from `p[n]` instead of the residual, distorts token frequencies and can make valid tokens impossible. The generated text still looks fluent and benchmarks speed up, but you are no longer serving the target model.
