Compute the perplexity of a token sequence under a real language model, IBM's Granite 4.0 350M, when the sequence is longer than the model's context window. You will slide an overlapping window across the sequence so that every predicted token is scored once with prior context.

## Specification

```python
def perplexity(forward, token_ids, max_len, stride): ...
```

- `forward`: a callable provided by the test. Given an integer tensor `ids` of length $T$ (with $1 \le T \le$ `max_len`), `forward(ids)` returns next-token logits over a vocabulary of size $V$. Row $t$ scores the token that follows `ids[t]` using only `ids[0]` through `ids[t]`. The model has no access to tokens outside the window it receives. Passing more than `max_len` tokens raises `ValueError`.
- `token_ids`: the input sequence of $N$ token IDs, with $N \ge 2$.
- `max_len`: the maximum window length, a Python `int`.
- `stride`: the number of tokens the window advances each step, a Python `int` with $1 \le$ `stride` $<$ `max_len`.
- Returns the sequence perplexity as a scalar float32 tensor.

**Windows.** Window $k = 0, 1, 2, \dots$ starts at index $b_k = k \cdot \texttt{stride}$ and ends before index $e_k = \min(b_k + \texttt{max\_len},\, N)$. Stop after the first window that reaches $e_k = N$. Each window runs a single model call, producing `logits_k = forward(token_ids[b_k:e_k])`.

**Which window scores which token.** Token 0 is never scored because nothing precedes it and there is no beginning-of-text token to predict it from. Each window $k$ scores only the positions that no earlier window has already scored, namely

$$
\max(e_{k-1},\, 1) \;\le\; p \;<\; e_k \qquad (e_{-1} = 0).
$$

All earlier tokens in window $k$ serve only as context. Within window $k$, token `token_ids[p]` is predicted from the row for position $p - 1$, which sits at local index $p - 1 - b_k$. Its natural log-probability is

$$
\ell_p = \operatorname{log\_softmax}\big(\text{logits}_k[\,p - 1 - b_k\,]\big)\big[\text{token\_ids}[p]\big].
$$

![$N = 7$ tokens with `max_len` $= 4$ and `stride` $= 2$. Each window scores its filled tokens and uses the outlined ones only as context.](assets/windows.svg)

**Perplexity.** Together, the windows score every position $1 \le p \le N - 1$ once. Perplexity is the exponential of the average negative log-probability across all $N - 1$ predicted tokens:

$$
\text{perplexity} = \exp\!\Big(-\frac{1}{N - 1} \sum_{p=1}^{N-1} \ell_p\Big).
$$

Because `stride` is strictly less than `max_len`, consecutive windows overlap. Every token scored after the first window therefore has at least `max_len - stride` tokens of context. A smaller stride provides more context per token at the cost of more `forward` calls.

You may use your framework's `log_softmax`, `logsumexp`, and cross-entropy functions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `token_ids` | `(N,)` | integer |
| argument of `forward` | `(T,)`, $1 \le T \le$ `max_len` | integer |
| result of `forward` | `(T, V)` | float32 |
| result | `()` | float32 |

## Example

The first public test uses a tiny counting model with vocabulary size $V = 4$. After reading $t + 1$ tokens in the current window, it assigns token $x$ the probability

$$
\frac{1 + \#\{x \text{ among the tokens read}\}}{4 + t + 1}.
$$

This model predicts tokens it has already seen in its window, so a shorter window hurts its predictions.

```python
token_ids = [0, 1, 0, 0, 2, 0, 1]
perplexity(forward, token_ids, max_len=4, stride=2)  # 3.784930
```

This runs three windows, matching the diagram above:

| window | tokens | scores positions | probabilities |
|---|---|---|---|
| 0 | `[0, 1, 0, 0]` (positions 0 to 3) | 1, 2, 3 | $\tfrac15,\ \tfrac26,\ \tfrac37$ |
| 1 | `[0, 0, 2, 0]` (positions 2 to 5) | 4, 5 | $\tfrac16,\ \tfrac37$ |
| 2 | `[2, 0, 1]` (positions 4 to 6) | 6 | $\tfrac16$ |

In window 1, position 4 holds token `2` after the model has read `[0, 0]`, giving probability $(1 + 0) / (4 + 2) = \tfrac16$. Position 5 holds token `0` after `[0, 0, 2]`, giving $(1 + 2) / (4 + 3) = \tfrac37$. Row 0 of window 1 predicts position 3, but we skip it because window 0 already scored position 3 with three tokens of context instead of one. Row 3 predicts position 6, which lies outside window 1.

The product of the six probabilities is $\tfrac{1}{2940}$, so the perplexity is $2940^{1/6} = 3.784930$.

On the real Granite model, a 42-token paragraph about whetstones has a perplexity of 18.35 with `max_len = 16` and `stride = 8` (five calls). Scored in a single 42-token window, which this problem does not allow, the same paragraph has a perplexity of 11.95 because every token gets full context.

## Constraints

- The output must be a 0-d float32 tensor. It is checked with relative tolerance `1e-3` and absolute tolerance `1e-4` against a float64 reference for the small models and against `transformers` (float32) for Granite.
- $2 \le N \le 64$, $2 \le$ `max_len` $\le 16$, and $1 \le$ `stride` $<$ `max_len`. $N$ may be smaller than `max_len`, in which case only one window runs.
- The tests that do not require a weight download use the counting model above and a random two-layer transformer with 64 tokens. The real model has $V = 100352$ and `max_len = 16`. Each call takes about 0.1 seconds, so call `forward` only once per window.
- Real language models can assign unlikely tokens probabilities around $10^{-16}$. Every $\ell_p$ must remain accurate down to about $-40$.
- The time limit per test case is 20 seconds. The first real-model test decodes the weights once per run (about 1 second).

## Why this matters

Perplexity on held-out text is the standard sanity check that a quantized, ported, or re-implemented language model matches the original. Real evaluation passages are longer than a single context window, so evaluation harnesses use a strided sliding window.

The classic sliding-window bugs all produce plausible-looking numbers. Scoring overlapping tokens twice, or splitting the text into disjoint chunks, evaluates many tokens with little or no context and inflates perplexity by an amount that depends on the window size. Averaging per-window means instead of per-token log-probabilities makes the score depend on where window boundaries fall. Perplexity is only comparable across runs when every token is counted once with the context that `stride` guarantees.
