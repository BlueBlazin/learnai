Measure the perplexity of a text under a real model, IBM's Granite 4.0 350M, when the
text is longer than the model's window. `forward` runs the model on at most `max_len`
tokens; you slide a window over the text, and the exercise is the bookkeeping: every
token is scored exactly once, by a window in which it has context.

## Specification

```python
def perplexity(forward, token_ids, max_len, stride): ...
```

- `forward`: a callable provided by the test. `forward(ids)` takes an integer tensor
  of shape `(T,)` with $1 \le T \le$ `max_len` and returns the logits, a float32 tensor
  of shape `(T, V)`. Row $t$ scores the token that follows `ids[t]`, and it is computed
  from `ids[0]` to `ids[t]` only: the model knows nothing outside the window it is
  given. More than `max_len` tokens raise `ValueError`.
- `token_ids`: the text, an integer tensor of shape `(N,)`, $N \ge 2$.
- `max_len`: the window length, a Python int.
- `stride`: how far the window advances, a Python int with $1 \le$ `stride` $<$
  `max_len`.
- Returns the perplexity, a 0-d float32 tensor.

**Windows.** Window $k = 0, 1, 2, \dots$ begins at token $b_k = k \cdot \texttt{stride}$
and ends before token $e_k = \min(b_k + \texttt{max\_len},\, N)$. The last window is the
first one with $e_k = N$; no window after it is run. Each window is one call,
`forward(token_ids[b_k:e_k])`.

**Which window scores which token.** Token 0 is never scored: nothing precedes it,
and the model has no beginning-of-text token to predict it from. Window $k$ scores the
tokens at positions

$$
\max(e_{k-1},\, 1) \;\le\; p \;<\; e_k \qquad (e_{-1} = 0),
$$

that is, the tokens no earlier window has scored. The tokens of the window before
those are context only. The log-probability of the token at position $p$ is read from
the row of the position before it:

$$
\ell_p = \operatorname{log\_softmax}\big(\text{logits}_k[\,p - 1 - b_k\,]\big)\big[\text{token\_ids}[p]\big],
$$

with the natural logarithm.

![A text of $N = 7$ tokens with `max_len` $= 4$ and `stride` $= 2$. Each window is one call of `forward` that scores the filled tokens and uses the outlined ones only as context, and token 0 is never scored.](assets/windows.svg)

**The result.** Every position $1 \le p \le N - 1$ is scored exactly once, so

$$
\text{perplexity} = \exp\!\Big(-\frac{1}{N - 1} \sum_{p=1}^{N-1} \ell_p\Big).
$$

Because `stride` is smaller than `max_len`, consecutive windows overlap, and every
token scored by a window after the first has at least `max_len - stride` tokens of
context. A smaller stride gives more context per token and costs more calls.

You may use your framework's `log_softmax`, `logsumexp`, and cross-entropy functions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `token_ids` | `(N,)` | integer |
| argument of `forward` | `(T,)`, $1 \le T \le$ `max_len` | integer |
| result of `forward` | `(T, V)` | float32 |
| result | `()` | float32 |

## Example

The first public case uses a model small enough to run in your head. Its vocabulary
has $V = 4$ tokens, and after reading $t + 1$ tokens it gives token $x$ the
probability

$$
\frac{1 + \#\{x \text{ among the tokens read}\}}{4 + t + 1},
$$

so it predicts what it has seen in its window, and a short window makes it worse.

```python
token_ids = [0, 1, 0, 0, 2, 0, 1]
perplexity(forward, token_ids, max_len=4, stride=2)  # 3.784930
```

Three windows are run, as in Figure 1:

| window | tokens | scores positions | probabilities |
|---|---|---|---|
| 0 | `[0, 1, 0, 0]` (positions 0 to 3) | 1, 2, 3 | $\tfrac15,\ \tfrac26,\ \tfrac37$ |
| 1 | `[0, 0, 2, 0]` (positions 2 to 5) | 4, 5 | $\tfrac16,\ \tfrac37$ |
| 2 | `[2, 0, 1]` (positions 4 to 6) | 6 | $\tfrac16$ |

In window 1, position 4 holds token `2`, and the model has read `[0, 0]`: the
probability is $(1 + 0) / (4 + 2) = \tfrac16$. Position 5 holds token `0` after
`[0, 0, 2]`: $(1 + 2) / (4 + 3) = \tfrac37$. Row 0 of this window, which predicts
position 3, is not used: position 3 was scored by window 0, where it had three tokens
of context instead of one. Row 3 predicts position 6, which is not in the window.

The product of the six probabilities is $\tfrac{1}{2940}$, so the perplexity is
$2940^{1/6} = 3.784930$.

On the real model, the 42 tokens of a paragraph about whetstones have a perplexity of
18.35 with windows of 16 and a stride of 8 (five calls). In one window of 42 tokens,
which this exercise does not allow, it would be 11.95: context matters.

## Constraints

- The result is compared to a relative tolerance of `1e-3` and an absolute tolerance
  of `1e-4`, against a float64 computation for the small models and against
  `transformers` (float32) for Granite. It must be a 0-d float32 tensor.
- $2 \le N \le 64$, $2 \le$ `max_len` $\le 16$, $1 \le$ `stride` $<$ `max_len`. $N$ may
  be smaller than `max_len` (one window).
- The cases that need no download use the counting model of the example and a random
  two-layer transformer with 64 tokens. The real model has $V = 100352$, `max_len` is
  16, and one call takes about 0.1 seconds, so call `forward` once per window.
- Real log-probabilities are not tame: a token the model finds absurd has a
  probability around $10^{-16}$. Every $\ell_p$ must be exact, down to about $-40$.
- The time limit per case is 20 seconds; the first real-model case decodes the
  weights, about 1 second, once per run.

## Why this matters

Perplexity on held-out text is the standard check that a model you quantized, ported,
or re-implemented is still the model you started with, and real evaluation texts are
longer than the context you can afford. The strided window is how every evaluation
harness handles that, and its two classic bugs both produce plausible numbers. Counting
the overlap twice, or cutting the text into disjoint chunks, scores many tokens with
almost no context: the perplexity comes out too high, by an amount that depends on the
window length, so two people evaluating the same model report different numbers.
Averaging per-window means instead of per-token terms makes the result depend on how
the text happened to be cut. The number is only comparable across runs when each token
is counted once, with the context the stride promises.
