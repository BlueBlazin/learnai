Implement `sequence_logprob` and `perplexity` to score a batch of token sequences from a language model's logits. Both functions align each position's next-token prediction with the following token and exclude padded positions.

## Specification

```python
def sequence_logprob(logits, tokens, mask): ...
def perplexity(logits, tokens, mask): ...
```

The inputs represent a batch of `B` padded token sequences of length `T` over a vocabulary of size `V`. `logits` holds the model's unnormalized next-token scores, `tokens` holds integer token IDs in $[0, V)$ (including at padding positions), and `mask` is `True` where a token is real and `False` where it is padding.

At batch item `b` and position `t`, the model has read `tokens[b, 0], ..., tokens[b, t]` and produces `logits[b, t]`, a vector of `V` scores for the **next** token at position `t + 1`. The log-probability assigned to that next token is

$$
\ell_{b,t} = \operatorname{log\_softmax}(\text{logits}[b, t])\big[\text{tokens}[b, t + 1]\big],
\qquad 0 \le t \le T - 2 .
$$

The logits at position `T - 1` predict a token outside the batch, and no logits predict the token at position `0`. A row of `T` tokens therefore has `T - 1` `(context, target)` pairs.

Pair `t` of row `b` is **counted** when both of its positions are real tokens:

$$
c_{b,t} = \text{mask}[b, t] \;\wedge\; \text{mask}[b, t + 1] .
$$

Neither a prediction of a padding token nor a prediction made from a padding position is counted. In particular, the first real token of a row is never scored.

**`sequence_logprob`** returns the sum of $\ell_{b,t}$ over the counted pairs in each row. **`perplexity`** returns the exponential of the negative mean over those counted pairs:

$$
\text{logprob}[b] = \sum_{t \,:\, c_{b,t}} \ell_{b,t},
\qquad
\text{perplexity}[b] = \exp\!\left( - \frac{\text{logprob}[b]}{n_b} \right),
\qquad
n_b = \#\{t : c_{b,t}\} .
$$

Both functions use the natural logarithm.

Uncounted pairs must not influence the result, regardless of their logits or token IDs. Logits may be `-inf` (for example, a banned token), so an uncounted pair can have log-probability `-inf`. The target of a counted pair always has a finite logit.

You may use your framework's `log_softmax`, `logsumexp`, and cross-entropy functions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, T, V)` | float32 |
| `tokens` | `(B, T)` | integer |
| `mask` | `(B, T)` | bool |
| result (both functions) | `(B,)` | float32 |

## Example

Consider two rows with `T = 3` and `V = 3`. Both rows share the same next-token probability distribution `probs`, and `logits` is its elementwise logarithm so `log_softmax` returns `logits` unchanged.

```python
probs = [[0.5, 0.25, 0.25],     # at position 0, for the token at position 1
         [0.1, 0.8,  0.1 ],     # at position 1, for the token at position 2
         [0.2, 0.3,  0.5 ]]     # at position 2, for a token that is not there
logits = log([probs, probs])                        # shape (2, 3, 3)
tokens = [[0, 2, 1],
          [1, 0, 2]]
mask   = [[True, True, True],
          [True, True, False]]

sequence_logprob(logits, tokens, mask)  # [-1.609438, -0.693147]
perplexity(logits, tokens, mask)        # [ 2.236068,  2.0     ]
```

Row 0 has two counted pairs. Position 0 predicts `tokens[0, 1] = 2` with probability $0.25$, and position 1 predicts `tokens[0, 2] = 1` with probability $0.8$. Their log-probability sum is $\ln 0.25 + \ln 0.8 = \ln 0.2 = -1.609438$, and the perplexity is $\exp(1.609438 / 2) = \sqrt{5} = 2.236068$. The token at position 0 and the third row of `probs` are never used.

Row 1 has one counted pair because position 2 is padding. Position 0 predicts `tokens[1, 1] = 0` with probability $0.5$, giving a sum of $\ln 0.5 = -0.693147$ and a perplexity of $2.0$.

## Constraints

- Results are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-4`. A NaN is always wrong.
- $1 \le B \le 16$, $2 \le T \le 64$, $1 \le V \le 64$.
- In every row, the real tokens form one contiguous run of at least two positions (with padding on the left or on the right), so $n_b \ge 1$.
- Finite logits lie in $[-10^4, 10^4]$, and `sequence_logprob` must remain finite and accurate across that full range. `perplexity` is only called on inputs whose result is below $10^6$.
- No Python loops over batch items or positions are needed.

## Why this matters

Log-likelihood and perplexity are core primitives for both evaluating and serving language models. Perplexity on held-out text checks whether a quantized or re-implemented model still matches the original. Multiple-choice benchmarks rank candidate answers by their log-likelihood, and speculative decoding verifies drafted tokens by reading the target model's probability for each drafted token using this exact one-step shift.

Bugs here fail silently. Scoring `tokens[t]` under `logits[t]` evaluates each token after the model has already seen it, which produces plausible-looking but invalid numbers. Including padded positions makes a sequence's score depend on the longest sequence in the batch, causing identical evaluations at different batch sizes to disagree.
