Score a batch of token sequences under a model's logits: the log-probability the model
assigns to each sequence, and its perplexity. The arithmetic is a `log_softmax` and a
sum. The exercise is the alignment (the logits at position `t` score the token at
position `t + 1`) and counting exactly the right positions when the batch is padded.

## Specification

```python
def sequence_logprob(logits, tokens, mask): ...
def perplexity(logits, tokens, mask): ...
```

A language model reads `tokens[b, 0], ..., tokens[b, t]` and produces `logits[b, t]`,
a vector of `V` scores for the **next** token, the one at position `t + 1`. The
log-probability it assigns to that token is

$$
\ell_{b,t} = \operatorname{log\_softmax}(\text{logits}[b, t])\big[\text{tokens}[b, t + 1]\big],
\qquad 0 \le t \le T - 2 .
$$

The logits at the last position predict a token that is not in the batch, and no
logits predict the token at position 0, so a row of `T` tokens has `T - 1`
(context, target) pairs.

`mask[b, t]` is `True` where `tokens[b, t]` is a real token and `False` where it is
padding. Pair `t` of row `b` is **counted** when both of its positions are real:

$$
c_{b,t} = \text{mask}[b, t] \;\wedge\; \text{mask}[b, t + 1] .
$$

A prediction of a padding token is not counted, and neither is a prediction made from
a padding position. In particular, the first real token of a row is never scored.

**`sequence_logprob`** returns, for each row, the sum over the counted pairs, and
**`perplexity`** the exponential of the negative mean over the counted pairs:

$$
\text{logprob}[b] = \sum_{t \,:\, c_{b,t}} \ell_{b,t},
\qquad
\text{perplexity}[b] = \exp\!\left( - \frac{\text{logprob}[b]}{n_b} \right),
\qquad
n_b = \#\{t : c_{b,t}\} .
$$

Both use the natural logarithm.

- `logits`: a float32 tensor of shape `(B, T, V)`.
- `tokens`: an integer tensor of shape `(B, T)` with values in $[0, V)$, also at
  padding positions.
- `mask`: a bool tensor of shape `(B, T)`.
- Both functions return a float32 tensor of shape `(B,)`.

Pairs that are not counted must not influence the result, whatever the logits and
tokens at those positions are. Logits may be `-inf` (a banned token), so the
log-probability of an uncounted pair may be `-inf`; the target of a counted pair
always has a finite logit.

You may use your framework's `log_softmax`, `logsumexp`, and cross-entropy functions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, T, V)` | float32 |
| `tokens` | `(B, T)` | integer |
| `mask` | `(B, T)` | bool |
| result (both functions) | `(B,)` | float32 |

## Example

Two rows with `T = 3` and `V = 3`. Both rows use the same next-token probabilities,
given here directly; the logits are their logarithms, so `log_softmax` returns them
unchanged.

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

Row 0 has two counted pairs. Position 0 predicts `tokens[0, 1] = 2` with probability
$0.25$, and position 1 predicts `tokens[0, 2] = 1` with probability $0.8$. The sum is
$\ln 0.25 + \ln 0.8 = \ln 0.2 = -1.609438$, and the perplexity is
$\exp(1.609438 / 2) = \sqrt{5} = 2.236068$. The token at position 0 and the third row
of `probs` are never used.

Row 1 has one counted pair, because position 2 is padding: position 0 predicts
`tokens[1, 1] = 0` with probability $0.5$. The sum is $\ln 0.5 = -0.693147$ and the
perplexity is $2$.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-4`. A NaN is always wrong.
- $1 \le B \le 16$, $2 \le T \le 64$, $1 \le V \le 64$.
- In every row the real tokens form one contiguous run of at least two positions
  (padding on the left or on the right), so $n_b \ge 1$.
- Finite logits lie in $[-10^4, 10^4]$, and `sequence_logprob` must be finite and
  correct over that whole range. `perplexity` is only called on inputs whose result is
  below $10^6$.
- No Python loops over batch items or positions are needed.

## Why this matters

This function is how a model is measured and, in several places, how it is run.
Perplexity on held-out text is the standard check that a quantized or re-implemented
model still behaves like the original; scoring answer options by their log-likelihood
is how multiple-choice benchmarks are evaluated; and speculative decoding verifies
a draft by reading off the probability the large model assigns to each drafted token,
with exactly this shift. An alignment that is off by one does not crash. Scoring
`tokens[t]` under `logits[t]` asks the model to predict a token it has already
seen, and the numbers are wrong in a way that only a careful comparison reveals.
Counting padding makes a model's score depend on the length of the longest sequence in
the batch, so two runs of the same evaluation with different batch sizes disagree.
