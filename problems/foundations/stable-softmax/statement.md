Implement `softmax` and `log_softmax` along an axis so that they return correct, finite
results for **any** float32 logits, including values like `1e4` or `-1e4` where the
textbook formula overflows or underflows.

## Specification

```python
def softmax(x, axis=-1): ...
def log_softmax(x, axis=-1): ...
```

For a vector $x \in \mathbb{R}^n$:

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}},
\qquad
\operatorname{log\_softmax}(x)_i = x_i - \log \sum_{j=1}^{n} e^{x_j}.
$$

For a tensor, apply the function independently to every 1-D slice along `axis`. All
other axes are batch axes.

- `x`: a float32 tensor with one to four dimensions.
- `axis`: the dimension to normalize over. Negative values count from the end, as usual.
- Returns a float32 tensor with the same shape as `x`.

Entries of `x` may be `-inf`. This is how masked-out tokens are represented, and they
must come out as probability `0` (log-probability `-inf`). Every slice along `axis`
contains at least one finite value, and `+inf` and NaN never appear.

Write the arithmetic yourself: do not call your framework's built-in `softmax`,
`log_softmax`, or `logsumexp`. Elementwise functions (`exp`, `log`) and reductions
(`max`, `sum`) are what you need.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(d_1, ..., d_k)`, $1 \le k \le 4$ | float32 |
| result | `(d_1, ..., d_k)` | float32 |

## Example

```python
softmax([1.0, 2.0, 3.0])  # [0.0900, 0.2447, 0.6652]
log_softmax([1.0, 2.0, 3.0])  # [-2.4076, -1.4076, -0.4076]

# Shifting every logit by the same amount must not change the answer:
softmax([1001.0, 1002.0, 1003.0])  # [0.0900, 0.2447, 0.6652]
log_softmax([1001.0, 1002.0, 1003.0])  # [-2.4076, -1.4076, -0.4076]
```

The second pair is where the textbook formula fails. In float32, $e^{x}$ overflows to
`inf` once $x > 88.7$, so $e^{1001}$ is `inf` and `inf / inf` is NaN. In the other
direction, $e^{x}$ underflows to `0` below about $-103$, so for
`[-1001.0, -1002.0, -1003.0]` the textbook formula computes `0 / 0`.

`log_softmax` has a failure of its own. For `x = [0.0, -200.0]` the exact answer is
`[-1.4e-87, -200.0]`, which rounds to `[0.0, -200.0]`. Computing
`log(softmax(x))` instead gives `[0.0, -inf]`, because the probability
$e^{-200}$ is not representable in float32 even though its logarithm is.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`. Infinities must match exactly, and a NaN is always
  wrong.
- Finite logits lie in $[-10^4, 10^4]$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.

## Why this matters

Every token an LLM emits passes through this function: the final layer produces logits,
and sampling, top-p filtering, beam scores, and perplexity are all computed from
`softmax` or `log_softmax` of them. Real logits routinely have a spread of tens to
hundreds once you divide by a small temperature, add a logit bias, or mask tokens with
large negative values. An unstable implementation does not fail loudly. It
produces a NaN that turns into token id 0, or a `-inf` log-probability that silently
poisons a beam score several steps later. Inference engines therefore always work in
log space with the shifted formulation you are about to write.
