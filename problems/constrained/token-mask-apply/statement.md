Constrained generation works by removing, before every sampling step, the tokens that
would break the format. Implement that removal for a batch: `mask_logits` deletes the
disallowed tokens from the logits, and `masked_probs` returns the distribution over
what is left. The catch is in the corners: the model may put nearly all of its
probability on tokens it is not allowed to produce, and a row in which nothing is
allowed must be reported, not turned into NaN.

## Specification

```python
def mask_logits(logits, allowed): ...
def masked_probs(logits, allowed): ...
```

- `logits`: a float32 tensor of shape `(B, V)`, one row of next-token logits per
  sequence in the batch.
- `allowed`: a boolean tensor of shape `(B, V)`. `allowed[b, v]` is true when sequence
  `b` may generate token `v` next. Every row has its own mask.

**`mask_logits(logits, allowed)`** returns a float32 tensor of shape `(B, V)`:

$$
\operatorname{mask\_logits}(\ell, a)_{b,v} =
\begin{cases}
\ell_{b,v} & \text{if } a_{b,v} \\
-\infty & \text{otherwise.}
\end{cases}
$$

Allowed logits are returned unchanged, bit for bit. Disallowed entries are exactly
`-inf`, not a large negative number.

**`masked_probs(logits, allowed)`** returns a float32 tensor of shape `(B, V)`: for
each row, the softmax over the allowed tokens only.

$$
\operatorname{masked\_probs}(\ell, a)_{b,v} =
\begin{cases}
\dfrac{e^{\ell_{b,v}}}{\sum_{w \,:\, a_{b,w}} e^{\ell_{b,w}}} & \text{if } a_{b,v} \\[2ex]
0 & \text{otherwise.}
\end{cases}
$$

A disallowed token has probability exactly `0`, and every row sums to 1. The result
must be correct whatever the disallowed logits are: the allowed tokens may all lie
hundreds of units below a disallowed one.

**A row with nothing allowed.** If any row of `allowed` is entirely false, both
functions raise `ValueError` (with any message), even when the other rows are fine.
Such a row means the constraint has reached a dead end, and there is no distribution
to return.

Details:

- `logits` may already contain `-inf` (an earlier filter such as top-k removed those
  tokens). Such an entry stays `-inf` in `mask_logits` and has probability `0` in
  `masked_probs`, allowed or not. Every row that has an allowed token has at least one
  allowed token with a finite logit. `+inf` and NaN never appear.
- Neither function modifies its arguments.
- You may use your framework's `softmax` or `whetstone.kit.ops.softmax`; this problem
  is about the mask.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `allowed` | `(B, V)` | bool |
| result of `mask_logits` | `(B, V)` | float32 |
| result of `masked_probs` | `(B, V)` | float32 |

## Example

```python
logits  = [[2.0, 1.0, 0.0, 3.0],
           [0.5, 0.5, 4.0, 0.5]]
allowed = [[True, True,  False, False],
           [True, False, False, True ]]

mask_logits(logits, allowed)
# [[2.0,  1.0, -inf, -inf],
#  [0.5, -inf, -inf,  0.5]]

masked_probs(logits, allowed)
# [[0.7311, 0.2689, 0.0, 0.0],
#  [0.5,    0.0,    0.0, 0.5]]
```

In row 0 the model's favorite token (index 3, logit 3.0) is not allowed. The two
allowed tokens share the whole probability mass in the ratio $e^{2} : e^{1}$, which
gives $e / (1 + e) = 0.7311$ and $1 / (1 + e) = 0.2689$. Without the mask these two
tokens had probabilities 0.2369 and 0.0871: masking is not just zeroing, the survivors
are renormalized. In row 1 the two allowed logits are equal, so each gets 0.5, however
large the disallowed logit 4.0 is.

```python
mask_logits([[1.0, 2.0]], [[False, False]])     # raises ValueError
masked_probs([[1.0, 2.0], [3.0, 4.0]],
             [[True, False], [False, False]])   # raises ValueError: row 1 is empty
```

## Constraints

- $1 \le B \le 64$ and $1 \le V \le 4096$. Finite logits lie in $[-10^4, 10^4]$.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`. Infinities must match exactly and a NaN is always
  wrong. The tests also check directly that disallowed probabilities are exactly `0`
  and that allowed logits come back unchanged.
- No Python loops over rows or tokens are needed.

## Why this matters

This is the last step of every structured-output feature: JSON mode, tool calls, regex
and grammar constraints all end with "compute the set of allowed tokens, then apply it
to the logits". Engines apply the mask in exactly this form, as `-inf` written into the
logits before the sampler runs, so that temperature, top-p, and the final softmax all
see a distribution in which forbidden tokens do not exist. Getting the corners wrong
is how a server emits a token outside the grammar once in a million requests, or
returns NaN probabilities and crashes the sampler when a constraint dead-ends.
