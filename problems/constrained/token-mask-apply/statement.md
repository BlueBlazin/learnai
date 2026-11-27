Constrained generation removes tokens that would break the output format before each sampling step. Implement `mask_logits` to mask out disallowed tokens from a batch of logits, and `masked_probs` to compute the next-token probability distribution over the remaining allowed tokens. Your implementation must stay numerically accurate even when disallowed tokens have much larger logits than allowed ones, and it must raise an error if any row has no allowed tokens.

![This problem is Mask, which removes the disallowed tokens from the logits.](assets/map.svg)

## Specification

```python
def mask_logits(logits, allowed): ...
def masked_probs(logits, allowed): ...
```

Both functions operate on a batch of `B` sequences over a vocabulary of `V` tokens. `logits` holds one row of next-token logits per sequence. `allowed` is a per-row boolean mask where `allowed[b, v]` is `True` when sequence `b` may generate token `v` next.

**`mask_logits(logits, allowed)`** keeps every allowed logit unchanged bit for bit and replaces every disallowed entry with exact `-inf` (not a large negative number). Writing $\ell_{b,v}$ for `logits[b, v]` and $a_{b,v}$ for `allowed[b, v]`, each output entry is

$$
\operatorname{mask\_logits}(\ell, a)_{b,v} =
\begin{cases}
\ell_{b,v} & \text{if } a_{b,v} \\
-\infty & \text{otherwise.}
\end{cases}
$$

**`masked_probs(logits, allowed)`** computes the softmax over the allowed tokens in each row, assigning exact probability `0` to every disallowed token so each row sums to 1.

$$
\operatorname{masked\_probs}(\ell, a)_{b,v} =
\begin{cases}
\dfrac{e^{\ell_{b,v}}}{\sum_{w \,:\, a_{b,w}} e^{\ell_{b,w}}} & \text{if } a_{b,v} \\[2ex]
0 & \text{otherwise.}
\end{cases}
$$

The output must remain accurate regardless of the disallowed logits, including when all allowed logits are hundreds of units below a disallowed logit.

**Empty mask rows.** If any row of `allowed` is all `False`, both functions must raise `ValueError` (with any message), even when other rows have allowed tokens. An all-`False` row means the constraint has reached a dead end and has no valid distribution.

Additional requirements:

- `logits` may already contain `-inf` from an earlier filter such as top-k. Those entries stay `-inf` in `mask_logits` and receive probability `0` in `masked_probs`, whether `allowed` is `True` or `False`.
- Every row with at least one `True` entry in `allowed` also has at least one allowed token with a finite logit. `+inf` and NaN never appear in `logits`.
- Neither function modifies its input tensors.
- You may use your framework's `softmax` or `learnai.kit.ops.softmax`.

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

In row 0, the model's highest-scoring token (index 3, logit 3.0) is disallowed. The two allowed tokens share all of the probability mass in the ratio $e^{2} : e^{1}$, giving $e / (1 + e) = 0.7311$ and $1 / (1 + e) = 0.2689$. Without the mask, those two tokens had probabilities 0.2369 and 0.0871. Masking renormalizes the surviving tokens rather than simply zeroing out the disallowed ones. In row 1, both allowed tokens have logit 0.5, so each gets 0.5 regardless of the disallowed logit 4.0.

```python
mask_logits([[1.0, 2.0]], [[False, False]])     # raises ValueError
masked_probs([[1.0, 2.0], [3.0, 4.0]],
             [[True, False], [False, False]])   # raises ValueError: row 1 is empty
```

## Constraints

- $1 \le B \le 64$ and $1 \le V \le 4096$. Finite logits lie in $[-10^4, 10^4]$.
- Results are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. Infinities must match exactly, and any NaN fails.
- Tests also verify directly that disallowed probabilities are exact `0` and that allowed logits are returned unchanged.
- No Python loops over rows or tokens are needed.

## Why this matters

Every structured-output feature ends with this step. JSON mode, tool calling, regex filters, and grammar constraints all compute a set of allowed tokens and then apply it to the logits. Inference engines write `-inf` directly into the logits before sampling so that temperature scaling, top-p filtering, and the final softmax operate on a distribution where forbidden tokens do not exist.

Handling the edge cases matters in production. A sloppy mask can leak a grammar-breaking token once in a million requests, or produce NaN probabilities that crash the sampler when a constraint dead-ends.
