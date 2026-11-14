Implement two logit processors in MLX, top-k filtering and a penalty on tokens that
were already used, **without leaving the lazy graph**. The NumPy and PyTorch idioms
for both (`x[x < t] = -inf`, `x[ids] -= v`) lean on boolean-mask indexing and in-place
scatter, which MLX supports only in part, so the usual escape is a round trip through
NumPy that evaluates the logits in the middle of a decoding step.

## Specification

```python
def top_k_filter(logits, k): ...
def scatter_penalty(logits, ids, value): ...
```

**`top_k_filter`** keeps the `k` largest logits of every row and masks the others:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if } \text{logits}[b, v] \text{ is one of the } k \text{ largest entries of row } b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

- `logits`: a float32 array of shape `(B, V)`. Entries may be `-inf` (a token that is
  already masked); the finite entries of a row are pairwise distinct, so "the `k`
  largest" is unambiguous. A row with fewer than `k` finite entries keeps all of them.
- `k`: a Python int, $k \ge 1$. With $k \ge V$ the logits are returned unchanged.
- Returns a float32 array of shape `(B, V)`.

**`scatter_penalty`** subtracts `value` from the logit of every token that occurs in
`ids`, **once per token, however often the token occurs**:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] - \text{value} & \text{if } v \in \{\text{ids}[b, 0], \dots, \text{ids}[b, N-1]\}, \\
\text{logits}[b, v] & \text{otherwise.}
\end{cases}
$$

- `ids`: an integer array of shape `(B, N)`, $N \ge 1$, with token ids in $[0, V)$:
  the recent tokens of each sequence. A row may contain the same id several times.
- `value`: a Python float (negative values are a bonus). A `-inf` logit stays `-inf`.
- Returns a float32 array of shape `(B, V)`.

Both functions are pieces of a decoding step, so two rules apply to both:

1. **Nothing is evaluated.** The result is a lazy graph over the arguments: no
   `mx.eval`, no `.item()` or `.tolist()`, no conversion to NumPy. The judge passes
   unevaluated arrays and checks that they are still unevaluated when the function
   returns.
2. **The arguments are not modified.** An assignment such as `logits[...] = ...`
   changes the caller's array; return a new one.

> [!NOTE]
> In MLX 0.32.4, reading through a boolean mask (`x[mask]`) raises
> `ValueError: boolean indices are not yet supported`. Assigning through one
> (`x[mask] = value`) works and is lazy, but it assigns into `x`. Updates through
> integer indices do not accumulate: `x[ids] += 1` adds 1 once to a position that
> `ids` names twice, while `x.at[ids].add(1)` adds 2.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| `ids` | `(B, N)` | int32 |
| result (both functions) | `(B, V)` | float32 |

## Example

```python
logits = [[1.0, 3.0, 0.5, 2.0, -1.0], [0.0, 2.5, -inf, 4.0, 1.5]]

top_k_filter(logits, 2)
# [[-inf, 3.0, -inf, 2.0, -inf],
#  [-inf, 2.5, -inf, 4.0, -inf]]

scatter_penalty(logits, [[1, 3, 1], [0, 0, 4]], 1.5)
# [[ 1.0, 1.5, 0.5, 0.5, -1.0],
#  [-1.5, 2.5, -inf, 4.0,  0.0]]
```

In the penalty, row 0 names token 1 twice and token 3 once: both lose 1.5 exactly
once ($3.0 \to 1.5$ and $2.0 \to 0.5$). Applying the penalty per occurrence would give
$3.0 - 3.0 = 0.0$ for token 1. Row 1 names token 0 twice and token 4 once.

## Constraints

- $1 \le B \le 16$, $1 \le V \le 2000$, $1 \le N \le 64$, $1 \le k \le 10^6$. Finite
  logits lie in $[-100, 100]$.
- Results are compared with a NumPy implementation: kept and unpenalized entries must
  equal the input and penalized ones the difference (relative tolerance `1e-4`,
  absolute `1e-6`); masked entries must be exactly `-inf`.
- The two rules above are checked on every call. A solution that evaluates, or that
  modifies `logits`, fails even when its numbers are right.
- The order in which `mx.argpartition`, `mx.partition`, and `mx.topk` return the
  elements of a part is undefined, and one test runs on MLX's CPU device, where it
  differs from the GPU's.
- No Python loop over rows or tokens is needed.

## Why this matters

Samplers run once per generated token, between the model's last layer and the choice
of the token, on logits that are still a lazy graph. A processor written with MLX
operations joins that graph and is computed by the step's single evaluation, on the
GPU, next to the argmax or the sampling that follows. A processor that converts to
NumPy cuts the step in two: the logits of the whole vocabulary are computed, copied
out, filtered on the CPU, and uploaded again, for every token. With a vocabulary of
151,936 on this machine that is 5.2 ms per step instead of 2.3 ms for the last layer,
penalty, top-40, and argmax together. The duplicate rule matters as much: a repetition
penalty that accumulates over repeated ids pushes a common token (a space, a newline)
down by a multiple of what was configured, and the output degrades in a way that is
hard to trace back to an indexing idiom.
