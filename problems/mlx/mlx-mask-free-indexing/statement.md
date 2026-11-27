Implement two logit processors in MLX — top-k filtering and a presence penalty on previously generated tokens — **without leaving the lazy graph**. Common NumPy and PyTorch idioms (`x[x < t] = -inf` and `x[ids] -= v`) rely on boolean-mask indexing and in-place updates that MLX only partially supports, which often tempts developers into a NumPy round trip that forces evaluation mid-step.

## Specification

```python
def top_k_filter(logits, k): ...
def scatter_penalty(logits, ids, value): ...
```

Both functions take a float32 array `logits` of shape `(B, V)` (batch size $B$, vocabulary size $V$) and return a new float32 array of the same shape.

**`top_k_filter(logits, k)`** keeps the `k` largest logits in each row and replaces all other entries with $-\infty$. For each batch row $b \in [0, B)$ and vocabulary index $v \in [0, V)$:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if } \text{logits}[b, v] \text{ is one of the } k \text{ largest entries of row } b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

- Entries of `logits` may already be `-inf` (masked tokens). All finite entries within a row are pairwise distinct, so the `k` largest entries are unambiguous. A row with fewer than `k` finite entries keeps all of them.
- `k` is a Python `int` with $k \ge 1$. When $k \ge V$, return `logits` unchanged.

**`scatter_penalty(logits, ids, value)`** takes an integer array `ids` of shape `(B, N)` ($N \ge 1$) containing recent token ids in $[0, V)$ and a Python `float` `value`. It subtracts `value` from the logit of every token appearing in `ids`, **once per unique token regardless of how many times it occurs in the row**:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] - \text{value} & \text{if } v \in \{\text{ids}[b, 0], \dots, \text{ids}[b, N-1]\}, \\
\text{logits}[b, v] & \text{otherwise.}
\end{cases}
$$

- A row of `ids` may contain duplicate token ids.
- Negative `value` acts as a bonus. Any `-inf` entry in `logits` remains `-inf`.

Because both processors run inside a decoding step, two rules apply to both functions:

1. **Do not evaluate anything.** The returned array must remain a lazy graph over the inputs. Do not call `mx.eval`, `.item()`, `.tolist()`, or convert to NumPy. The judge passes unevaluated arrays and verifies they are still unevaluated on return.
2. **Do not mutate the inputs.** Assignments such as `logits[...] = ...` modify the caller's array in place. Always return a new array.

> [!NOTE]
> In MLX 0.32.4, reading through a boolean mask (`x[mask]`) raises `ValueError: boolean indices are not yet supported`. Assigning through a boolean mask (`x[mask] = value`) works lazily, but it mutates `x` in place. Integer-index assignments do not accumulate duplicates. For example, `x[ids] += 1` adds 1 once even if `ids` repeats a position, whereas `x.at[ids].add(1)` adds 2.

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

In `scatter_penalty`, row 0 lists token 1 twice and token 3 once. Each loses 1.5 exactly once ($3.0 \to 1.5$ and $2.0 \to 0.5$). Applying the penalty per occurrence would instead give $3.0 - 3.0 = 0.0$ for token 1. Row 1 lists token 0 twice and token 4 once, subtracting 1.5 once from each.

## Constraints

- $1 \le B \le 16$, $1 \le V \le 2000$, $1 \le N \le 64$, and $1 \le k \le 10^6$. Finite logits lie in $[-100, 100]$.
- Outputs are compared against a NumPy reference. Kept and unpenalized entries must match the input, penalized entries must match the difference within relative tolerance `1e-4` and absolute tolerance `1e-6`, and masked entries must be exactly `-inf`.
- Both rules (no evaluation and no input mutation) are checked on every call. Violating either fails the test even when the output values are right.
- The element order within partitions returned by `mx.argpartition`, `mx.partition`, and `mx.topk` is undefined. One test runs on MLX's CPU device, where that order differs from the GPU.
- No Python loop over rows or tokens is needed.

## Why this matters

Logit processors run once per generated token between the model's final layer and token selection, while the logits are still a lazy graph. Writing them with lazy MLX operations lets them fuse into the step's single GPU evaluation alongside argmax or sampling. Converting to NumPy splits the step in two, forcing the full vocabulary's logits to be evaluated, copied to the CPU, filtered, and uploaded back on every token. On an M1 Max with a 151,936-token vocabulary, that round trip raises the step time for the final layer, penalty, top-40, and argmax from 2.3 ms to 5.2 ms.

The duplicate-id rule is equally important. A presence penalty that accidentally accumulates over repeated ids suppresses common tokens (like spaces or newlines) by a multiple of the configured penalty, degrading generation quality in a way that is hard to trace back to an indexing choice.
