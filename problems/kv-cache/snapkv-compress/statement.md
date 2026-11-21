Implement the selection step of SnapKV, which shrinks the KV cache of a long prompt
once, right after the prefill: the last few prompt tokens vote, with their attention
weights, for the earlier positions worth keeping, separately in every head. Votes are
max-pooled along the sequence before the top positions are chosen, so that a position
with a high vote brings its neighbors along.

![This problem is the KV cache, here shrunk once after the prefill.](assets/map.svg)

## Specification

```python
def snapkv_select(attn, window, budget, kernel): ...  # -> kept positions, per head
```

`attn` is the attention of one layer over a prompt of $L$ tokens, shape `(B, H, L, L)`:
`attn[b, h, i, j]` is the weight query $i$ puts on position $j$. It is causal (0 for
$j > i$) and every row sums to 1.

The last `window` positions, $L - \text{window}, \dots, L - 1$, are the **observation
window**. The $P = L - \text{window}$ positions before it are the **prefix**. The
function returns, for every batch row and head, the positions whose keys and values
the compressed cache keeps.

If $L \le \text{budget}$, nothing is dropped: every head keeps positions
$0, \dots, L - 1$. Otherwise, independently for each batch row $b$ and head $h$:

**1. Vote.** Each prefix position collects the attention it receives from the queries
of the window:

$$
\text{vote}_j = \sum_{i = P}^{L - 1} \text{attn}_{b,h,i,j}, \qquad 0 \le j < P .
$$

Queries before the window do not vote, and the window's own positions ($j \ge P$)
receive no vote.

**2. Pool.** Max-pool the votes along the sequence with width `kernel` and stride 1.
With $r = \lfloor \text{kernel} / 2 \rfloor$:

$$
\text{pooled}_j = \max_{\substack{j - r \,\le\, j' \,\le\, j + r \\ 0 \,\le\, j' \,<\, P}} \text{vote}_{j'} .
$$

At the two ends the maximum runs over the positions that exist: nothing outside the
prefix takes part, in particular not the window.

**3. Select.** Keep the $\text{budget} - \text{window}$ prefix positions with the
highest pooled votes. Of two positions with equal pooled votes, the lower position
is taken first.

**4. Return** the selected prefix positions in **increasing order**, followed by the
window's positions $P, \dots, L - 1$: `budget` positions per head.

- `window`, `budget`, `kernel`: Python ints with $1 \le \text{window} \le \text{budget}$
  and `kernel` odd and $\ge 1$. `kernel = 1` means no pooling. `kernel` may exceed $P$.
  With `budget = window` only the window is kept.
- Returns an integer tensor of shape `(B, H, min(L, budget))`.

Pooling makes equal pooled votes the normal case, not a corner case: every position
within $r$ of a peak carries the peak's value. The tie rule above decides which of
them fit when the budget cuts through such a plateau.

## Shapes

| name | shape | dtype |
|---|---|---|
| `attn` | `(B, H, L, L)` | float32 |
| result | `(B, H, min(L, budget))` | integer |

## Example

One batch row, one head, $L = 10$, `window = 2`, `budget = 5`, `kernel = 3`. The
prefix is positions 0 to 7 and the window is positions 8 and 9. The rows of the two
window queries:

```python
attn[0, 0, 8] = [0.02, 0.02, 0.40, 0.02, 0.02, 0.02, 0.20, 0.02, 0.28, 0.00]
attn[0, 0, 9] = [0.02, 0.02, 0.30, 0.02, 0.02, 0.02, 0.10, 0.06, 0.14, 0.30]
```

The eight queries before the window put all their weight on position 0; they do not
vote.

| position $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| vote | 0.04 | 0.04 | 0.70 | 0.04 | 0.04 | 0.04 | 0.30 | 0.08 |
| pooled ($r = 1$) | 0.04 | 0.70 | 0.70 | 0.70 | 0.04 | 0.30 | 0.30 | 0.30 |

Columns 8 and 9 are not in the table: the weights 0.28, 0.14, and 0.30 that the
window puts on itself are ignored, and position 7 is pooled over positions 6 and 7
only.

`budget - window = 3` prefix positions are kept. The three highest pooled votes are
the plateau around position 2, so the result is `[1, 2, 3, 8, 9]`: the most attended
position, its two neighbors, and the window.

With `kernel = 1` the same votes give `[2, 6, 7, 8, 9]`: the three highest raw
votes, each on its own.

With `budget = 6` and `kernel = 3` a fourth prefix position is kept. Positions 5, 6,
and 7 tie at 0.30, and the lowest goes first: `[1, 2, 3, 5, 8, 9]`.

## Constraints

- $1 \le B \le 2$, $1 \le H \le 4$, $L \le 32$, $\text{kernel} \le 9$.
- Positions are compared exactly.
- Two votes of one head are either exactly equal or differ by at least $10^{-4}$,
  so float rounding in the sum cannot change a ranking.
- No Python loops over batch rows, heads, or positions are needed (a loop over the
  `kernel` offsets is fine).

## Why this matters

With a long prompt and a short answer, almost all of the KV cache is prompt, and
every generated token attends over all of it. SnapKV rests on an observation about
when to decide what to keep: the attention that the end of the prompt (usually the
question or instruction) pays to the rest of it predicts well what the generation
will attend to. So the cache can be compressed once, before the first token is
generated, with no per-step bookkeeping, and each head keeps the positions it
looks at. The pooling step is what makes the selection usable. Attention marks the
first token of a name or a number; the answer needs the tokens around it too, and
the paper's example of the failure is a model that retrieves the country code of a
phone number and makes up the rest. Bugs here do not raise errors: a vote taken over
the wrong rows or pooled across the window boundary still returns `budget`
positions per head, just not the ones the model will need.
