Implement the selection step of SnapKV, which compresses a prompt's KV cache once right after prefill. The last few prompt tokens use their attention weights to vote for earlier positions worth keeping, independently in each head. Before selecting the top positions, the votes are max-pooled along the sequence so that a high-scoring position brings its neighbors along.

![This problem is the KV cache, here shrunk once after the prefill.](assets/map.svg)

## Specification

```python
def snapkv_select(attn, window, budget, kernel): ...  # -> kept positions, per head
```

The input `attn` is the attention weight tensor of one layer over a prompt of $L$ tokens, with batch size $B$ and $H$ attention heads. Entry `attn[b, h, i, j]` is the weight that query $i$ places on key position $j$. It is causal (0 for $j > i$), and every query row sums to 1.

Split the prompt into two contiguous regions using the integer `window`:
- The **observation window** is the last `window` positions, $P, \dots, L - 1$, where $P = L - \text{window}$.
- The **prefix** is the first $P$ positions, $0, \dots, P - 1$.

The function returns, for every batch row and head, the positions whose keys and values the compressed cache keeps.

If $L \le \text{budget}$, nothing is dropped. Every head keeps all positions $0, \dots, L - 1$. Otherwise, compute the kept positions independently for each batch row $b$ and head $h$:

1. **Vote.** Each prefix position collects the attention it receives from the queries in the observation window:

   $$
   \text{vote}_j = \sum_{i = P}^{L - 1} \text{attn}_{b,h,i,j}, \qquad 0 \le j < P .
   $$

   Queries before the window do not vote, and the window's own positions ($j \ge P$) receive no vote.

2. **Pool.** Max-pool the prefix votes along the sequence with width `kernel` and stride 1. With half-width $r = \lfloor \text{kernel} / 2 \rfloor$:

   $$
   \text{pooled}_j = \max_{\substack{j - r \,\le\, j' \,\le\, j + r \\ 0 \,\le\, j' \,<\, P}} \text{vote}_{j'} .
   $$

   At the two ends of the prefix, the maximum runs only over valid prefix positions. Nothing outside the prefix participates, including the observation window.

3. **Select.** Keep the $\text{budget} - \text{window}$ prefix positions with the highest pooled votes. If two positions have equal pooled votes, choose the lower position index first.

4. **Return.** Return the selected prefix positions in **increasing order**, followed by the window's positions $P, \dots, L - 1$, giving `budget` positions per head.

Parameter rules:
- `window`, `budget`, and `kernel` are Python ints with $1 \le \text{window} \le \text{budget}$.
- `kernel` is odd and $\ge 1$. `kernel = 1` means no pooling, and `kernel` may exceed $P$.
- When `budget = window`, only the observation window is kept.

Pooling makes tied pooled votes the common case rather than an edge case. Every position within $r$ of a peak inherits the peak's value, so the tie-breaking rule determines which positions survive when the budget cuts through a plateau.

## Shapes

| name | shape | dtype |
|---|---|---|
| `attn` | `(B, H, L, L)` | float32 |
| result | `(B, H, min(L, budget))` | integer |

## Example

Consider $B = 1$, $H = 1$, $L = 10$, `window = 2`, `budget = 5`, and `kernel = 3`. The prefix is positions 0 to 7, and the observation window is positions 8 and 9. The two window query rows are:

```python
attn[0, 0, 8] = [0.02, 0.02, 0.40, 0.02, 0.02, 0.02, 0.20, 0.02, 0.28, 0.00]
attn[0, 0, 9] = [0.02, 0.02, 0.30, 0.02, 0.02, 0.02, 0.10, 0.06, 0.14, 0.30]
```

Suppose the eight queries before the window put all their weight on position 0. They do not vote, so they have no effect. Summing the two window rows over prefix positions 0 to 7 and max-pooling with $r = 1$ gives:

| position $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| vote | 0.04 | 0.04 | 0.70 | 0.04 | 0.04 | 0.04 | 0.30 | 0.08 |
| pooled ($r = 1$) | 0.04 | 0.70 | 0.70 | 0.70 | 0.04 | 0.30 | 0.30 | 0.30 |

Columns 8 and 9 are omitted because the window's self-attention weights (0.28, 0.14, and 0.30) are ignored. Position 7 is pooled over positions 6 and 7 only.

We keep `budget - window = 3` prefix positions. The three highest pooled votes form the plateau around position 2, so the result is `[1, 2, 3, 8, 9]`. This keeps the most attended position, its two neighbors, and the observation window.

With `kernel = 1`, no pooling occurs and the same votes yield `[2, 6, 7, 8, 9]`, the three highest raw votes on their own.

With `budget = 6` and `kernel = 3`, a fourth prefix position is kept. Positions 5, 6, and 7 tie at 0.30, and the lowest index wins, giving `[1, 2, 3, 5, 8, 9]`.

## Constraints

- $1 \le B \le 2$, $1 \le H \le 4$, $L \le 32$, $\text{kernel} \le 9$.
- Returned positions are compared exactly.
- Within any head, two votes are either exactly equal or differ by at least $10^{-4}$, so floating-point rounding in the sum cannot change the ranking.
- No Python loops over batch rows, heads, or positions are needed (a loop over the `kernel` offsets is fine).

## Why this matters

For a long prompt with a short answer, almost the entire KV cache comes from the prompt, and every generated token must attend over all of it. SnapKV relies on a structural pattern in attention. The weights that the end of the prompt (usually the question or instruction) places on earlier tokens predict what the generated tokens will attend to. That lets a runtime compress the cache once before the first token is generated, with no per-step eviction bookkeeping, while letting each head keep the positions it actually uses.

Pooling is what keeps the compressed cache useful. Attention often spikes on the first token of a name or number, while the answer also requires the tokens next to it. Without pooling, the SnapKV paper shows a model retrieving the country code of a phone number and hallucinating the rest. Bugs in this step fail silently. Voting over the wrong rows or pooling across the window boundary still returns `budget` valid positions per head, just not the ones the model needs.
