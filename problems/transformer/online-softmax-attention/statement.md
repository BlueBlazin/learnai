Compute scaled dot-product attention when the keys and values do not fit in memory at
once: they arrive in blocks, each block is seen once, and the result must equal
attention over all of them. A softmax needs its normalizer, which is only known at the
end, so the function carries a running maximum, a running normalizer, and a running
output, and rescales the last two whenever the maximum grows. This is the online
softmax at the core of FlashAttention.

## Specification

```python
def blockwise_attention(q, kv_blocks): ...
```

- `q`: float32 tensor of shape `(T, D)`, the queries.
- `kv_blocks`: an iterator that yields tuples `(k, v, mask)`:
  - `k`: float32 tensor of shape `(n, D)`, the keys of the block;
  - `v`: float32 tensor of shape `(n, D_v)`, their values;
  - `mask`: bool tensor of shape `(T, n)`; `mask[t, j]` is `True` when query `t` may
    attend to key `j` of the block.

  The block size $n \ge 1$ varies from block to block; $D_v$ is the same for all.
  There is at least one block, and the number of blocks is not known in advance.
- Returns a float32 tensor of shape `(T, D_v)`.

Let $k_1, \ldots, k_N$ and $v_1, \ldots, v_N$ be the keys and values of all blocks in
order, and $M_{t,j}$ the corresponding mask entries. The score of query $t$ for key
$j$ is

$$
s_{t,j} = \frac{q_t \cdot k_j}{\sqrt{D}},
$$

and the result is ordinary attention over the visible keys:

$$
\text{out}_t = \sum_{j \,:\, M_{t,j}} \frac{e^{s_{t,j}}}{\sum_{i \,:\, M_{t,i}} e^{s_{t,i}}}\; v_j .
$$

Every query has at least one visible key among all blocks. Nothing else is promised:

- **A block may be entirely masked** for some queries, or for all of them.
- **A query may see nothing at first.** Its first visible key can be in any block,
  so the blocks before it contribute nothing to that query, and must not turn its
  running state into NaN.
- **Scores are large.** $|s_{t,j}|$ reaches several hundred, far beyond the range in
  which $e^{s}$ is a float32, and different queries live on different scales.

**One pass, one block at a time.** `kv_blocks` can be iterated once. The memory the
function may use is $O(T \cdot D_v)$ of running state plus one block; it must not
collect the blocks, or anything of their size, to process them at the end. The
iterator enforces the first half of that: it tracks which of the blocks it has
handed out are still referenced, and **when the next block is requested, at most
one earlier block may still be alive** (the one just processed, which a `for` loop
still holds in its loop variables). The same holds for the request that ends the
iteration. The measured number is reported as `peak_blocks_held`, with a limit of 1.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(T, D)` | float32 |
| `k` (per block) | `(n, D)` | float32 |
| `v` (per block) | `(n, D_v)` | float32 |
| `mask` (per block) | `(T, n)` | bool |
| result | `(T, D_v)` | float32 |

## Example

Two queries with $D = 1$ (so the scale is 1), three keys in two blocks:

```python
q = [[1.0], [100.0]]
block_1 = (k=[[0.0], [1.0]], v=[[1.0], [2.0]], mask=[[True, True], [True, True]])
block_2 = (k=[[3.0]],        v=[[4.0]],        mask=[[True], [False]])

blockwise_attention(q, iter([block_1, block_2]))  # [[3.645579], [2.0]]
```

Query 0 has scores $[0, 1]$ in the first block and $[3]$ in the second. In full,
$\operatorname{softmax}([0, 1, 3]) = [0.0420, 0.1142, 0.8438]$ and the output is
$0.0420 \cdot 1 + 0.1142 \cdot 2 + 0.8438 \cdot 4 = 3.6456$. Processed block by
block, with sums kept relative to the largest score so far:

| after | largest score $m$ | $\sum e^{s - m}$ | $\sum e^{s - m}\, v$ |
|---|---|---|---|
| block 1 | 1 | $e^{-1} + 1 = 1.3679$ | $e^{-1} \cdot 1 + 1 \cdot 2 = 2.3679$ |
| block 2 | 3 | $1.3679\, e^{-2} + 1 = 1.1851$ | $2.3679\, e^{-2} + 4 = 4.3205$ |

and $4.3205 / 1.1851 = 3.6456$. When the maximum moved from 1 to 3, both sums from
block 1 were multiplied by $e^{1 - 3}$.

Query 1 has scores $[0, 100]$ in the first block: $e^{100}$ is not a float32, and
relative to the maximum the weights are $[e^{-100}, 1]$. The second block is masked
for this query and changes nothing. The output is $2.0$.

## Constraints

- $1 \le T \le 64$, $1 \le D \le 64$, $1 \le D_v \le 64$; at most 64 blocks of 1 to
  64 keys each. $|s_{t,j}| \le 1000$.
- The result is compared with attention over all keys at once to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-5`. A NaN is always wrong.
- `peak_blocks_held` must be at most 1, in every case.
- The function must read every block: it returns only after the iterator has ended.
- A Python loop over the blocks is expected; no loop over queries or keys is needed.

## Why this matters

Attention over a long context is a $T \times N$ table of scores that nobody can
afford to store: at 100,000 tokens it is 40 GB in float32 for one head. FlashAttention
(Dao et al., 2022) never builds it. It walks over the keys in blocks that fit in the
GPU's fast memory and keeps, per query, exactly the three running quantities of this
problem; the same recurrence lets attention run over a KV cache that is paged,
chunked across devices, or streamed from disk. The rescaling step is where
implementations go wrong, and the errors are data dependent: a missing rescale is
invisible until a later block contains a larger score, and the `-inf - (-inf)` of a
fully masked block appears only with causal masks or padding, where it turns one
query's output into NaN and, one layer later, the whole sequence.
