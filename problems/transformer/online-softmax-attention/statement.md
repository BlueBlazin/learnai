Compute scaled dot-product attention when keys and values arrive one block at a time from a single-pass iterator. The output must match standard attention over all keys combined, while keeping memory usage independent of the total number of keys.

![This problem is attention, computed one block of keys at a time with a running softmax.](assets/map.svg)

## Specification

```python
def blockwise_attention(q, kv_blocks): ...
```

- `q`: query matrix containing $T$ queries of dimension $D$.
- `kv_blocks`: a single-pass iterator yielding `(k, v, mask)` tuples. Each block contains $n \ge 1$ keys `k` of dimension $D$, values `v` of dimension $D_v$, and a boolean visibility `mask` where `mask[t, j]` is `True` if query `t` may attend to key `j` of that block. The block size $n$ may vary from block to block, while $D_v$ is constant across all blocks. There is at least one block, and the total number of blocks is unknown in advance.
- Returns the attention output for all $T$ queries, with dimension $D_v$.

Let $N$ be the total number of keys across all blocks, with keys $k_1, \ldots, k_N$, values $v_1, \ldots, v_N$, and mask entries $M_{t,j}$ concatenated in block order. The scaled dot-product score between query $t$ and key $j$ is

$$
s_{t,j} = \frac{q_t \cdot k_j}{\sqrt{D}}.
$$

The required output for query $t$ is standard attention over its visible keys:

$$
\text{out}_t = \sum_{j \,:\, M_{t,j}} \frac{e^{s_{t,j}}}{\sum_{i \,:\, M_{t,i}} e^{s_{t,i}}}\; v_j.
$$

Every query has at least one visible key across all blocks, but no other guarantees hold:

- **Fully masked blocks:** A block may be completely masked for some queries, or for all queries.
- **Delayed first visible key:** A query's first visible key may appear in any block. Earlier blocks contribute nothing to that query and must not produce `NaN` in its running state.
- **Large scores:** $|s_{t,j}|$ can reach several hundred, well outside the range where $e^s$ is representable in `float32`. Different queries can also have very different score scales.

**Single-pass streaming.** `kv_blocks` can be iterated only once. Your function may use $O(T \cdot D_v)$ running state plus one block of working memory. Do not collect the blocks, or concatenate data proportional to all blocks, to compute attention at the end. The test iterator tracks how many yielded blocks are still referenced. Whenever the next block is requested (and when iteration ends), at most one earlier block may still be alive—the block just processed, which a `for` loop holds in its loop variables until rebound. The judge reports this count as `peak_blocks_held`, with a limit of 1.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(T, D)` | float32 |
| `k` (per block) | `(n, D)` | float32 |
| `v` (per block) | `(n, D_v)` | float32 |
| `mask` (per block) | `(T, n)` | bool |
| result | `(T, D_v)` | float32 |

## Example

Consider two queries with $D = 1$ (so $1/\sqrt{D} = 1$) and three keys split across two blocks:

```python
q = [[1.0], [100.0]]
block_1 = (k=[[0.0], [1.0]], v=[[1.0], [2.0]], mask=[[True, True], [True, True]])
block_2 = (k=[[3.0]],        v=[[4.0]],        mask=[[True], [False]])

blockwise_attention(q, iter([block_1, block_2]))  # [[3.645579], [2.0]]
```

Query 0 has scores $[0, 1]$ in block 1 and $[3]$ in block 2. Over all keys at once, $\operatorname{softmax}([0, 1, 3]) = [0.0420, 0.1142, 0.8438]$, giving $0.0420 \cdot 1 + 0.1142 \cdot 2 + 0.8438 \cdot 4 = 3.6456$. Processed block by block with running sums shifted by the largest score seen so far:

| after | largest score $m$ | $\sum e^{s - m}$ | $\sum e^{s - m}\, v$ |
|---|---|---|---|
| block 1 | 1 | $e^{-1} + 1 = 1.3679$ | $e^{-1} \cdot 1 + 1 \cdot 2 = 2.3679$ |
| block 2 | 3 | $1.3679\, e^{-2} + 1 = 1.1851$ | $2.3679\, e^{-2} + 4 = 4.3205$ |

Dividing the final sums gives $4.3205 / 1.1851 = 3.6456$. When the running maximum increases from 1 to 3, both sums from block 1 are multiplied by $e^{1 - 3}$.

![The two queries of the example, after each step: the maximum, the normalizer, and the output.](assets/online-steps.svg)

Query 1 has scores $[0, 100]$ in block 1. Computing $e^{100}$ directly overflows `float32`, so shifting by the maximum 100 gives weights $[e^{-100}, 1]$. Block 2 is masked for query 1 and leaves its state unchanged, yielding $2.0$.

## Constraints

- $1 \le T \le 64$, $1 \le D \le 64$, $1 \le D_v \le 64$, at most 64 blocks of 1 to 64 keys each, and $|s_{t,j}| \le 1000$.
- Results are compared with full attention over all keys to a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. Any `NaN` fails.
- `peak_blocks_held` must be at most 1 in every test case.
- The function must consume every block and return only after the iterator ends.
- A Python loop over the blocks is expected. No loop over individual queries or keys is needed.

## Why this matters

Standard attention builds a $T \times N$ score matrix. At 100,000 tokens, that matrix takes 40 GB in `float32` for a single head. FlashAttention (Dao et al., 2022) avoids materializing it by streaming key-value blocks through fast on-chip GPU memory and maintaining these three running quantities per query. The same recurrence also powers attention over KV caches that are paged, sharded across devices, or streamed from disk.

Bugs in online softmax are data-dependent and easy to miss. A missing rescale stays hidden until a later block contains a larger score. A `-inf - (-inf)` error from a fully masked block appears only with causal masks or padding, turning one query into `NaN` and corrupting the entire sequence by the next layer.
