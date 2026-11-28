DeepSeek-V4 carries $N$ residual streams per token instead of one. Around each sublayer, a hyper-connection computes input-dependent weights that collapse the streams into one sublayer input, broadcast the sublayer output back to the streams, and mix the old streams with an $N \times N$ matrix normalized by Sinkhorn iterations. Implement the weight computation and the merge step.

![This problem is the residual connections around the two sublayers, which here carry several streams.](assets/map.svg)

## Specification

```python
def hc_weights(streams, fn, base, scale, iters, eps): ...  # -> (collapsed, post, comb)
def hc_merge(streams, y, post, comb): ...  # -> (T, N, D)
```

`streams` holds $T$ tokens, each with $N$ residual streams of hidden width $D$. Both functions operate independently on each token. For a single token, let $X \in \mathbb{R}^{N \times D}$ denote its streams, with rows $X_0, \dots, X_{N-1}$.

**`hc_weights`.** Flatten $X$ row by row into $v \in \mathbb{R}^{ND}$, placing stream 0's channels first. Normalize $v$ as a single vector with RMS normalization (no learnable gain) and apply one linear projection:

$$
u = \frac{v}{\sqrt{\operatorname{mean}(v^2) + 10^{-6}}}, \qquad
r = u\,\texttt{fn}^\top \in \mathbb{R}^{2N + N^2}.
$$

The $10^{-6}$ term in the norm is a fixed constant, not `eps`. Split both $r$ and `base` into three contiguous slices:

- the first $N$ entries form $r_\text{pre}$ and $\texttt{base}_\text{pre}$,
- the next $N$ entries form $r_\text{post}$ and $\texttt{base}_\text{post}$, and
- the last $N^2$ entries form $r_\text{comb}$ and $\texttt{base}_\text{comb}$, reshaped row by row into an $N \times N$ matrix where entry $(j, n)$ comes from flat index $jN + n$.

Scale each slice by its corresponding entry of `scale`, add the matching slice of `base`, and apply the activation:

$$
\begin{aligned}
\texttt{pre} &= \sigma(\texttt{scale}_0\, r_\text{pre} + \texttt{base}_\text{pre}) + \texttt{eps}, \\
\texttt{post} &= 2\,\sigma(\texttt{scale}_1\, r_\text{post} + \texttt{base}_\text{post}), \\
C &= \operatorname{softmax}(\texttt{scale}_2\, r_\text{comb} + \texttt{base}_\text{comb}) + \texttt{eps}.
\end{aligned}
$$

Here $\sigma$ is the logistic sigmoid, and $\operatorname{softmax}$ is applied over the last axis (each row of the $N \times N$ matrix).

Next, push $C$ toward a doubly stochastic matrix (non-negative entries with every row and column summing to 1) using `iters` Sinkhorn normalization steps. Each step operates on the output of the previous one:

1. Divide every **column** by its column sum plus `eps`.
2. Repeat `iters - 1` times: divide every **row** by its row sum plus `eps`, then divide every **column** by its column sum plus `eps`.

The resulting matrix is `comb`. When `iters = 1`, only step 1 runs. For any `iters >= 1`, the final step is always a column normalization.

Finally, combine the $N$ streams into a single sublayer input vector weighted by `pre`:

$$
\texttt{collapsed} = \sum_{n=0}^{N-1} \texttt{pre}_n\, X_n.
$$

Return the tuple `(collapsed, post, comb)`.

**`hc_merge`.** Given the original `streams`, the sublayer output `y` (computed from `collapsed`), and `post` and `comb` from `hc_weights`, compute each output stream $n \in \{0, \dots, N-1\}$ per token:

$$
X'_n = \texttt{post}_n\, y + \sum_{j=0}^{N-1} \texttt{comb}_{j,n}\, X_j.
$$

Because the sum runs over the first index $j$, `comb` acts transposed. Column $n$ of `comb` gives the weights of the old streams that form new stream $n$.

![One hyper-connection around a sublayer, with the weights of the example below. Column $n$ of `comb` mixes the old streams into new stream $n$.](assets/streams.svg)

`iters` is a Python `int` $\ge 1$ and `eps` is a Python `float` $> 0$. Neither function may modify any input tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `streams`, output of `hc_merge` | `(T, N, D)` | float32 |
| `fn` | `(2N + N*N, N*D)` | float32 |
| `base` | `(2N + N*N,)` | float32 |
| `scale` | `(3,)` | float32 |
| `collapsed`, `y` | `(T, D)` | float32 |
| `post` | `(T, N)` | float32 |
| `comb` | `(T, N, N)` | float32 |

## Example

Consider one token with two streams of width 2. Because `fn` is all zeros, the weights come from `base` alone:

```python
streams = [[[1.0, 2.0], [3.0, -1.0]]]
fn = zeros((8, 4))
base = [0, log(3), 0, -log(3), 0, log(3), 0, 0]  # pre | post | comb, row after row
scale = [1.0, 1.0, 1.0]

collapsed, post, comb = hc_weights(streams, fn, base, scale, 1, 1e-6)
# collapsed = [[2.75, 0.25]]   post = [[1.0, 0.5]]
# comb = [[[0.333333, 0.6], [0.666667, 0.4]]]

hc_merge(streams, [[10.0, 20.0]], post, comb)
# [[[12.333333, 20.0], [6.8, 10.8]]]
```

- `pre` is $(\sigma(0), \sigma(\ln 3)) = (0.5, 0.75)$, so `collapsed` is $0.5\,(1, 2) + 0.75\,(3, -1) = (2.75, 0.25)$.
- `post` is $2\,(\sigma(0), \sigma(-\ln 3)) = (1.0, 0.5)$.
- The row-wise softmax of $(0, \ln 3)$ and $(0, 0)$ gives rows $(0.25, 0.75)$ and $(0.5, 0.5)$. Their column sums are $0.75$ and $1.25$. Dividing each column by its sum produces `comb` (the `eps` terms only affect the sixth decimal place).
- With `iters = 2`, one row step and one column step follow, yielding `comb = [[0.363636, 0.631579], [0.636364, 0.368421]]`.
- In `hc_merge`, new stream 0 uses column 0 of `comb`, giving $1.0 \cdot (10, 20) + \tfrac{1}{3}(1, 2) + \tfrac{2}{3}(3, -1) = (12.333333, 20.0)$. New stream 1 uses column 1, giving $0.5 \cdot (10, 20) + 0.6\,(1, 2) + 0.4\,(3, -1) = (6.8, 10.8)$.

## Constraints

- $1 \le T \le 64$, $2 \le N \le 8$, $1 \le D \le 64$, $1 \le \texttt{iters} \le 20$, and $10^{-8} \le \texttt{eps} \le 10^{-2}$.
- All sigmoid and softmax inputs lie in $[-60, 60]$. Stream entries have magnitudes from $10^{-3}$ to $100$.
- Outputs are checked against the reference and against `DeepseekV4HyperConnection` and `DeepseekV4DecoderLayer` from Hugging Face Transformers at 1, 2, 3, and 20 iterations. The tolerances are `rtol = 1e-4` and `atol = 1e-6` (`atol = 1e-5` for the streams after a full layer).
- When every token's $N^2$ softmax inputs span at most 2 (largest minus smallest) and `eps` is `1e-6`, every column of `comb` must sum to 1 within `1e-5` for every `iters`, and every row must sum to 1 within `1e-4` for `iters = 20`.
- You may use your framework's `softmax` and `sigmoid`.

## Why this matters

In DeepSeek-V4, the token embedding is copied into all $N = 4$ streams. Each layer wraps its attention and MLP sublayers with `collapsed, post, comb = hc_weights(...)`, `y = sublayer(norm(collapsed))`, and `streams = hc_merge(streams, y, post, comb)`, using 20 Sinkhorn iterations.

Because `comb` is applied at every sublayer, a deep model multiplies well over a hundred mixing matrices in sequence. If a matrix's columns do not sum to 1, the residual stream rescales exponentially with depth. Reading `comb` untransposed or ending Sinkhorn on a row step still produces tensors of the right shape, so the bug does not crash. Instead, it quietly degrades quality as depth grows.
