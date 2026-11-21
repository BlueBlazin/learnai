DeepSeek-V4 carries $N$ residual streams per token instead of one. Around every
sublayer a hyper-connection computes, from the streams themselves, how to collapse them
into the sublayer's input, how to spread its output back over them, and an $N \times N$
matrix that mixes the old streams, pushed toward a doubly stochastic matrix by a fixed
number of Sinkhorn iterations. Implement the weights and the merge.

## Specification

```python
def hc_weights(streams, fn, base, scale, iters, eps): ...  # -> (collapsed, post, comb)
def hc_merge(streams, y, post, comb): ...  # -> (T, N, D)
```

Everything is per token; let $X \in \mathbb{R}^{N \times D}$ be the streams of one
token, with rows $X_0, \dots, X_{N-1}$.

**`hc_weights`.** Flatten $X$ row after row into $v \in \mathbb{R}^{ND}$ (stream 0's
channels first), normalize it as a whole, without a gain, and apply one linear map:

$$
u = \frac{v}{\sqrt{\operatorname{mean}(v^2) + 10^{-6}}}, \qquad
r = u\,\texttt{fn}^\top \in \mathbb{R}^{2N + N^2} .
$$

The $10^{-6}$ is fixed; it is not `eps`. The first $N$ entries of $r$ and of `base`
belong to `pre`, the next $N$ to `post`, and the last $N^2$ to `comb`, read row after
row as an $N \times N$ matrix (entry $(j, n)$ is number $jN + n$):

$$
\begin{aligned}
\texttt{pre} &= \sigma(\texttt{scale}_0\, r_\text{pre} + \texttt{base}_\text{pre}) + \texttt{eps}, \\
\texttt{post} &= 2\,\sigma(\texttt{scale}_1\, r_\text{post} + \texttt{base}_\text{post}), \\
C &= \operatorname{softmax}(\texttt{scale}_2\, r_\text{comb} + \texttt{base}_\text{comb}) + \texttt{eps},
\end{aligned}
$$

with $\sigma$ the logistic sigmoid and the softmax over the last axis (each row of the
matrix). Then $C$ is normalized by Sinkhorn steps, each applied to the result of the
one before:

1. divide every **column** by (its sum $+$ `eps`);
2. repeat `iters - 1` times: divide every **row** by (its sum $+$ `eps`), then every
   column by (its sum $+$ `eps`).

The result is `comb`. So `iters = 1` normalizes the columns once, and the last step is
always a column step. The sublayer's input is the streams weighted by `pre`:

$$
\texttt{collapsed} = \sum_{n=0}^{N-1} \texttt{pre}_n\, X_n .
$$

**`hc_merge`.** `y` is what the sublayer returned for `collapsed`, one vector of width
$D$ per token. Output stream $n$ is

$$
X'_n = \texttt{post}_n\, y + \sum_{j=0}^{N-1} \texttt{comb}_{j,n}\, X_j ,
$$

so `comb` acts transposed: column $n$ of `comb` holds the weights of the old streams in
new stream $n$. `streams` here are the streams the weights were computed from.

`iters` is a Python int $\ge 1$ and `eps` a Python float $> 0$. `hc_weights` returns the
tuple `(collapsed, post, comb)`. No function modifies a tensor it is given.

## Shapes

| name | shape | dtype |
|---|---|---|
| `streams`, result of `hc_merge` | `(T, N, D)` | float32 |
| `fn`; `base`; `scale` | `(2N + N*N, N*D)`; `(2N + N*N,)`; `(3,)` | float32 |
| `collapsed`, `y` | `(T, D)` | float32 |
| `post` | `(T, N)` | float32 |
| `comb` | `(T, N, N)` | float32 |

## Example

One token, two streams of width 2. `fn` is zero, so the weights come from `base` alone:

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

`pre` is $(\sigma(0), \sigma(\ln 3)) = (0.5, 0.75)$, so `collapsed` is
$0.5\,(1, 2) + 0.75\,(3, -1)$. `post` is $2\,(\sigma(0), \sigma(-\ln 3)) = (1, 0.5)$.
The softmax of the rows $(0, \ln 3)$ and $(0, 0)$ is $(0.25, 0.75)$ and $(0.5, 0.5)$;
the column sums are $0.75$ and $1.25$, and dividing by them gives `comb` (the `eps`
terms change the sixth digit). With `iters = 2` a row step and a column step follow,
and `comb` is `[[0.363636, 0.631579], [0.636364, 0.368421]]`.

In the merge, new stream 0 is $1 \cdot (10, 20) + \tfrac13 (1, 2) + \tfrac23 (3, -1)$,
using column 0 of `comb`, and new stream 1 is
$0.5 \cdot (10, 20) + 0.6\,(1, 2) + 0.4\,(3, -1)$.

## Constraints

- $1 \le T \le 64$, $2 \le N \le 8$, $1 \le D \le 64$, $1 \le \texttt{iters} \le 20$,
  $10^{-8} \le \texttt{eps} \le 10^{-2}$; the arguments of the sigmoids and of the
  softmax are within $\pm 60$; stream entries have magnitudes from $10^{-3}$ to 100.
- Results are compared with the reference and with arrays computed by
  `DeepseekV4HyperConnection` and `DeepseekV4DecoderLayer` of Hugging Face
  Transformers, at 1, 2, 3, and 20 iterations: relative tolerance `1e-4`, absolute
  tolerance `1e-6` (`1e-5` for the streams after a whole layer).
- When the $N^2$ softmax inputs of every token span at most 2 (largest minus smallest)
  and `eps` is $10^{-6}$: every column of `comb` sums to 1 within `1e-5` for every
  `iters`, and every row sums to 1 within `1e-4` for `iters = 20`.
- You may use your framework's `softmax` and `sigmoid`.

## Why this matters

In DeepSeek-V4 the embedding is copied to all $N = 4$ streams, and each of a layer's
two sublayers runs as `collapsed, post, comb = hc_weights(...)`,
`y = sublayer(norm(collapsed))`, `streams = hc_merge(streams, y, post, comb)`, with 20
Sinkhorn iterations. The mixing matrix is applied at every sublayer of every layer, so
a stack multiplies well over a hundred of them together: a matrix whose columns do not
sum to 1 rescales the streams each time. Reading the matrix the wrong way round, or
normalizing rows last, gives streams of the right shape and a model that degrades with
depth instead of failing.
