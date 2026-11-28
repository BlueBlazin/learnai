Implement streaming window compression for key-value caches (`compress`) and a visibility helper (`visible_entries`). A stream of key-value rows arrives in chunks of arbitrary size, and every `rate` consecutive rows form a fixed stream window that compresses into a single cache entry. Because windows belong to the stream rather than to individual calls, rows from an unfinished window must wait in a state between calls.

## Specification

```python
def compress(kv, gate, position_bias, gain, state, rate): ...  # -> (entries, state)
def visible_entries(t, rate): ...  # -> int
```

### Window compression

Number all rows in the stream $0, 1, 2, \dots$ in arrival order across calls, where each row has width $d$. For a window size `rate` (a Python `int`, $\texttt{rate} \ge 1$), window $w$ ($w \ge 0$) consists of rows $w \cdot \texttt{rate}$ through $(w + 1) \cdot \texttt{rate} - 1$. Within a window, offset $j$ refers to its $j$-th row ($0 \le j < \texttt{rate}$).

Let $c_j, z_j \in \mathbb{R}^d$ denote the `kv` and `gate` rows at offset $j$ of window $w$, and let $b_j \in \mathbb{R}^d$ denote row $j$ of `position_bias`. The compressed entry $e_w \in \mathbb{R}^d$ is computed in three steps:

1. Compute per-channel attention weights $a_j$ by taking a softmax over the `rate` rows of the window, separately for each channel:
   $$
   a_j = \frac{\exp(z_j + b_j)}{\sum_{l=0}^{\texttt{rate}-1} \exp(z_l + b_l)}
   $$
2. Pool the window's `kv` rows using those weights, where $\odot$ is the elementwise product:
   $$
   p = \sum_{j=0}^{\texttt{rate}-1} a_j \odot c_j
   $$
3. Apply RMS normalization across the $d$ channels of $p$ and scale by `gain`:
   $$
   e_w = \texttt{gain} \odot \frac{p}{\sqrt{\tfrac{1}{d}\sum_{i=0}^{d-1} p_i^2 + 10^{-6}}}
   $$

### Streaming state and `compress`

Each call to `compress` receives the next $n$ rows of the stream ($n \ge 1$) along with `state = (buf_kv, buf_gate, count)`:

- `buf_kv` and `buf_gate` hold the $r$ waiting rows of the current unfinished window ($0 \le r < \texttt{rate}$), oldest first, exactly as they arrived (with no position bias added).
- `count` is a Python `int` giving the total number of entries emitted before this call.

Before the first call, `state` contains two empty float32 tensors of shape `(0, d)` and `count = 0`.

`compress` returns `(entries, new_state)`. Let $m = (r + n) \mathbin{//} \texttt{rate}$ be the number of windows completed by this call.

- `entries` contains one row per newly completed window, oldest first. If no window completes ($m = 0$), `entries` is empty with shape `(0, d)`. Never emit an unfinished window.
- `new_state` holds the remaining $(r + n) - m \cdot \texttt{rate}$ rows in `buf_kv` and `buf_gate`, together with `count + m`.
- Do not modify any input tensor or the incoming `state` in place, so a caller can continue from the same `state` more than once.

### Entry visibility

`visible_entries(t, rate)` returns a Python `int` giving how many compressed entries a query at token position $t$ (a Python `int`, $t \ge 0$) may attend to. Row $i$ of the stream corresponds to token position $i$. Entry $w$ is visible to position $t$ if and only if the last row of window $w$ is at a position $\le t$.

The result must be exact for every valid $t$ and computed in $O(1)$ time. A loop over windows will time out for large $t$.

![The example below, with `rate` 2: four rows arrive as three and then one, and the last frame gives `visible_entries(t, 2)` for each position.](assets/window-steps.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `kv`, `gate` | `(n, d)` | float32 |
| `position_bias` | `(rate, d)` | float32 |
| `gain` | `(d,)` | float32 |
| `buf_kv`, `buf_gate` | `(r, d)`, $0 \le r <$ `rate` | float32 |
| `entries` | `(m, d)`, $m \ge 0$ | float32 |

## Example

Consider `rate = 2` and channel width `d = 2`, with four rows arriving in two calls of three rows and then one row:

```python
kv = [[4, 2], [0, 6], [1, 1], [-3, 3]]
gate = [[0, 0], [0, 0], [0, 0], [log(3), 0]]
position_bias = [[log(3), 0], [0, 0]]
gain = [1, 2]
state = (zeros((0, 2)), zeros((0, 2)), 0)

entries, state = compress(kv[:3], gate[:3], position_bias, gain, state, 2)
# entries = [[0.848528, 2.262742]]; state = ([[1, 1]], [[0, 0]], 1)
entries, state = compress(kv[3:], gate[3:], position_bias, gain, state, 2)
# entries = [[-0.632455, 2.529822]]; state = (shape (0, 2), shape (0, 2), 2)
```

In the first call, window 0 consists of rows 0 and 1. In channel 0, adding `position_bias` to `gate` gives $(\ln 3, 0)$, which produces softmax weights $(0.75, 0.25)$. In channel 1, both logits are $0$, giving weights $(0.5, 0.5)$. Pooling yields $p = (0.75 \cdot 4,\ 0.5 \cdot 2 + 0.5 \cdot 6) = (3, 4)$, whose root mean square is $\sqrt{12.5} \approx 3.5355$. Multiplying the normalized vector $(0.8485, 1.1314)$ by `gain = [1, 2]` gives $e_0 \approx [0.848528, 2.262742]$. Row 2 starts window 1 and waits in the buffer.

In the second call, the buffered row sits at offset 0 of window 1 and the new row sits at offset 1. In channel 0, the logits are $(0 + \ln 3,\ \ln 3 + 0)$, giving equal weights $(0.5, 0.5)$. Pooling gives $p = (-1, 2)$ with root mean square $\sqrt{2.5}$, producing $e_1 \approx [-0.632455, 2.529822]$ and an empty buffer with `count = 2`.

For `visible_entries(t, 2)`, the results for $t = 0, 1, 2, 3$ are `0, 1, 1, 2`. At $t = 1$ (the last row of window 0), window 0 is complete, so entry 0 is already visible.

## Constraints

- $1 \le n \le 512$, $1 \le d \le 64$, $1 \le \texttt{rate} \le 128$, total stream length at most 512 rows, $|z_j + b_j| \le 400$, and $0 \le t < 2^{31}$.
- `entries` and the buffers in the returned state are compared against a float64 reference implementation (and, for two streams, against `DeepseekV4HCACompressor` in Hugging Face Transformers) with relative tolerance `1e-4` and absolute tolerance `1e-5`. `count` and `visible_entries` are checked for exact equality.
- Outputs must be finite for every valid input, including a window whose `kv` rows are all zero (which must produce a zero entry).
- Concatenating the emitted `entries` across calls must give the same result regardless of how the stream is split into chunks.
- You may use your framework's built-in `softmax`.

## Why this matters

DeepSeek-V4 maintains two kinds of key-value memory per layer. It keeps recent tokens uncompressed and compresses the distant past in fixed windows, collapsing 128 tokens into a single cache entry in its heavily compressed layers. Because the compressed cache grows by only one entry per 128 tokens, a million-token context fits in memory.

In inference serving, a prompt never arrives all at once. It is processed in prefill chunks followed by one token per decode step. If a compressor restarts windows at call boundaries, emits partial windows early, or lets a query attend to an unfinished window, the same prompt produces different cache entries depending on how the server chunked or batched the request.
