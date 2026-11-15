Implement a KV cache that stores old tokens as small integer codes and only the most
recent tokens in float32. Each token is quantized exactly once, when it leaves the
float window, with a scale and a bias for every short run of channels; the work is in
the bookkeeping of what leaves the window when, and in a layout the attention can
read back.

## Specification

```python
def quantized_append(cache, k_new, v_new, bits, group_size, keep_recent): ...  # -> cache
def read_cache(cache): ...  # -> (keys, values)
```

Keys and values have the layout of [`kv-cache-append`](../kv-cache-append/statement.md):
shape `(B, H, S, D)` for batch, heads, sequence, and head dimension, oldest token
first. Number the tokens $0, 1, 2, \dots$ in the order they are appended.

**The cache.** After $S$ tokens have been appended, the cache is a tuple
`(k_quant, v_quant, k_recent, v_recent)`:

- `k_recent` and `v_recent` hold the newest $S_r = \min(S, \text{keep\_recent})$
  tokens as float32 tensors of shape `(B, H, S_r, D)`, bit for bit as they were
  appended.
- `k_quant` and `v_quant` hold the other $S_q = S - S_r$ tokens, the oldest ones. Each
  is a tuple `(codes, scale, bias)`: `codes` is a **uint8** tensor of shape
  `(B, H, S_q, D)`, and `scale` and `bias` are float32 tensors of shape
  `(B, H, S_q, D / group_size)`.

$S_q$ or $S_r$ may be 0. The tensors then have an axis of length 0; they are never
`None`. Before the first call the cache itself is `None`.

**The quantization layout.** Groups run along the channel axis, per token. The `D`
channels of one token in one head (of one batch row) are cut into `D / group_size`
consecutive groups, and group $g$ holds the channels $j$ with
$g \cdot \text{group\_size} \le j < (g + 1) \cdot \text{group\_size}$. Every group of
every token is quantized on its own. With $L = 2^{\text{bits}} - 1$, the largest code,
and $x_j$ the float values of the group:

$$
\text{bias} = \min_j x_j,
\qquad
\text{scale} = \begin{cases}
\dfrac{\max_j x_j - \min_j x_j}{L} & \text{if the maximum exceeds the minimum}, \\[1.5ex]
1 & \text{if every element of the group is equal},
\end{cases}
$$

$$
\text{code}_j = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x_j - \text{bias}}{\text{scale}}\Big),\ 0,\ L\Big),
\qquad
\hat x_j = \text{code}_j \cdot \text{scale} + \text{bias}.
$$

`scale[b, h, t, g]` and `bias[b, h, t, g]` belong to group $g$ of token $t$. Keys and
values are quantized separately, by the same rule. This is the group-wise affine
layout of [`groupwise-4bit`](../../quantization/groupwise-4bit/statement.md), applied
to each token's vector instead of each row of a weight matrix; here one uint8 holds
one code, whatever `bits` is.

**`quantized_append(cache, k_new, v_new, bits, group_size, keep_recent)`** appends the
keys and values of `T` new tokens and returns the cache for $S + T$ tokens. Every token
that is no longer among the newest `keep_recent` is quantized in this call, from its
float32 values, and its codes, scale, and bias are appended to the quantized part in
token order. Codes, scales, and biases that are already in the cache are carried over
unchanged: a token is quantized once in its life.

**`read_cache(cache)`** returns `(keys, values)`, two float32 tensors of shape
`(B, H, S_q + S_r, D)`: the dequantized values $\hat x$ of the quantized tokens,
followed by the float window, oldest token first. This is what an attention step
attends over. It takes no `group_size`: the shapes in the cache determine it.

Details:

- `bits` is a Python int from 2 to 8, `group_size` a Python int that divides `D`
  (it may be 1 or `D`), `keep_recent` a Python int $\ge 0$. They are the same in every
  call on one cache.
- `T` may be larger than `keep_recent`. The oldest tokens of the new chunk are then
  quantized in the call that appends them, after the tokens of the old window.
- With `keep_recent = 0` nothing stays float. While $S \le \text{keep\_recent}$ nothing
  is quantized, and `read_cache` returns exactly what was appended.
- A constant group has `scale = 1` and codes 0, and is read back exactly. No output
  may be NaN or infinite.
- $\operatorname{round}$ is rounding to the nearest integer, ties to even, which is
  what `round` does in PyTorch, JAX, and MLX. Where codes are compared exactly, no
  value of $(x - \text{bias}) / \text{scale}$ is within $0.05$ of a tie.
- State in, state out: the function returns the cache to continue with and does not
  modify the tensors it was given.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k_new`, `v_new` | `(B, H, T, D)` | float32 |
| `codes` in `k_quant`, `v_quant` | `(B, H, S_q, D)` | uint8 |
| `scale`, `bias` in `k_quant`, `v_quant` | `(B, H, S_q, D / group_size)` | float32 |
| `k_recent`, `v_recent` | `(B, H, S_r, D)` | float32 |
| `keys`, `values` from `read_cache` | `(B, H, S_q + S_r, D)` | float32 |

## Example

`bits = 2` (codes 0 to 3, $L = 3$), `group_size = 4`, `keep_recent = 1`, one batch
row, one head, `D = 8`, so every token has two groups. The keys of three tokens:

```python
k0 = [0.0, 3.0, 0.9, 2.2, 5.0, 5.0, 5.0, 5.0]
k1 = [-1.0, 0.5, -0.4, 0.1, 10.0, 40.0, 22.0, 31.0]
k2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

**Call 1** appends tokens 0 and 1 to an empty cache (`cache = None`). Only one token
may stay float, so token 0 is quantized on arrival and token 1 is the float window.

| group | values | bias | scale | $(x - \text{bias}) / \text{scale}$ | codes | read back |
|---|---|---|---|---|---|---|
| token 0, group 0 | `0.0, 3.0, 0.9, 2.2` | 0 | $3 / 3 = 1$ | `0, 3, 0.9, 2.2` | `0, 3, 1, 2` | `0.0, 3.0, 1.0, 2.0` |
| token 0, group 1 | `5.0, 5.0, 5.0, 5.0` | 5 | 1 (constant) | `0, 0, 0, 0` | `0, 0, 0, 0` | `5.0, 5.0, 5.0, 5.0` |

**Call 2** appends token 2. Token 1 leaves the window and is quantized now; the codes
of token 0 are not touched.

| group | values | bias | scale | $(x - \text{bias}) / \text{scale}$ | codes | read back |
|---|---|---|---|---|---|---|
| token 1, group 0 | `-1.0, 0.5, -0.4, 0.1` | $-1$ | $1.5 / 3 = 0.5$ | `0, 3, 1.2, 2.2` | `0, 3, 1, 2` | `-1.0, 0.5, -0.5, 0.0` |
| token 1, group 1 | `10, 40, 22, 31` | 10 | $30 / 3 = 10$ | `0, 3, 1.2, 2.1` | `0, 3, 1, 2` | `10, 40, 20, 30` |

After call 2 the key part of the cache is

```python
codes = [[0, 3, 1, 2, 0, 0, 0, 0], [0, 3, 1, 2, 0, 3, 1, 2]]  # shape (1, 1, 2, 8)
scale = [[1.0, 1.0], [0.5, 10.0]]  # shape (1, 1, 2, 2)
bias = [[0.0, 5.0], [-1.0, 10.0]]
k_recent = [k2]  # shape (1, 1, 1, 8), exact
```

and `read_cache` returns the keys `[0, 3, 1, 2, 5, 5, 5, 5]`,
`[-1, 0.5, -0.5, 0, 10, 40, 20, 30]`, and `k2`. The two groups of token 1 differ in
range by a factor of 20, and each is still resolved into four levels. With one range
for the whole token (from $-1$ to $40$, a step of $13.7$), its first group would
collapse onto the single code 0.

The test passes `v = 2 * k`: the values go through the same steps, with scales and
biases twice as large and the same codes.

## Constraints

- $1 \le B \le 2$, $1 \le H \le 4$, $D \le 16$, $1 \le T \le 16$, at most 32 tokens per
  cache; $|x| \le 100$.
- Codes are compared **exactly** and must be uint8. Scales, biases, and dequantized
  values are compared to a relative tolerance of `1e-4` and an absolute tolerance of
  `1e-5`. The float window and the shapes are compared exactly.
- Every dequantized element is within half a step, $\text{scale} / 2$, of the float
  value it came from (with `scale` the scale of its group as defined above).
- Quality is measured with the house metric of
  [`kl-divergence-from-logits`](../../foundations/kl-divergence-from-logits/statement.md):
  a tiny model (head dimension 16, `group_size = 8`, `keep_recent = 4`) caches a
  24-token prompt through your functions, layer by layer, and scores 8 further tokens
  against what `read_cache` returns. The mean KL divergence from its predictions with
  the float cache must be at most $10^{-5}$ nats at 8 bits (the reference measures
  $2.9 \cdot 10^{-6}$) and at most $1.5 \cdot 10^{-3}$ nats at 4 bits (the reference
  measures $5.3 \cdot 10^{-4}$).
- No Python loops over tokens, heads, groups, or elements are needed.

## Why this matters

For a long context the KV cache, not the weights, is what fills memory: every token
keeps a key and a value vector in every layer, usually in 16-bit floats. Storing them
in 8 or 4 bits halves or quarters that. Engines differ in where the grid lives.
mlx-lm's `QuantizedKVCache` quantizes each incoming token's keys and values group-wise
along the channel axis (`kv_bits`, `kv_group_size`), which is this problem's layout,
and switches a cache to it once the context reaches `quantized_kv_start` tokens.
Hugging Face Transformers' quantized cache keeps the newest tokens in full precision
(`residual_length`), which is this problem's float window: the tokens the next step
attends to most are the ones just produced, and they are the cheapest to keep exact.
llama.cpp chooses a storage type per tensor with `--cache-type-k` and
`--cache-type-v`. The mistakes are quiet: a window that is one token off, one range
shared across groups, or codes that are truncated instead of rounded all produce a
model that keeps talking, a little worse, and only a metric on the logits shows it.
