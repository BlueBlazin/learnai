Rotary position embeddings (RoPE) encode position by rotating pairs of query and key components. Because a dot product depends only on the angle between vectors, the attention score between positions $m$ and $n$ then depends only on $m - n$.

Split a head dimension $d$ (even) into $d/2$ pairs. Pair $i$ at position $m$ is rotated by the angle $m\,\theta_i$:

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
=
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \phantom{-}\cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix},
\qquad
\theta_i = b^{-2i/d},
$$

with base $b = 10000$.

![Positions $m = 0, 1, 2, 3$ for $d = 4$ and $b = 100$. Each pair is a point in its own plane, turned counterclockwise by $m\,\theta_i$: the first pair moves one radian per position, the second a tenth of that.](assets/pair-rotation.svg)

Implement `apply_rope(x, positions, base)`.

## Interface

```python
def apply_rope(x: Tensor, positions: Tensor, base: float = 10000.0) -> Tensor:
    """
    x: (B, H, T, d) with d even.    positions: (T,) integer positions.
    Returns x rotated, same shape.
    """
```

`positions` is an argument, not `arange(T)`, because during cached decoding a single new token arrives with position $t$, not $0$.

### Example 1

With $d = 2$ there is one pair and $\theta_0 = 1$, so position $m$ rotates by $m$ radians.

```python
x = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])     # (1, 1, 2, 2)
apply_rope(x, torch.tensor([0, 1]))
# tensor([[[[1.0000, 0.0000], [0.5403, 0.8415]]]])
```

## What the judges check

| Judge | Check |
|---|---|
| numeric | output matches the oracle to $10^{-5}$ on random inputs |
| property | $\lVert x' \rVert = \lVert x \rVert$ for every position (rotations preserve norm) |
| property | $\langle \operatorname{rope}(q, m), \operatorname{rope}(k, n) \rangle$ is unchanged when $m$ and $n$ shift by the same offset |
| speed | at $T = 2048$, $d = 128$: no slower than $1.5\times$ the reference, median of 9 interleaved runs |

> [!TIP]
> The speed case fails for implementations that loop over positions or pairs in Python. Build the angles as one outer product.

## Constraints

- $d \in \{2, 16, 64, 128\}$, $1 \le T \le 4096$, positions up to $2^{17}$.
- float32 throughout; compute angles in float32, not float64 (MLX has no float64 on the GPU).
