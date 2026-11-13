Compute the angles once, rotate the two halves, interleave.

```python
half = x.shape[-1] // 2
freqs = base ** (-torch.arange(half, dtype=torch.float32) / half)
angles = positions.to(torch.float32)[:, None] * freqs[None, :]      # (T, d/2)
cos, sin = angles.cos(), angles.sin()
even, odd = x[..., 0::2], x[..., 1::2]
out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
return out.flatten(-2)
```

## Why relative position falls out

Treat each pair as a complex number. Rotation multiplies $q$ by $e^{\mathrm{i} m \theta}$ and $k$ by $e^{\mathrm{i} n \theta}$, and

$$
\operatorname{Re}\bigl( q\, e^{\mathrm{i} m\theta}\; \overline{k\, e^{\mathrm{i} n\theta}} \bigr) = \operatorname{Re}\bigl( q \bar{k}\; e^{\mathrm{i}(m-n)\theta} \bigr).
$$

Only $m - n$ survives.
