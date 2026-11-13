```python
lo, hi = min(x.min().item(), 0.0), max(x.max().item(), 0.0)
scale = (hi - lo) / 255.0 or 1.0
zero_point = round(-128 - lo / scale)
q = torch.clamp(torch.round(x / scale) + zero_point, -128, 127).to(torch.int8)
```

Because $z$ is an integer and $0$ lies in $[\alpha, \beta]$, the real value $0$ maps to $q = z$ and back to $s (z - z) = 0$ exactly. The error bound follows from rounding: $x/s$ moves by at most $\tfrac12$, so $\hat{x}$ moves by at most $s/2$.
