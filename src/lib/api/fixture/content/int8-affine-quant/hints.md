Start with $\alpha = \min(x_{\min}, 0)$ and $\beta = \max(x_{\max}, 0)$. Why does including zero in the range make it exactly representable?
---
Compute in float, round, add the zero point, clamp, and only then cast to int8. Casting before clamping wraps around.
