Eight-bit quantization stores each weight as an integer and recovers an approximation with one multiply and one add. The affine scheme maps a real interval $[\alpha, \beta]$ onto the integers $[q_{\min}, q_{\max}] = [-128, 127]$:

$$
q = \operatorname{clamp}\!\left( \operatorname{round}\!\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max} \right),
\qquad
\hat{x} = s\,(q - z),
$$

with scale $s$ and integer zero point $z$ chosen so that $\alpha \mapsto q_{\min}$ and $\beta \mapsto q_{\max}$:

$$
s = \frac{\beta - \alpha}{q_{\max} - q_{\min}}, \qquad z = \operatorname{round}\!\left( q_{\min} - \frac{\alpha}{s} \right).
$$

Implement the pair `quantize` and `dequantize`.

## Interface

```python
def quantize(x: Tensor) -> tuple[Tensor, float, int]:
    """Return (q as int8, scale, zero_point) using the min and max of x as the range."""

def dequantize(q: Tensor, scale: float, zero_point: int) -> Tensor:
    """Return the float32 reconstruction."""
```

Two details matter. The range must be widened to include $0$, so that zero is exactly representable (padding and ReLU outputs depend on it). And rounding is round-half-to-even, which is what every framework's `round` does.

### Example 1

```python
q, s, z = quantize(torch.tensor([-1.0, 0.0, 0.5, 1.0]))
# q = tensor([-128, 0, 64, 127], dtype=torch.int8)   s = 0.00784   z = 0
dequantize(q, s, z)
# tensor([-1.0039, 0.0000, 0.5020, 0.9961])
```

## Properties the judge checks

| Property | Statement |
|---|---|
| Error bound | $\lvert x - \hat{x} \rvert \le s/2$ for every element inside the range |
| Exact zero | `dequantize(quantize(0))` is exactly $0.0$ |
| Idempotence | quantizing $\hat{x}$ with the same $s, z$ returns the same $q$ |
| Degenerate range | a constant tensor does not divide by zero |

## Constraints

- `x` is float32 with at least one element; values are finite.
- For a constant tensor use $s = 1$.
