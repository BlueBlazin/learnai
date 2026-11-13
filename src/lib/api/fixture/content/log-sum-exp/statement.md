Many quantities in inference are sums of probabilities that are stored as logarithms: the normalizer of a softmax, the score of a beam, the marginal over alignments. Adding them requires

$$
\operatorname{LSE}(x) = \log \sum_{i=1}^{n} e^{x_i},
$$

and the literal formula fails in both directions. For large $x_i$ the exponential overflows; for very negative $x_i$ every term underflows to $0$ and the logarithm returns $-\infty$.

Implement `logsumexp(x, dim)` that is accurate across the whole float32 range.

## Interface

```python
def logsumexp(x: Tensor, dim: int = -1) -> Tensor:
    """Return log(sum(exp(x))) along `dim`; `dim` is removed from the shape."""
```

### Example 1

```python
logsumexp(torch.tensor([-1000.0, -1000.0]))
# tensor(-999.3069)        # -1000 + log(2)
```

## Properties the judge checks

Besides comparing against the oracle, the property judge draws seeded random inputs and verifies three invariants:

| Property | Statement |
|---|---|
| Shift | $\operatorname{LSE}(x + c) = \operatorname{LSE}(x) + c$ |
| Bounds | $\max_i x_i \le \operatorname{LSE}(x) \le \max_i x_i + \log n$ |
| Masking | appending $-\infty$ entries leaves the result unchanged |

## Constraints

- `x` is float32. Entries may be $-\infty$, never $+\infty$ or `nan`.
- A slice that is all $-\infty$ must return $-\infty$, not `nan`.
