The softmax function turns a vector of logits $z \in \mathbb{R}^V$ into a probability distribution:

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

Evaluated literally, this overflows as soon as any logit exceeds about $88.7$ in float32, because $e^{88.73} > 3.4 \times 10^{38}$. The result is `inf / inf`, which is `nan`, and one `nan` in a sampler poisons every token that follows.

Implement `softmax(logits, dim)` so that it is exact to float32 tolerance for any finite input, including logits in the thousands.

## Interface

```python
def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Return softmax along `dim`. Same shape and dtype as `logits`."""
```

Do not call the framework's built-in `softmax` or `log_softmax`; the judge replaces them with functions that raise.

## The identity you need

Softmax is invariant to adding a constant to every logit. For any $c$,

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c}\, e^{z_i}}{e^{-c} \sum_j e^{z_j}} = \operatorname{softmax}(z)_i .
$$

Choosing $c = \max_j z_j$ makes the largest exponent exactly $0$, so the numerator never overflows and the denominator is at least $1$.

### Example 1

```python
softmax(torch.tensor([1.0, 2.0, 3.0]))
# tensor([0.0900, 0.2447, 0.6652])
```

### Example 2

Large logits must not produce `nan`.

```python
softmax(torch.tensor([1000.0, 1001.0, 1002.0]))
# tensor([0.0900, 0.2447, 0.6652])
```

## Cases

| Case | Shape | Logit range | Tolerance |
|---|---|---|---:|
| small vector | `(3,)` | $[1, 3]$ | $10^{-6}$ |
| large magnitudes | `(3,)` | $[1000, 1002]$ | $10^{-6}$ |
| batch, last dim | `(8, 512)` | $[-30, 30]$ | $10^{-6}$ |
| batch, middle dim | `(4, 64, 16)` | $[-10^4, 10^4]$ | $10^{-6}$ |
| masked logits | `(2, 32)` | contains $-\infty$ | $10^{-6}$ |

> [!NOTE]
> A row that is entirely $-\infty$ has no valid distribution. The tests never include one, but a fully masked row is a real bug source in attention; you will meet it again in *Causal Mask*.

## Constraints

- Inputs are float32 and finite, except for $-\infty$ entries used as masks.
- Every slice along `dim` has at least one finite entry.
- $1 \le V \le 65{,}536$.
