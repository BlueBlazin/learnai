Temperature rescales logits before the softmax; top-$k$ keeps only the $k$ most likely tokens. Together they are the two knobs on almost every sampling API.

With temperature $\tau > 0$,

$$
P_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}} .
$$

As $\tau \to 0$ the distribution approaches a one-hot on the arg max; as $\tau \to \infty$ it approaches uniform.

Implement `top_k_probs(logits, k, temperature)`, returning the **probabilities** after both transforms. Sampling from them is a separate problem.

## Interface

```python
def top_k_probs(logits: Tensor, k: int, temperature: float) -> Tensor:
    """logits: (B, V). Returns (B, V) probabilities; zero outside the top k."""
```

### Example 1

```python
top_k_probs(torch.tensor([[2.0, 1.0, 0.0, -1.0]]), k=2, temperature=1.0)
# tensor([[0.7311, 0.2689, 0.0000, 0.0000]])
```

### Example 2

Halving the temperature doubles every logit gap.

```python
top_k_probs(torch.tensor([[2.0, 1.0, 0.0, -1.0]]), k=2, temperature=0.5)
# tensor([[0.8808, 0.1192, 0.0000, 0.0000]])
```

## Effect of temperature

The largest probability for logits $[2, 1, 0, -1]$ with $k = 4$:

| $\tau$ | Scaled logits | $P_{\max}$ |
|---:|---|---:|
| $0.5$ | $[4, 2, 0, -2]$ | $0.865$ |
| $1$ | $[2, 1, 0, -1]$ | $0.644$ |
| $2$ | $[1, 0.5, 0, -0.5]$ | $0.455$ |

## Constraints

- $1 \le k \le V \le 50{,}000$ and $0.05 \le \tau \le 5$.
- Ties at the $k$-th place: keep the lower token ids, so exactly $k$ entries are non-zero.
- Tolerance $10^{-6}$.
