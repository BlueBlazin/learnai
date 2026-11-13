An autoregressive model may not look at the future. Position $i$ attends to position $j$ only when $j \le i$, and never to padding.

Build the boolean attention mask for a padded batch.

## Interface

```python
def causal_mask(lengths: Tensor, max_len: int) -> Tensor:
    """
    lengths: (B,) integer sequence lengths, each in [1, max_len].
    Returns a bool tensor of shape (B, 1, max_len, max_len),
    True where query position i may attend to key position j.
    """
```

The singleton axis is the head axis, so the result broadcasts against scores of shape $(B, H, T, T)$.

## Definition

For batch element $b$ with length $\ell_b$,

$$
M_{b,i,j} = \bigl[\, j \le i \,\bigr] \;\land\; \bigl[\, j < \ell_b \,\bigr].
$$

Padded **queries** ($i \ge \ell_b$) still get a non-empty row: they attend to the real tokens before them. This keeps every softmax row well defined; the loss ignores those positions anyway.

### Example 1

```python
causal_mask(torch.tensor([3, 2]), max_len=3)[:, 0].int()
# tensor([[[1, 0, 0],
#          [1, 1, 0],
#          [1, 1, 1]],
#
#         [[1, 0, 0],
#          [1, 1, 0],
#          [1, 1, 0]]])
```

## Properties the judge checks

| Property | Statement |
|---|---|
| Causality | $M_{b,i,j}$ is false whenever $j > i$ |
| Padding | $M_{b,i,j}$ is false whenever $j \ge \ell_b$ |
| Liveness | every row has at least one true entry |
| No loops | the implementation runs in under 5 ms for $B = 64$, $T = 1024$ |

## Constraints

- $1 \le B \le 64$, $1 \le$ `max_len` $\le 1024$.
- Return dtype is bool.
