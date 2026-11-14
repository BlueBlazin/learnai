A decode loop that looks correct can still be slow for a reason no profiler line shows: a tensor created on the CPU inside the loop forces a host round trip on every step.

Write `causal_bias(length, device)`: return the additive attention bias of shape $(T, T)$ for a sequence of `length` tokens, built **on `device`**, with

$$
B_{ij} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

## What is judged

| case | where it runs | checks |
|---|---|---|
| values for $T = 4$ | CPU | the mask itself |
| `float32`, no gradient | CPU | dtype and `requires_grad` |
| result is on the CUDA device | NVIDIA GPU | placement |
| no host round trip | NVIDIA GPU | zero CPU-to-GPU copies |

The function takes `device` as an argument and must work for `torch.device("cpu")`: the CPU cases carry the logic and run everywhere. The two GPU cases run only on a machine with an NVIDIA GPU.

```python
def causal_bias(length, device):
    """Return a (length, length) float32 bias on `device`."""
```
