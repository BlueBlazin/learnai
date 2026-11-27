## The idea

Partition the $h$ hidden units into $n$ disjoint groups $H_0, \dots, H_{n-1}$, and trace what each layer needs.

**First layer (independent per hidden unit).** Hidden unit $j$ equals $m_j = \operatorname{silu}(w_j \cdot x + b_j)$, where $w_j$ is row $j$ of $W_{\text{up}}$. It depends only on the full input $x$, its own weight row, and its own bias. A device holding the rows in $H_i$ can compute all of its hidden activations locally with no communication. Splitting the first layer by inputs instead would leave every device with partial pre-activations. Because $\operatorname{silu}(a + b) \ne \operatorname{silu}(a) + \operatorname{silu}(b)$, the devices would have to synchronize before applying the activation.

**Second layer (sum over hidden units).** Express the matrix-vector product as a sum over columns of $W_{\text{down}}$:

$$
W_{\text{down}}\, m = \sum_{j} m_j\, c_j = \sum_{i=0}^{n-1} \; \sum_{j \in H_i} m_j\, c_j ,
$$

where $c_j$ is column $j$ of $W_{\text{down}}$. Each inner sum uses only the hidden units in $H_i$ and the matching columns of $W_{\text{down}}$. When the second layer is split by inputs along the same boundaries, device $i$ already holds everything it needs to compute its partial output of shape `(..., d_out)`. One all-reduce then performs the outer sum across devices.

Both layers are therefore cut along the same hidden dimension, viewed from opposite sides of the activation. The only tensor that crosses between devices has the width of the output, $d_{out}$. Because $b_{\text{down}}$ does not belong to any individual term in the sum, it is added once after the all-reduce.

**Why not gather between layers?** Cutting both layers by outputs would also produce the right answer. Each device would compute its slice of hidden units, call an all-gather to assemble the full hidden vector on every device, and then compute a slice of the output rows. That transfers $(n - 1) \cdot h$ floats per token instead of $n \cdot d_{out}$. In a typical transformer MLP, $h$ is roughly four times $d_{out}$, so gathering puts communication at the widest tensor in the block for no benefit.

## Reference solution (PyTorch)

```python
def shard_weights(w_up, b_up, w_down, b_down, n):
    h = w_up.shape[0]
    bounds = [i * h // n for i in range(n + 1)]
    return [
        {
            "up": w_up[lo:hi],
            "up_bias": b_up[lo:hi],
            "down": w_down[:, lo:hi],
            "down_bias": b_down,
        }
        for lo, hi in pairwise(bounds)
    ]


def tp_mlp(x, devs):
    parts = []
    for i in range(devs.n_devices):
        hidden = F.silu(x @ devs.get(i, "up").T + devs.get(i, "up_bias"))
        parts.append(hidden @ devs.get(i, "down").T)
    return devs.all_reduce(parts) + devs.get(0, "down_bias")
```

The JAX and MLX implementations differ only in the SiLU call (`jax.nn.silu` in JAX, `pre * mx.sigmoid(pre)` in MLX). Using `i * h // n` produces balanced blocks without special-casing the remainder, since consecutive boundaries always differ by $\lfloor h / n \rfloor$ or $\lceil h / n \rceil$.

## Common mistakes

- **Adding the output bias on every device.** Each partial output then includes $b_{\text{down}}$, and the all-reduce sums $n$ copies of it. The final output is off by $(n - 1)\, b_{\text{down}}$ on every token.
- **Gathering between the layers.** An all-gather of the hidden activations produces the right values but violates the communication budget. In production, this causes poor scaling as you add GPUs.
- **Computing on one device.** Fetching every shard, concatenating them, and running the unsplit layer avoids real parallelism. Even if you pass the full result alongside zeros to `all_reduce`, the tests catch this by verifying each device's partial output individually.
- **Using `send` to gather partial outputs on one device.** Point-to-point sends require $n - 1$ messages and leave the summed output on only one device, whereas the next layer expects the output on all devices.
- **Slicing `w_down` by rows.** Slicing `w_down[lo:hi]` splits the second layer by outputs, which requires every device to hold all $h$ hidden units. The second layer must be sliced along its input axis with `w_down[:, lo:hi]`.
- **Overlapping or gapped shards.** Boundaries like `hi = lo + h // n + 1` or `range(0, h, h // n)` with a remainder either double-count or drop hidden units. The slices must form an exact partition of $[0, h)$.
- **Placing the entire remainder on the last device.** Assigning `h // n` units to the first $n - 1$ devices and the remainder to the last gives sizes `2, 2, 2, 4` for $h = 10, n = 4$. The last device then exceeds its memory budget.
- **Running one all-reduce per row of the batch.** Splitting a batch into separate reductions sends the same total bytes across many small collectives, paying collective launch latency on every row instead of once.

## Complexity

Each device performs $1 / n$ of the layer's arithmetic, taking $O(h_i (d + d_{out}))$ operations per input vector, and stores $1 / n$ of the weights. Communication per input vector is $n \cdot d_{out}$ floats in a single all-reduce, independent of the hidden width $h$.

## In the wild

- *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (arXiv 1909.08053), section 3, introduces this sharding pattern. Writing the layers as $Y = \operatorname{GeLU}(XA)$ and $Z = YB$, it splits $A$ along its columns and $B$ along its rows, notes that splitting $A$ the other way would require a synchronization point before GeLU, and shows that the MLP block needs only one all-reduce in the forward pass. Attention uses the same idea by splitting heads across devices and reducing after the output projection, giving two all-reduces per transformer block. Small parameters like layer-norm weights are replicated on every device, just like `down_bias` here.
- vLLM enables this scheme via `--tensor-parallel-size N` and can pair it with pipeline parallelism across nodes (`--pipeline-parallel-size`, covered in the next problem). Its scaling documentation recommends a fast interconnect such as NVLink for tensor parallelism.
- `expert-parallel-dispatch` (Frontier track) distributes whole experts of a mixture-of-experts layer across devices instead of slicing a single dense matrix.
