## The idea

Split the hidden units into $n$ groups, $H_0, \dots, H_{n-1}$, and look at what each
layer needs.

**First layer: independent per hidden unit.** Hidden unit $j$ is
$m_j = \operatorname{silu}(w_j \cdot x + b_j)$, where $w_j$ is row $j$ of
$W_{\text{up}}$. It depends on $x$, its own row, and its own bias, and on nothing
else. A device that holds the rows of $H_i$ can compute those hidden units
completely, activation included, with no communication. This is why the first
layer is cut by *outputs*: a cut by inputs would leave every device with a partial
pre-activation, and $\operatorname{silu}(a + b) \ne \operatorname{silu}(a) +
\operatorname{silu}(b)$, so the partial sums would have to be combined *before*
the activation.

**Second layer: a sum over hidden units.**

$$
W_{\text{down}}\, m = \sum_{j} m_j\, c_j = \sum_{i=0}^{n-1} \; \sum_{j \in H_i} m_j\, c_j ,
$$

where $c_j$ is column $j$ of $W_{\text{down}}$. The inner sum uses only the hidden
units of $H_i$ and the matching columns: exactly what device $i$ has if the second
layer is cut by *inputs* along the same boundaries. Each device produces a partial
output of full output size, and the outer sum is one all-reduce.

So the two cuts are the same cut, seen from both sides of the hidden layer, and
the only tensor that ever crosses between devices has the size of the output. The
bias $b_{\text{down}}$ is not part of any term of the sum; it is added once, after
the reduce.

**Why not gather?** Cutting both layers by outputs also "works": each device
computes its hidden slice, an all-gather rebuilds the full hidden vector on every
device, and each device then computes some of the output rows. That moves
$(n - 1) \cdot h$ floats per token instead of $n \cdot d_{out}$. In a transformer
MLP $h$ is about four times $d_{out}$, so the synchronization point sits where
the tensor is widest, and it is there for no reason.

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

The JAX and MLX versions differ only in how `silu` is spelled (`jax.nn.silu`,
`pre * mx.sigmoid(pre)`). `i * h // n` gives
balanced blocks without a special case for the remainder: consecutive boundaries
differ by $\lfloor h/n \rfloor$ or $\lceil h/n \rceil$.

## Common mistakes

- **Adding the output bias on every device.** Each partial output then carries
  $b_{\text{down}}$, and the all-reduce sums $n$ copies. The result is off by
  $(n - 1)\, b_{\text{down}}$: small if the bias is small, and present in every
  token.
- **Gathering between the layers.** Correct output, with an all-gather of the
  hidden activations on top of (or instead of) the all-reduce. The cost check
  rejects it; in production it shows up as a model that scales poorly with the
  number of GPUs.
- **Computing on one device.** Fetching every shard, concatenating, and running
  the unsplit layer needs no communication because nothing was parallel. Handing
  the finished result to `all_reduce` next to zeros does not change that; the
  tests compare what each device contributes with the partial output of its own
  shard.
- **Shipping the partial outputs to one device with `send` and adding them
  there.** It is a reduction, done as $n - 1$ point-to-point messages, and it
  leaves the result on one device only; the next layer needs it on all of them.
- **Slicing `w_down` by rows.** `w_down[lo:hi]` is the cut of an output-split
  layer: each device then holds some output rows and would need *all* hidden
  units to compute them. The second layer is cut along its inputs, `w_down[:, lo:hi]`.
- **Shards that overlap or leave a gap.** `hi = lo + h // n + 1`, or
  `range(0, h, h // n)` with a remainder, counts a hidden unit twice or drops
  one. The blocks must partition $[0, h)$.
- **Putting the whole remainder on the last device** (`h // n` units each, the
  rest at the end). With $h = 10$ and $n = 4$ that is `2, 2, 2, 4`; the last
  device needs twice the memory of the others and does not fit its budget.
- **One all-reduce per row of the batch.** The same bytes in many small messages.
  Collectives have a fixed latency each; the batch is reduced as one tensor.

## Complexity

Each device does $1/n$ of the layer's arithmetic: $O(h_i (d + d_{out}))$ per input
vector, and holds $1/n$ of the weights. Communication per input vector is
$n \cdot d_{out}$ floats in one all-reduce, independent of $h$.

## In the wild

- *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
  Parallelism* (arXiv 1909.08053), section 3, introduces this scheme. With layers
  written as $Y = \operatorname{GeLU}(XA)$ and $Z = YB$, it splits $A$ along its
  columns and $B$ along its rows, notes that splitting $A$ the other way "will
  require a synchronization point before the GeLU function", and concludes that
  the MLP block "requires only a single all-reduce operation in the forward
  pass". The same idea is applied to attention (heads are split across devices,
  the output projection is the reduced layer), for two all-reduces per
  transformer layer in the forward pass. The paper also keeps duplicate copies of
  small parameters such as layer normalization on every device instead of
  communicating them, which is what `down_bias` does here.
- vLLM serves a model with this scheme when started with
  `--tensor-parallel-size N`, and can combine it with pipeline parallelism across
  nodes (`--pipeline-parallel-size`, the subject of the next problem); its
  documentation on parallelism and scaling recommends a fast interconnect for
  tensor parallelism across nodes.
- `expert-parallel-dispatch` (Frontier track) places whole experts of a
  mixture-of-experts layer on different devices instead of slices of one matrix.
