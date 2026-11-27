Implement weight sharding and the forward pass for a tensor-parallel two-layer MLP across simulated devices. Split the first layer along its output features and the second layer along its input features so that both cuts align and the entire MLP requires only one all-reduce at the end.

## Specification

```python
def shard_weights(w_up, b_up, w_down, b_down, n): ...  # -> list of n dicts
def tp_mlp(x, devs): ...
```

**The layer.** Given an input vector $x \in \mathbb{R}^d$, hidden width $h$, and output width $d_{out}$, the MLP computes

$$
y = W_{\text{down}}\, \operatorname{silu}(W_{\text{up}}\, x + b_{\text{up}}) + b_{\text{down}},
\qquad \operatorname{silu}(z) = \frac{z}{1 + e^{-z}} .
$$

Weight matrices use `(out, in)` layout, so `w_up` has shape `(h, d)` and `w_down` has shape `(d_out, h)`. The hidden width $h$ is the large intermediate dimension. The input `x` has shape `(..., d)`, and the layer applies independently to every vector along its last axis.

**The split.** Divide the $h$ hidden units across $n$ devices, indexed $i = 0, \dots, n - 1$. Device $i$ owns the contiguous index range $[lo_i, hi_i)$, where

$$
lo_i = \left\lfloor \frac{i\,h}{n} \right\rfloor, \qquad hi_i = \left\lfloor \frac{(i+1)\,h}{n} \right\rfloor .
$$

Let $h_i = hi_i - lo_i$ be the number of hidden units on device $i$. These ranges partition $[0, h)$ into contiguous blocks whose sizes differ by at most 1. `shard_weights` returns a list of `n` dicts in device order, each with these exact keys:

| key | value | shape |
|---|---|---|
| `"up"` | rows $lo_i$ to $hi_i$ of `w_up` | `(h_i, d)` |
| `"up_bias"` | entries $lo_i$ to $hi_i$ of `b_up` | `(h_i,)` |
| `"down"` | **columns** $lo_i$ to $hi_i$ of `w_down` | `(d_out, h_i)` |
| `"down_bias"` | all of `b_down` (replicated on every device) | `(d_out,)` |

Every hidden unit belongs to exactly one device, and only `b_down` is stored more than once.

**The devices.** `devs` is a kit `SimDevices` instance holding each device's shard. `devs.n_devices` gives $n$, and `devs.get(i, key)` returns device $i$'s tensor for `"up"`, `"up_bias"`, `"down"`, or `"down_bias"`. Every device already has access to the input `x`. All communication between devices is explicit and metered:

- `devs.all_reduce(parts)` takes a list of `n` tensors of identical shape (one per device) and returns their elementwise sum in device order. It counts 1 all-reduce and `n * nbytes(part)` bytes.
- `devs.all_gather(parts, axis=-1)` concatenates the tensors along `axis` and counts `(n - 1) * nbytes(result)` bytes.
- `devs.send(src, dst, tensor)` sends a tensor from `src` to `dst` and counts 1 message of `nbytes(tensor)` bytes.

**`tp_mlp(x, devs)`** computes $y$ in three steps:

1. Each device $i$ computes its **partial output** from its own shard alone:

   $$
   p_i = \operatorname{silu}\!\big(x\, W_{\text{up},i}^{\top} + b_{\text{up},i}\big)\, W_{\text{down},i}^{\top} ,
   \qquad \text{shape } (\ldots, d_{out}) .
   $$

2. Sum the partial outputs across all devices with one all-reduce: `total = devs.all_reduce([p_0, ..., p_{n-1}])`.
3. Add the output bias once after the reduction: $y = \text{total} + b_{\text{down}}$.

The tests inspect the list passed to `all_reduce`, so device $i$'s entry must equal $p_i$. Make exactly 1 `all_reduce` call and no `all_gather` or `send` calls. Do not store any additional tensors on `devs`, because each device's memory budget is full.

> [!NOTE]
> The Megatron-LM paper writes a linear layer as $Y = XA$ with `(in, out)` weights, so it calls the first split "column-parallel" and the second "row-parallel". In the `(out, in)` layout used here, those same cuts slice the rows of `w_up` and the columns of `w_down`. In both conventions, both layers are cut along the **hidden** dimension.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to three dimensions | float32 |
| `w_up` | `(h, d)` | float32 |
| `b_up` | `(h,)` | float32 |
| `w_down` | `(d_out, h)` | float32 |
| `b_down` | `(d_out,)` | float32 |
| result of `tp_mlp` | `(..., d_out)` | float32 |

## Example

Let $d = 2$, $h = 4$, $d_{out} = 2$, and $n = 2$:

```python
w_up = [[1, 0], [0, 1], [1, 1], [1, -1]]  # (4, 2)
b_up = [0, -1, 0, 3]
w_down = [[1, 0, 1, 0], [0, 2, 0, -1]]  # (2, 4)
b_down = [10, 20]

shard_weights(w_up, b_up, w_down, b_down, 2)
# device 0: up [[1, 0], [0, 1]],  up_bias [0, -1], down [[1, 0], [0, 2]],  down_bias [10, 20]
# device 1: up [[1, 1], [1, -1]], up_bias [0, 3],  down [[1, 0], [0, -1]], down_bias [10, 20]
```

![The example on two devices. Each device turns its hidden units into a partial output; one all-reduce sums them, and $b_\text{down}$ is added once.](assets/tp-mlp.svg)

For `x = [1, 2]`:

- Device 0 computes pre-activations `[1, 1]`, hidden activations `[0.731059, 0.731059]`, and partial output `[0.731059, 1.462117]`.
- Device 1 computes pre-activations `[3, 2]`, hidden activations `[2.857723, 1.761594]`, and partial output `[2.857723, -1.761594]`.
- The all-reduce sums the two partial outputs to `[3.588781, -0.299477]`. Adding `b_down` once produces `y = [13.588781, 19.700523]`, which matches the unsplit layer.

The all-reduce transfers $2 \times 2$ floats (16 bytes), one output vector of length 2 per device. Adding `b_down` on both devices before the reduction would count the bias twice. Gathering hidden activations between the two layers instead would communicate tensors of width $h$. In a transformer MLP where $h$ is typically four times $d_{out}$, that intermediate gather moves four times as much data even though each device can multiply its hidden slice by its columns of $W_{\text{down}}$ locally.

## Constraints

- $2 \le n \le h$, $1 \le d, d_{out} \le 64$, and $h \le 256$. `x` has at most 4,096 elements. The widths are independent, so $d_{out}$ may equal $h$.
- `shard_weights` is checked for exact equality. `tp_mlp` and the partial outputs passed to `all_reduce` are checked against the unsplit reference with `rtol=1e-4` and `atol=1e-5`.
- **Cost.** `devs.counters` must record `all_reduces == 1`, `all_gathers == 0`, `sends == 0`, and `comm_bytes == n * nbytes(y)`.
- **Memory.** One test sets each device's memory budget to the exact size required by the sharding rule above, so oversized shards will fail.
- Do not write a Python loop over rows of `x` or over hidden units. Looping over the `n` devices is expected.

## Why this matters

Tensor parallelism lets a single large model run across multiple GPUs when one device cannot fit the weights or meet latency targets. In serving engines like vLLM, this is configured with `--tensor-parallel-size`. Because every token crosses the interconnect at every transformer block, the number of all-reduces and the size of the reduced tensor determine how well throughput scales across GPUs.

Implementation bugs here are often silent. Adding `b_down` on every rank or overlapping shards by a single column shifts outputs slightly on every token without raising an error. Gathering hidden states between the two layers produces exact numerical outputs while silently multiplying communication volume by $h / d_{out}$.
