When a layer is too large for one device, its weight matrices are cut into slices,
one per device, and every device computes with its slice. Split a two-layer MLP
this way: the first layer by its **output** features, the second by its **input**
features, so that the two cuts line up and the whole MLP needs a single
communication step, one all-reduce of a tensor the size of the output. A cut along
the other axis gives the same numbers and pays for communication in the middle,
where the tensor is widest.

## Specification

```python
def shard_weights(w_up, b_up, w_down, b_down, n): ...  # -> list of n dicts
def tp_mlp(x, devs): ...
```

**The layer.** For an input vector $x \in \mathbb{R}^d$,

$$
y = W_{\text{down}}\, \operatorname{silu}(W_{\text{up}}\, x + b_{\text{up}}) + b_{\text{down}},
\qquad \operatorname{silu}(z) = \frac{z}{1 + e^{-z}} ,
$$

with weights in `(out, in)` layout: `w_up` is `(h, d)`, `w_down` is `(d_out, h)`.
The hidden width $h$ is the large dimension. `x` has shape `(..., d)` and the layer
is applied to every vector along its last axis.

**The split.** Device $i$ of $n$ owns the hidden units $[\,lo_i, hi_i)$ with

$$
lo_i = \left\lfloor \frac{i\,h}{n} \right\rfloor, \qquad hi_i = \left\lfloor \frac{(i+1)\,h}{n} \right\rfloor ,
$$

contiguous blocks that differ in size by at most one. `shard_weights` returns a list
of `n` dicts, one per device in device order, with exactly these keys:

| key | value | shape |
|---|---|---|
| `"up"` | rows $lo_i$ to $hi_i$ of `w_up` | `(h_i, d)` |
| `"up_bias"` | entries $lo_i$ to $hi_i$ of `b_up` | `(h_i,)` |
| `"down"` | **columns** $lo_i$ to $hi_i$ of `w_down` | `(d_out, h_i)` |
| `"down_bias"` | all of `b_down` (every device holds a copy) | `(d_out,)` |

with $h_i = hi_i - lo_i$. Nothing is stored twice except `b_down`, and every hidden
unit belongs to exactly one device.

**The devices.** `devs` is a kit `SimDevices` on which the test has stored each
shard: `devs.get(i, "up")`, `devs.get(i, "up_bias")`, `devs.get(i, "down")`, and
`devs.get(i, "down_bias")` return device $i$'s tensors, and `devs.n_devices` is $n$.
The input `x` is available on every device. Communication between devices is
explicit and metered:

- `devs.all_reduce(parts)` takes a list with one tensor per device, all of one
  shape, and returns their elementwise sum (in device order). It counts one
  all-reduce and `n * nbytes(part)` bytes.
- `devs.all_gather(parts, axis=-1)` returns the concatenation and counts
  `(n - 1) * nbytes(result)` bytes; `devs.send(src, dst, tensor)` counts one message
  of `nbytes(tensor)`.

**`tp_mlp(x, devs)`** computes $y$ like this:

1. Device $i$ computes its **partial output** from its own shard alone,

   $$
   p_i = \operatorname{silu}\!\big(x\, W_{\text{up},i}^{\top} + b_{\text{up},i}\big)\, W_{\text{down},i}^{\top} ,
   \qquad \text{shape } (\ldots, d_{out}) .
   $$

2. One all-reduce sums them: `total = devs.all_reduce([p_0, ..., p_{n-1}])`.
3. The bias is added once, after the reduce: $y = \text{total} + b_{\text{down}}$.

The tests read what was handed to the all-reduce: the tensor for device $i$ must be
$p_i$, the partial output of device $i$'s shard. The function makes exactly one
all-reduce and no other communication (no `all_gather`, no `send`), and stores
nothing on the devices (their memory budgets are full).

> [!NOTE]
> The Megatron-LM paper writes a layer as $Y = XA$, so its weight matrices are
> `(in, out)` and it calls the first split "along the columns" of $A$ and the
> second "along the rows" of $B$. In the `(out, in)` layout of checkpoints, used
> here, the same cuts are along the rows of `w_up` and the columns of `w_down`.
> Either way: both cuts are along the **hidden** dimension.

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

$d = 2$, $h = 4$, $d_{out} = 2$, $n = 2$:

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

- Device 0: pre-activations `[1, 1]`, hidden `[0.731059, 0.731059]`, partial output
  `[0.731059, 1.462117]`.
- Device 1: pre-activations `[3, 2]`, hidden `[2.857723, 1.761594]`, partial output
  `[2.857723, -1.761594]`.
- All-reduce: `[3.588781, -0.299477]`. Plus the bias: `y = [13.588781, 19.700523]`,
  the output of the unsplit layer.

The all-reduce moved $2 \times 2$ floats (16 bytes): two devices, an output of two
floats each. Adding `b_down` on both devices before the reduce would have added it
twice. Had the activations been gathered between the layers instead, each device
would have received the other's hidden units; with $h$ four times $d_{out}$, as in a
transformer MLP, that is the larger transfer by the same factor, and it buys
nothing, because $W_{\text{down}}$'s columns can be applied where the hidden units
already are.

## Constraints

- $2 \le n \le h$; $1 \le d, d_{out} \le 64$; $h \le 256$; `x` has at most 4,096
  elements. The widths are independent: $d_{out}$ may equal $h$.
- `shard_weights` is compared exactly (the shards are slices). `tp_mlp` is compared
  with the unsplit layer to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-5`, and so are the partial outputs handed to the all-reduce.
- **Cost.** `devs.counters`: `all_reduces` exactly 1, `all_gathers` and `sends`
  exactly 0, and `comm_bytes` exactly `n * nbytes(y)`.
- **Memory.** One case stores your shards on devices whose budgets are exactly the
  sizes the rule above gives; a shard that is too large does not fit.
- No Python loop over rows of `x` or over hidden units; a loop over the `n` devices
  is the point.

## Why this matters

This is tensor parallelism, the way a single large model is served from several
GPUs when one cannot hold it or cannot run it fast enough; in vLLM it is the
`--tensor-parallel-size` option. Every token of every request crosses the
interconnect in each block, so the amount moved per all-reduce, and the number of
them, decides how well a model scales across devices (vLLM's documentation asks
for a fast interconnect for this reason).
The implementation mistakes are quiet ones. A bias added on every rank, or shards
that overlap by a column, give outputs that are slightly wrong everywhere. A
gather between the two layers gives outputs that are exactly right at several
times the communication.
