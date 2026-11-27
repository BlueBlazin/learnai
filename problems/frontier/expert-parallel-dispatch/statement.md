Implement `ep_forward`, an expert-parallel mixture-of-experts (MoE) layer where experts are distributed across multiple devices. Each token starts on a home device, is routed to its top $k$ experts, and must have its weighted expert outputs returned to its home device. Batch transfers into one dispatch message and one combine message per active device pair, send each token at most once per remote device, and call each active expert once on all rows routed to it.

## Specification

```python
def ep_forward(x, home, w_router, experts, placement, devs, k): ...
```

Let $N$ be the number of tokens, $D$ the hidden dimension, $E$ the number of experts, and `n_devices` the number of devices (`devs.n_devices`).

- `x`: input token representations, one row per token.
- `home`: list of device indices where `home[t]` is the home device of token $t$. Token $t$'s input row starts on `home[t]`, and its final output row must end up there.
- `w_router`: router weight matrix in `(out, in)` layout. Every device holds a copy, so routing is computed on each token's home device.
- `experts`: a [`CountingExperts`](../../../python/src/learnai/kit/moe.py) object, as in `moe-sparse-dispatch`. Calling `experts(e, rows)` applies expert `e` row-wise to a tensor of shape `(n, D)`, returns shape `(n, D)`, increments `expert_calls` by 1, and adds `n` to `expert_rows`.
- `placement`: list of device indices where expert $e$ lives exclusively on device `placement[e]`.
- `devs`: a [`SimDevices`](../../../python/src/learnai/kit/simdevices.py) simulator with `devs.n_devices` devices. Calling `devs.send(src, dst, tensor)` sends one message from device `src` to device `dst`, returns `tensor` for the receiver to use, increments `sends` by 1, and adds the tensor's byte size to `comm_bytes`. Sending from a device to itself raises an error.
- `k`: number of experts selected per token ($1 \le k \le E$).
- Returns `y`: the MoE layer output, one row per token.

**Routing and output.** Routing follows `moe-sparse-dispatch`. For each token $t$ with input row $x_t$, compute router logits $r = W_{\text{router}} x_t$. Let $S_t$ be the set of the $k$ experts with the largest logits, breaking ties by smaller expert index first. Normalize the softmax probabilities over the selected experts to get weights $w_{t,e} = \operatorname{softmax}(r)_e / \sum_{j \in S_t} \operatorname{softmax}(r)_j$. Letting $f_e(x_t)$ denote the output of expert $e$ on $x_t$, the layer output is:

$$
y_t = \sum_{e \in S_t} w_{t,e}\, f_e(x_t).
$$

**Where computation happens.** Write $h_t = \text{home}[t]$ and $P(e) = \text{placement}[e]$. A device may only compute on rows present on it, namely its own home tokens and rows it received via `devs.send`. For each ordered pair of devices $(s, d)$, define the set of tokens on $s$ that need at least one expert on $d$:

$$
U_{s \to d} = \{\, t : h_t = s \text{ and } P(e) = d \text{ for some } e \in S_t \,\}.
$$

For each token $t$ and device $d$, define the partial weighted output produced by device $d$:

$$
p_{t,d} = \sum_{e \in S_t,\; P(e) = d} w_{t,e}\, f_e(x_t).
$$

Because $y_t = \sum_d p_{t,d}$, the layer executes in three phases:

1. **Dispatch.** For every ordered pair of distinct devices $(s, d)$ with non-empty $U_{s \to d}$, send **one** message `devs.send(s, d, rows)`. Here `rows` has shape $(|U_{s \to d}|, D)$ and contains $x_t$ for $t \in U_{s \to d}$ in increasing order of $t$. Include each token once even if multiple selected experts live on $d$. If $U_{s \to d}$ is empty, send no message.
2. **Compute.** On device $d$, call each local expert selected by at least one token **exactly once**, passing the rows of all tokens routed to it (both local and received, in any row order). Do not call experts that no token selected. Then weight each expert output and sum over token $t$'s experts on $d$ to form $p_{t,d}$ for every token present on $d$.
3. **Combine.** For every pair $(s, d)$ that had a dispatch message, send **one** return message `devs.send(d, s, results)` of shape $(|U_{s \to d}|, D)$ containing $p_{t,d}$ for the same tokens in the same order. Each home device sums the received partial outputs with its own local part $p_{t,h_t}$.

When $s = d$, send nothing because the device already holds its own rows and keeps its own partial results. Routing metadata (which experts and weights belong to each row) is treated as traveling with the rows and is not metered. Only the float32 row tensors are counted.

**The ledger.** Let $M$ be the number of ordered pairs $(s, d)$ with $s \ne d$ and non-empty $U_{s \to d}$. At the end of `ep_forward`, the counters must match:

| counter | value | why |
|---|---|---|
| `devs.counters.sends` | exactly $2M$ | one dispatch message out and one combine message back per active pair |
| `devs.counters.comm_bytes` | exactly $2 \cdot 4D \cdot \sum_{s \ne d} \lvert U_{s \to d} \rvert$ | each routed row sent once per remote device, and one result row returned |
| `experts.counters.expert_calls` | number of experts with at least one token | one call per active expert |
| `experts.counters.expert_rows` | exactly $k \cdot N$ | no expert processes a row not routed to it |

Do not use collectives (`all_reduce`, `all_gather`) or `devs.put`.

You may use your framework's `softmax` and sorting functions, and plain Python or NumPy for index, group, and device bookkeeping.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(N, D)` | float32 |
| `home` | list of $N$ ints in `[0, n_devices)` | Python `int` |
| `w_router` | `(E, D)` | float32 |
| `placement` | list of $E$ ints in `[0, n_devices)` | Python `int` |
| dispatch message from $s$ to $d$ | $(\lvert U_{s \to d} \rvert, D)$ | float32 |
| result message from $d$ to $s$ | $(\lvert U_{s \to d} \rvert, D)$ | float32 |
| result `y` | `(N, D)` | float32 |

## Example

Consider the layer from `moe-sparse-dispatch` ($D = 2$, $E = 5$, $k = 2$, where the router maps a token $[a, b]$ to logits $[a, b, -a, -b, 0]$) placed across three devices:

```python
x = [[3.0, 1.0], [-1.0, 2.0], [2.0, -3.0], [1.0, 3.0], [-3.0, 1.0]]
home = [0, 0, 1, 1, 1]  # tokens 0, 1 on device 0; tokens 2, 3, 4 on device 1
placement = [0, 0, 1, 1, 2]  # experts 0, 1 on device 0; 2, 3 on device 1; 4 on device 2
```

| token | home | experts (weights) | their devices | crosses to |
|---|---|---|---|---|
| 0 | 0 | 0 (0.8808), 1 (0.1192) | 0, 0 | nowhere |
| 1 | 0 | 1 (0.7311), 2 (0.2689) | 0, 1 | device 1 |
| 2 | 1 | 3 (0.7311), 0 (0.2689) | 1, 0 | device 0 |
| 3 | 1 | 1 (0.8808), 0 (0.1192) | 0, 0 | device 0, **once** |
| 4 | 1 | 2 (0.8808), 1 (0.1192) | 1, 0 | device 0 |

This gives $U_{0 \to 1} = \{1\}$ and $U_{1 \to 0} = \{2, 3, 4\}$. Device 2 is completely idle because no token selects expert 4.

| phase | message | payload |
|---|---|---|
| dispatch | `send(0, 1, x[[1]])` | 1 row, 8 bytes |
| dispatch | `send(1, 0, x[[2, 3, 4]])` | 3 rows, 24 bytes |
| combine | `send(1, 0, ...)` | $p_{1,1} = 0.2689\, f_2(x_1)$ (1 row, 8 bytes) |
| combine | `send(0, 1, ...)` | $p_{2,0} = 0.2689\, f_0(x_2)$, $p_{3,0} = 0.8808\, f_1(x_3) + 0.1192\, f_0(x_3)$, $p_{4,0} = 0.1192\, f_1(x_4)$ (3 rows, 24 bytes) |

In total, there are 4 messages and 64 bytes transferred. On device 0, expert 0 runs once on tokens 0, 2, 3 and expert 1 runs once on tokens 0, 1, 3, 4, pooling local and received rows in each call. On device 1, expert 2 runs on tokens 1 and 4 and expert 3 runs on token 2. That totals 4 expert calls and $3 + 4 + 2 + 1 = 10 = k \cdot N$ rows.

Notice token 3. Both of its selected experts live on device 0, so its row crosses once and a single combined result row returns. Sending one row per `(token, expert)` assignment instead would cost 80 bytes rather than 64.

## Constraints

- $1 \le N \le 64$, $2 \le D \le 32$, $1 \le E \le 8$, $1 \le k \le E$, and $1 \le$ `n_devices` $\le 8$. A device may hold no experts, no tokens, or neither.
- The returned `y` is compared against the dense MoE evaluation with relative tolerance `1e-4` and absolute tolerance `1e-5`.
- All four ledger counters are checked in every test case after checking `y`.
- **Message contents are checked.** Tests record every tensor passed to `devs.send`. Each dispatch message must contain the exact rows of $U_{s \to d}$ in increasing token order, and each combine message must match $p_{t,d}$ for those tokens within the tolerances above. Computing the layer on one device and sending dummy tensors of the right shape will fail.
- In every test, a token's $k$-th and $(k+1)$-th largest router logits are either exactly equal (small integers) or differ by more than $10^{-3}$.
- Python loops over devices and experts are expected. No loop over tokens is needed.

## Why this matters

Large MoE models do not fit all their experts on a single GPU, and each token only activates a small subset of them anyway. Expert parallelism shards experts across devices and moves token activations instead of weights, making the all-to-all dispatch and combine exchange the primary bottleneck for throughput.

Every counter in this problem reflects a real distributed-systems pitfall. Sending one message per token replaces a few large transfers with thousands of latency-bound small messages. Broadcasting all tokens to every device wastes bandwidth by a factor of `n_devices`. Sending one row per `(token, expert)` assignment duplicates traffic whenever co-selected experts share a device. Calling an expert separately per sender shrinks GPU batch sizes, and returning unweighted outputs sends more rows back than were dispatched.
