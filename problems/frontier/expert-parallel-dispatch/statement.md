Run a mixture-of-experts layer whose experts live on different devices. Every token
lives on a home device, is routed to $k$ experts, and has to reach the devices that
hold them; the weighted results have to come back. Do it with one message of rows
per pair of devices and one message of results back, each token crossing once per
remote device however many of its experts live there, and each expert called once
on everything that arrived for it.

## Specification

```python
def ep_forward(x, home, w_router, experts, placement, devs, k): ...
```

- `x`: float32 tensor of shape `(N, D)`, one row per token.
- `home`: a list of $N$ Python ints. `home[t]` is the device on which token $t$
  lives: its row is there before the layer, and its output is needed there after.
- `w_router`: float32 `(E, D)`, in `(out, in)` layout. Every device has a copy, so
  routing is computed at home.
- `experts`: a [`CountingExperts`](../../../python/src/learnai/kit/moe.py), as
  in `moe-sparse-dispatch`: `experts(e, rows)` runs expert `e` on rows `(n, D)`,
  returns `(n, D)` row for row, and counts one `expert_calls` and `n`
  `expert_rows`.
- `placement`: a list of $E$ Python ints. Expert $e$ lives on device
  `placement[e]`, and only there.
- `devs`: a [`SimDevices`](../../../python/src/learnai/kit/simdevices.py) with
  `devs.n_devices` devices. `devs.send(src, dst, tensor)` is one message: it
  returns the tensor for the receiving side to use and adds one to `sends` and the
  tensor's size in bytes to `comm_bytes`. Sending from a device to itself is an
  error.
- `k`: a Python int, $1 \le k \le E$.
- Returns `y`: float32 `(N, D)`.

**Routing and output** are those of `moe-sparse-dispatch`. For token $t$: router
logits $r = W_{\text{router}} x_t$; $S_t$ is the set of the $k$ experts with the
largest logits, equal logits ranked by index, lowest first; weights
$w_{t,e} = \operatorname{softmax}(r)_e / \sum_{j \in S_t} \operatorname{softmax}(r)_j$;
and

$$
y_t = \sum_{e \in S_t} w_{t,e}\, f_e(x_t).
$$

**Where things happen.** Write $h_t$ = `home[t]` and $P(e)$ = `placement[e]`. A
device computes only with rows that are on it: its own tokens and the rows it was
sent. For every device $d$, let

$$
U_{s \to d} = \{\, t : h_t = s \text{ and } P(e) = d \text{ for some } e \in S_t \,\}
$$

be the tokens of device $s$ that need an expert on $d$, and

$$
p_{t,d} = \sum_{e \in S_t,\; P(e) = d} w_{t,e}\, f_e(x_t)
$$

the part of token $t$'s output that device $d$ produces. Then $y_t = \sum_d
p_{t,d}$, and the layer runs in three phases:

1. **Dispatch.** For every ordered pair of different devices $(s, d)$ with
   $U_{s \to d}$ not empty: **one** message `devs.send(s, d, rows)`, where `rows`
   has shape $(|U_{s \to d}|, D)$ and holds $x_t$ for $t \in U_{s \to d}$ in
   increasing order of $t$. A token is in it once, even if several of its experts
   live on $d$. An empty $U_{s \to d}$ means no message.
2. **Compute.** On device $d$, each expert that lives there and was selected by at
   least one token is called **exactly once**, on the rows of all tokens routed to
   it, whether they are at home on $d$ or arrived from any other device (in any row
   order). An expert that no token selected is not called. Device $d$ then forms
   $p_{t,d}$ for every token whose row it has: weighted, and summed over that
   token's experts on $d$.
3. **Combine.** For every pair with a dispatch message: **one** message
   `devs.send(d, s, results)` of shape $(|U_{s \to d}|, D)$ holding $p_{t,d}$ for
   the same tokens in the same order. The home device adds what it receives to its
   own part $p_{t,h_t}$.

For $s = d$ nothing is sent: a device uses its own rows and keeps its own results.
Which experts a row is for, and with what weights, is known wherever the row is:
that bookkeeping travels with the rows and is not metered. Only the float rows
are.

**The ledger.** With $M$ the number of ordered pairs $(s, d)$, $s \ne d$, that have
a non-empty $U_{s \to d}$:

| counter | value | why |
|---|---|---|
| `devs.counters.sends` | exactly $2M$ | rows out and results back, once per pair |
| `devs.counters.comm_bytes` | exactly $2 \cdot 4D \cdot \sum_{s \ne d} \lvert U_{s \to d} \rvert$ | each routed row once per remote device, and one result row for it |
| `experts.counters.expert_calls` | the number of experts with a token | one call per active expert |
| `experts.counters.expert_rows` | exactly $k \cdot N$ | no expert sees a row that was not routed to it |

No collective (`all_reduce`, `all_gather`) and no `devs.put` is used.

You may use your framework's `softmax` and sorting functions, and plain Python or
NumPy for the bookkeeping of indices, groups, and devices.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(N, D)` | float32 |
| `home` | list of $N$ ints in `[0, n_devices)` | Python `int` |
| `w_router` | `(E, D)` | float32 |
| `placement` | list of $E$ ints in `[0, n_devices)` | Python `int` |
| a dispatch message from $s$ to $d$ | $(\lvert U_{s \to d} \rvert, D)$ | float32 |
| the result message from $d$ to $s$ | $(\lvert U_{s \to d} \rvert, D)$ | float32 |
| result | `(N, D)` | float32 |

## Example

The layer of the `moe-sparse-dispatch` example ($D = 2$, five experts, $k = 2$, a
router that maps a token $[a, b]$ to the logits $[a, b, -a, -b, 0]$) on three
devices:

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

So $U_{0 \to 1} = \{1\}$ and $U_{1 \to 0} = \{2, 3, 4\}$, and nothing involves
device 2: expert 4 was selected by no token.

| phase | message | payload |
|---|---|---|
| dispatch | `send(0, 1, x[[1]])` | 1 row, 8 bytes |
| dispatch | `send(1, 0, x[[2, 3, 4]])` | 3 rows, 24 bytes |
| combine | `send(1, 0, ...)` | $p_{1,1} = 0.2689\, f_2(x_1)$: 1 row, 8 bytes |
| combine | `send(0, 1, ...)` | $p_{2,0} = 0.2689\, f_0(x_2)$, $p_{3,0} = 0.8808\, f_1(x_3) + 0.1192\, f_0(x_3)$, $p_{4,0} = 0.1192\, f_1(x_4)$: 3 rows, 24 bytes |

Four messages and 64 bytes. On device 0, expert 0 is called once with the rows of
tokens 0, 2, 3 and expert 1 once with those of tokens 0, 1, 3, 4: local and received
rows in the same call. On device 1, expert 2 gets tokens 1 and 4 and expert 3 gets
token 2. Four calls, $3 + 4 + 2 + 1 = 10 = k \cdot N$ rows.

Token 3 is the one to look at. Both of its experts live on device 0, so its row
crosses once and one result row comes back, already weighted and summed. Sending a
row per (token, expert) would cost 80 bytes instead of 64.

## Constraints

- $1 \le N \le 64$, $2 \le D \le 32$, $1 \le E \le 8$, $1 \le k \le E$,
  $1 \le$ `n_devices` $\le 8$. A device may hold no expert, no token, or neither.
- The result is compared with the dense evaluation of the layer to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- The four counters of the ledger are checked in every case, after the result.
- **What the messages carry is checked too.** The tests record the tensor of every
  `devs.send`. For each pair, the dispatch message must be exactly the rows of
  $U_{s \to d}$ in increasing token order, and the result message must equal the
  $p_{t,d}$ of those tokens within the tolerance above. A solution that evaluates
  the layer in one place and sends tensors of the right sizes fails.
- In every test, the $k$-th and $(k+1)$-th largest router logits of a token are
  either exactly equal (small integers) or differ by more than $10^{-3}$.
- Python loops over devices and experts are expected. No loop over tokens is
  needed.

## Why this matters

A large mixture-of-experts model does not fit its experts on one GPU, and it does
not need to: each token uses a few of them. Expert parallelism puts different
experts on different devices and moves the tokens instead of the weights, and the
all-to-all exchange it needs (rows out, results back) becomes the cost that
decides throughput. Every inefficiency in this problem is a real one. One message
per token turns a handful of large transfers into thousands of small ones, each
with its own latency. Broadcasting every token to every device multiplies the
traffic by the number of devices for nothing. Sending a row once per expert
instead of once per device doubles the bytes exactly when placement worked well
and put a token's experts together. Running an expert once per sender shrinks the
batches that make a GPU efficient. And returning unweighted outputs for the home
device to combine sends more rows back than went out.
