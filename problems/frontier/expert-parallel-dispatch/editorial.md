## The idea

`moe-sparse-dispatch` had one list of assignments, (token, expert, weight), and one
question: which rows does each expert get? With experts on several devices every
assignment has two more attributes, the device the token is on and the device the
expert is on, and there are two groupings instead of one.

- **Messages are grouped by device pair.** What crosses from $s$ to $d$ is the set
  of *tokens* on $s$ with at least one expert on $d$. It is a set of tokens, not of
  assignments: a row is the same bytes whichever of the experts on $d$ will read
  it, so it crosses once.
- **Expert calls are grouped by expert.** Once the rows are on $d$, where they came
  from no longer matters. An expert runs once, on its local and received rows
  together, which is the reason to pool tokens from many devices at all: bigger
  batches per expert.

The way back mirrors the way out. Device $d$ could return every expert output and
let the home device weight and sum them, but the weights are known wherever the
row is, so $d$ can do the weighting and the sum over its own experts and return one
row per row it received: $p_{t,d} = \sum_{e \text{ on } d} w_{t,e} f_e(x_t)$. The
home device only adds. The exchange is then symmetric: the same pairs, the same row
counts, the same order in both directions, which is what lets a real system reuse
the dispatch layout as a handle for the combine.

Correctness is linearity: $y_t = \sum_{e \in S_t} w_{t,e} f_e(x_t)$ is a sum over
experts, and any partition of the experts (here: by device) gives partial sums
that add up to it.

## Reference solution (PyTorch)

```python
def ep_forward(x, home, w_router, experts, placement, devs, k):
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    routes = chosen.numpy()
    homes, placed = np.asarray(home), np.asarray(placement)
    target = placed[routes]  # (N, k): the device of each chosen expert

    y = torch.zeros_like(x)
    for dst in range(devs.n_devices):
        batches = []
        for src in range(devs.n_devices):
            tokens = np.flatnonzero((homes == src) & (target == dst).any(axis=1))
            if tokens.size == 0:
                continue
            rows = x[torch.from_numpy(tokens)]
            batches.append((src, tokens, rows if src == dst else devs.send(src, dst, rows)))
        if not batches:
            continue
        rows = torch.cat([batch[2] for batch in batches])
        tokens = np.concatenate([batch[1] for batch in batches])

        partial = torch.zeros_like(rows)
        for expert in np.flatnonzero(placed == dst).tolist():
            at, slot = np.nonzero(routes[tokens] == expert)
            if at.size == 0:
                continue
            weight = weights[torch.from_numpy(tokens[at]), torch.from_numpy(slot)]
            at = torch.from_numpy(at)
            partial = partial.index_add(0, at, weight[:, None] * experts(expert, rows[at]))

        start = 0
        for src, sent, _ in batches:
            part = partial[start : start + sent.size]
            start += sent.size
            if src != dst:
                part = devs.send(dst, src, part)
            y = y.index_add(0, torch.from_numpy(sent), part)
    return y
```

`(target == dst).any(axis=1)` is the deduplication: a token is selected once
however many of its `k` experts are on `dst`. `routes[tokens] == expert` is then
evaluated on the rows that are on `dst`, so `at` indexes those rows directly. The
JAX and MLX references are the same with `.at[...].add(...)`.

## Common mistakes

- **One message per token.** The bytes are the same and the number of messages is
  the number of rows. On a real interconnect every message has a fixed cost, and
  the kernels exist to batch.
- **Broadcasting.** Sending all tokens everywhere and letting each device pick is
  the simplest thing that works. It moves `n_devices - 1` copies of every row.
- **A row per (token, expert).** Flattening to assignments first, as in the sparse
  dispatch, and grouping those by device sends a token twice when two of its
  experts share a device, and returns two rows for it.
- **Overwriting instead of accumulating.** `partial[at] = w * out` is right as long
  as every row on a device is used by one expert there. A token with two experts
  on the device keeps only the second contribution. The bug is invisible with
  $k = 1$ or one expert per device.
- **Weights applied at home.** Returning raw expert outputs works, costs more
  bytes back than out, and leaves the combine with more bookkeeping than a sum.
- **Results in the wrong order.** The results of a pair must line up with the rows
  that were sent. Returning them in the order the experts produced them (grouped
  by expert) scatters them onto the wrong tokens.
- **No way back.** Adding the results into `y` on the expert's device forgets that
  `y_t` is needed at home; in a simulation the numbers are right and half of the
  communication is missing from the ledger.
- **A call per sender.** Processing each incoming message on its own gives an
  expert as many calls as it has senders, and smaller batches.
- **Every local expert on the whole inbox.** Running all experts of a device on
  all rows that arrived, with weight zero where a row was not routed, is the dense
  evaluation again, per device.
- **Empty messages.** A fixed all-to-all pattern sends zero-row messages between
  pairs that share no token.
- **Following only the first expert.** Sending a token to the device of its top
  expert and evaluating its other experts there assumes weights that are not on
  that device.
- **Not distributing at all.** Evaluating the layer in one place gives the right
  output and no messages. Padding the ledger with tensors of the right sizes does
  not help: the tests look at what the messages carry.

## Complexity

With $R = \sum_{s \ne d} |U_{s \to d}|$ routed row-crossings: $2M \le 2\,n(n-1)$
messages and $8DR$ bytes, where $R \le N \cdot \min(k, n - 1)$. Expert work is that
of the sparse dispatch, $k \cdot N$ rows in one call per active expert. Placement
decides $R$: experts that are selected together are cheap to serve from one
device, because their tokens cross once.

## In the wild

- **DeepEP** (the `deepseek-ai/DeepEP` repository) is a communication library for
  exactly this exchange. Its README describes expert-parallel all-to-all GPU
  kernels for MoE *dispatch* and *combine*: dispatch routes tokens to their experts
  across all ranks, and combine reduces expert outputs back to their original
  ranks. Its dispatch call takes the rows with their top-k indices and weights and
  returns, on the receiving side, the rows together with the received indices and
  weights and a handle that the combine call takes. That is why this problem lets
  the routing bookkeeping travel with the rows, and applies the weights where the
  experts are.
- **Deployment at scale** is described in the "DeepSeek-V3 Technical Report"
  (arXiv 2412.19437), section 3.4. For prefill, the MoE part runs with 32-way
  expert parallelism on a unit of 4 nodes and 32 GPUs, chosen so that each expert
  gets a large enough batch; each GPU hosts 8 experts plus a redundant copy of a
  heavily loaded one, and placement is rebalanced from load statistics. For
  decoding, 320-way expert parallelism puts one expert on each GPU and the
  all-to-all transfers go point to point over InfiniBand. The same report trains
  with 64-way expert parallelism.
- **What this problem simplifies.** Everything happens in one process and the
  devices are a ledger: nothing is concurrent, and a message costs its bytes and a
  count. Real systems overlap the exchange with computation, quantize the rows in
  transit, replicate hot experts, and send the indices and weights as metered
  payload too. The rule "one row per token and destination device" and the exact
  message layout checked here are this problem's protocol; they follow the shape of
  dispatch and combine in the sources above without reproducing any library's wire
  format.
