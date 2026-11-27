## The idea

In `moe-sparse-dispatch`, every routing assignment `(token, expert, weight)` only needed to be grouped by expert. When experts are sharded across devices, each assignment also has a source device (`home[t]`) and a destination device (`placement[e]`), creating two distinct groupings.

- **Group messages by device pair.** The payload sent from device $s$ to device $d$ is the set of *tokens* on $s$ that need at least one expert on $d$. Deduplicating by token matters because a token's input row is identical regardless of how many experts on $d$ will read it, so it should cross the interconnect only once.
- **Group expert calls by expert.** Once rows arrive on device $d$, their origin no longer matters. Each active expert runs once on all local and received rows routed to it, which gives each expert the largest possible batch size.

The combine phase mirrors dispatch. Device $d$ could send back every raw expert output and let the home device weight and sum them, but device $d$ already knows the routing weights for every row it holds. Weighting and summing over its local experts yields one partial row $p_{t,d} = \sum_{e \text{ on } d} w_{t,e} f_e(x_t)$ per received row, leaving the home device with a simple addition. This keeps the exchange symmetric, with the same device pairs, row counts, and token order in both directions. That symmetry lets real libraries reuse the dispatch layout as a handle for combine.

Correctness follows from linearity. Because $y_t = \sum_{e \in S_t} w_{t,e} f_e(x_t)$ is a sum over selected experts, partitioning those experts by device produces partial sums that add up to the exact output.

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

The mask `(target == dst).any(axis=1)` deduplicates tokens so each token is selected once no matter how many of its $k$ experts live on `dst`. Evaluating `routes[tokens] == expert` against the rows already on `dst` makes `at` index directly into `rows`. The JAX and MLX solutions follow the same structure using `.at[...].add(...)`.

## Common mistakes

- **One message per token.** This transfers the same total bytes but increases the message count to the number of rows. Real interconnects pay a fixed latency per message, which is why transfers must be batched per device pair.
- **Broadcasting.** Sending every token to every device is simple to write, but it moves `n_devices - 1` copies of every row needlessly.
- **Sending a row per `(token, expert)`.** Flattening to assignments first and grouping by device sends a token twice whenever two of its experts live on the same device, and returns two rows for it.
- **Overwriting instead of accumulating.** Writing `partial[at] = w * out` works only when each token uses at most one expert on a device. When two experts on the same device serve the same token, overwriting silently drops the first expert's contribution. This bug stays hidden when $k = 1$ or when there is one expert per device.
- **Applying weights at home.** Returning unweighted expert outputs sends more rows back than went out and complicates the combine step.
- **Returning results in the wrong order.** Combine payloads must align with the dispatched token order for each pair. Returning outputs grouped by expert scatters values onto the wrong tokens.
- **Forgetting the combine transfer.** Accumulating results straight into `y` on the expert's device omits the return message. In simulation the numbers match, but half the communication is missing from the ledger.
- **Calling an expert once per sender.** Processing each incoming batch independently calls each expert once per sender device, shrinking batch sizes.
- **Running every local expert on the full inbox.** Evaluating all local experts on every arrived row and zeroing unrouted entries reverts to dense evaluation on each device.
- **Sending empty messages.** Running a fixed all-to-all loop without skipping empty pairs sends zero-row messages between devices that share no tokens.
- **Routing only to the top expert's device.** Sending a token to its top-1 expert's device and evaluating its other experts there assumes weights exist on a device that does not hold them.
- **Not distributing at all.** Evaluating the layer in one place produces zero messages, and faking the ledger with dummy tensors fails because the tests verify every message payload.

## Complexity

Let $n$ be `devs.n_devices`, $M$ the number of active remote device pairs, and $R = \sum_{s \ne d} |U_{s \to d}|$ the total number of remote token crossings. Communication requires $2M \le 2n(n - 1)$ messages and $8DR$ bytes (4 bytes per float32 element, sent out and returned), where $R \le N \cdot \min(k, n - 1)$. Expert computation matches `moe-sparse-dispatch`, processing $k \cdot N$ rows in one call per active expert. Placement directly controls $R$. Co-locating experts that are frequently selected together reduces crossings because shared tokens travel only once.

## In the wild

- **DeepEP.** The `deepseek-ai/DeepEP` library implements expert-parallel all-to-all GPU kernels for MoE *dispatch* and *combine*. Its dispatch kernel routes tokens along with their top-$k$ indices and weights across ranks, returning the received rows, routing metadata, and a layout handle that the combine kernel reuses to reduce expert outputs back to their home ranks. That design is why this problem treats routing bookkeeping as riding along with the rows and applies weights on the expert device.
- **Deployment at scale.** Section 3.4 of the "DeepSeek-V3 Technical Report" (arXiv 2412.19437) describes expert parallelism in production. During prefill, the MoE layers use 32-way expert parallelism across 4 nodes (32 GPUs) to keep per-expert batches large, with each GPU holding 8 experts plus a redundant copy of a heavily loaded expert rebalanced from load statistics. During decoding, 320-way expert parallelism places one expert per GPU with point-to-point all-to-all transfers over InfiniBand. Training uses 64-way expert parallelism.
- **Simplifications.** Everything here runs in a single process with `SimDevices` acting as a ledger rather than concurrent hardware. Production systems overlap communication with computation, quantize activations in transit, replicate hot experts, and count routing indices and weights in the wire payload. The exact message layout checked here captures the structure of dispatch and combine without replicating any specific library's wire format.
