## The idea

On a GPU, PyTorch operations are asynchronous. Each call queues work on the device and returns immediately so Python can stay ahead of the GPU. Reading a tensor value on the host breaks this overlap because the CPU must wait for all queued work to finish and then copy the value back.

To avoid per-token stalls, inspect every value in a standard decode loop and ask whether Python actually needs it:

- **The sampled token** is only written into the output buffer and passed back into `model`. It can remain a 0-D tensor on `device`.
- **The output sequence** is only needed after the loop ends. A preallocated buffer on `device` replaces `list.append`, and a single `.cpu()` transfers it at the end.
- **The EOS condition** is the only value Python must inspect, because exiting a Python `for` loop is a host control-flow decision. We track EOS in a 0-D boolean tensor `done` on `device` and read `done.item()` only every `check_every` steps.

Between an EOS token and the next checkpoint, the loop keeps running because Python does not know the sequence has ended yet. Instead of branching with an `if`, we neutralize those extra tokens on the device by calling `torch.where(done, eos_id, token)` to overwrite any post-EOS token with `eos_id`. Because the output buffer is also prefilled with `eos_id`, any uncomputed tail positions already hold the right value when the loop breaks early.

Sampling also has a hidden synchronization trap. Writing `torch.nonzero(cdf > threshold)[0]` translates "the first index above the threshold" literally, but `nonzero` returns a tensor whose length depends on the data. The host must synchronize to learn that length before allocating the output. Instead, let $t = u \cdot c_{V-1}$ be the threshold. Because the cumulative probabilities $c_k$ are non-decreasing, the first index with $c_k > t$ equals the count of entries with $c_k \le t$, written `(cdf <= t).sum()`. That reduction has a fixed scalar shape and needs no host sync. Clamping to `V - 1` handles the rare floating-point case where no cumulative sum exceeds $t$.

## Reference solution (PyTorch)

```python
@torch.inference_mode()
def decode(model, prompt, n, uniforms, device, eos_id, check_every):
    prompt = prompt.to(device)
    uniforms = uniforms.to(device)
    tokens = torch.full((n,), eos_id, dtype=torch.long, device=device)
    done = torch.zeros((), dtype=torch.bool, device=device)

    logits, cache = model(prompt)
    for step in range(n):
        cdf = torch.softmax(logits[-1], dim=-1).cumsum(dim=-1)
        token = (cdf <= uniforms[step] * cdf[-1]).sum().clamp(max=cdf.shape[0] - 1)
        token = torch.where(done, eos_id, token)
        tokens[step] = token
        done = done | (token == eos_id)

        if step + 1 == n:
            break
        if (step + 1) % check_every == 0 and done.item():
            break
        logits, cache = model(token.reshape(1), cache)

    return tokens.cpu()
```

Indexing with Python integers (`uniforms[step]`, `cdf[-1]`, `tokens[step] = token`) uses indices already known to the host, so those operations queue cleanly without synchronizing. Reading `cdf.shape[0]` only inspects tensor metadata. Placing the checkpoint before the next `model(...)` call ensures we never run an extra forward pass after deciding to stop.

## Common mistakes

- **Calling `.item()` on every token** and storing tokens in a Python list. This synchronizes once per token and defeats the goal of the problem.
- **Writing `if token == eos_id:` on a tensor.** Even without `.item()`, Python's `if` calls `bool()` on the tensor, triggering a host synchronization.
- **Calling `int(token)` to append to a list** even while tracking `done` on the device. The EOS flag is sync-free, but the per-token cast still forces a sync.
- **Sampling with `torch.nonzero` or boolean-mask indexing.** Any op with a data-dependent output shape synchronizes the host, even if you never inspect the values in Python.
- **Forgetting to mask tokens after EOS.** Tokens sampled between the first EOS and the next checkpoint leak into the output tensor.
- **Never checking `done`.** Skipping checkpoints minimizes host reads, but after an early EOS the model keeps running for all remaining steps up to `n`, exceeding the `forward_calls` limit.
- **Checking on the wrong step** (such as `step % check_every == 0` with 0-based `step`). This checks after tokens `1`, `check_every + 1`, and so on, causing an extra forward pass before stopping.
- **Trimming the output at the EOS position.** Slicing to a data-dependent length requires an extra host read and breaks the fixed `(n,)` output contract.
- **Allocating `tokens` or `done` without `device=`.** This passes silently on CPU, but on a GPU every write into a CPU buffer triggers a device-to-host copy.

## Complexity

Let $s \le n$ be the stopping step (the first checkpoint at or after the EOS, or $n$ if no EOS occurs). The loop runs at most $s$ model calls and passes $P + s - 1$ tokens through `model`. It performs at most $\lfloor s / \texttt{check\_every} \rfloor$ scalar host reads and 1 final host transfer. Increasing `check_every` reduces host syncs at the cost of up to `check_every - 1` wasted decode steps after an EOS.

## In the wild

With real randomness, the sampling step is typically `torch.multinomial(probs, 1)`, which draws on the device using the device RNG and returns a tensor. Here `uniforms` is passed as an input so results are deterministic.

Batched generation loops in serving frameworks maintain a per-sequence `unfinished` tensor and pad completed rows with EOS in the same way. The periodic checkpoint here is the batch-size-1 version of checking whether all sequences in a batch have finished. The GPU behavior described here follows the PyTorch 2.15 documentation. On CPU, the test suite measures synchronizations using `SyncSpy`, which intercepts the operations that would stall a GPU (the GPU test cases were not run by the author). Because `SyncSpy` does not count `.to("cpu")`, the specification requires using `.cpu()` instead.
