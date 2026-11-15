## The idea

On a GPU, every PyTorch call queues work and returns; the host is usually many
operations ahead of the device. A host read ends that: to produce a Python number
the host must wait for everything queued so far and copy the value over. PyTorch's
CUDA notes describe both halves: operations are asynchronous, and copying data
between the CPU and the GPU synchronizes. So the question for a decode loop is which
of its values Python actually needs.

- **The sampled token** is only passed on: into the buffer and back into the model.
  It can stay a 0-dimensional tensor.
- **The list of tokens** is needed once, after the loop. A preallocated buffer on
  the device replaces `list.append`, and one `.cpu()` moves it.
- **The EOS test** is the only place where Python must know something, because
  leaving a Python loop is a Python decision. The statement rations it: one read per
  `check_every` tokens. Between reads the state lives in a boolean tensor.

That leaves the steps between an EOS and the next check. They run, because Python
does not know yet. Their output is neutralized with arithmetic instead of control
flow: `torch.where(done, eos_id, token)` writes `eos_id` once `done` is set. Since
the buffer is prefilled with `eos_id` as well, the result is right whether the loop
stopped at a check or ran to the end.

Sampling has the same shape of problem. `torch.nonzero(cdf > threshold)[0]` is the
literal translation of "the first index above the threshold", and its result has a
size that depends on the data, so the host must wait to learn it before it can
allocate the output. Counting is equivalent and has a fixed shape:
$\#\{k : c_k \le t\}$ is the first index with $c_k > t$, because the cumulative sums
are non-decreasing. The clamp guards the one float case in which no cumulative sum
exceeds the threshold.

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

`uniforms[step]`, `cdf[-1]`, and `tokens[step] = token` index with Python ints, which
the host knows; they queue work and wait for nothing. `cdf.shape[0]` is metadata.
The check comes before the model call, so a stop costs no extra forward pass.

## Common mistakes

- **`.item()` on every token**, with a Python list and a Python EOS test. One wait
  per token; this is the loop the problem exists to replace.
- **`if token == eos_id:` on a tensor.** There is no `.item()` in sight, but `if`
  calls `bool()` on the comparison, which is the same read.
- **`int(token)` into a list** while tracking EOS correctly on the device. The flag
  logic is right and the per-token read is still there.
- **`torch.nonzero` (or a boolean-mask index) to sample.** A data-dependent output
  size is a synchronization even though no value reaches Python.
- **Not masking after the EOS.** The tokens sampled between the EOS and the next
  check end up in the result.
- **Never checking.** The cheapest loop in host reads, and after an early EOS it
  runs the model for the remaining tokens for nothing.
- **Checking on the wrong step** (`step % check_every` with a zero-based step).
  The checks happen after tokens 1, `check_every + 1`, ..., one model call later
  than they could have stopped the loop.
- **Trimming the result at the EOS.** A length that depends on the data needs
  another host read, and a caller that batches sequences wanted a fixed shape.
- **Creating the buffer or the flag without `device=`.** On a CPU nothing happens;
  on a GPU every write into the buffer copies a token to the host.

## Complexity

At most $s \le n$ model calls and $P + s - 1$ tokens through the model, where $s$ is
the first check at or after the EOS; $\lfloor s / \texttt{check\_every} \rfloor$ host
reads and one transfer. The price of a larger `check_every` is up to
`check_every - 1` wasted steps per sequence.

## In the wild

With real randomness the sampling line is `torch.multinomial(probs, 1)`, which draws
on the device from the device's generator and returns a tensor; here the uniforms
are an input so that the result is reproducible. Batched generation code keeps an
`unfinished` tensor per sequence and pads finished rows exactly as this loop does,
and the periodic check is the batch-of-one version of "stop when all sequences are
done". What is said here about GPUs is from the PyTorch 2.15 documentation; the
counts were measured on the CPU with the kit's `SyncSpy`, which sees the reads a GPU
would wait for, and the GPU cases have not been run by the author. One limit of that
instrument: `.to("cpu")` is not counted, which is why the statement prescribes
`.cpu()`.
