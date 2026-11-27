List every place in a standard decode loop that turns a tensor into a Python value: the sampled token, the comparison with `eos_id`, and the output list. For each one, ask whether Python truly needs the value right now or can simply pass a tensor to the next op. What is the single decision Python cannot make without reading a value?

---

Python only needs a value to decide whether to break out of the loop, and you are allowed to check that once every `check_every` tokens. Everything else can stay on `device`:

- Keep the sampled token as a 0-D tensor and pass `token.reshape(1)` to `model`.
- Preallocate `tokens = torch.full((n,), eos_id, dtype=torch.long, device=device)` and write into it with `tokens[step] = token`.
- Track EOS in a 0-D boolean tensor `done` updated via `done = done | (token == eos_id)`.
- Mask post-EOS steps with `torch.where(done, eos_id, token)` instead of an `if` branch.

To sample without `nonzero`, note that the first index whose cumulative sum exceeds the threshold equals the number of cumulative sums that do not exceed it.

---

```python
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
```

Short-circuit `and` evaluates `done.item()` only when `(step + 1) % check_every == 0` is true, so the flag is read once per `check_every` tokens. Return `tokens.cpu()` after the loop.
