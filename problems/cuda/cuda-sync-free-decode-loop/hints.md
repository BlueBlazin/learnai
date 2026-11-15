List everything in the obvious loop that turns a tensor into a Python value: the
sampled token, the comparison with `eos_id`, the list of results. For each one, ask
whether Python really needs the *value* at that moment, or only needs to pass a
tensor on to the next operation. What is the one thing Python cannot do without
knowing a value?

---

Python needs a value only to decide whether to leave the loop, and that is allowed
once per `check_every` tokens. Everything else can stay a tensor on the device:

- the token: a 0-dimensional tensor; `token.reshape(1)` is what the model takes next;
- the results: a buffer `torch.full((n,), eos_id, device=device)` that you write into
  with `tokens[step] = token`;
- the EOS state: a 0-dimensional boolean tensor `done`, updated with
  `done | (token == eos_id)`;
- "after an EOS, write EOS": `torch.where(done, eos_id, token)` instead of an `if`.

Sample without `nonzero`: the first index whose cumulative sum exceeds the threshold
equals the *number* of cumulative sums that do not exceed it.

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

`and` evaluates `done.item()` only when the left side is true, so the flag is read
once per `check_every` tokens. Return `tokens.cpu()`.
