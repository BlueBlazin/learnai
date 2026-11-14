A row behaves as it does alone if (a) nothing real ever attends to its padding,
(b) its real tokens sit at positions $0, 1, 2, \dots$, and (c) its next token is read
from its last real token. Which side must the padding be on so that (c) is the same
column for every row, and stays so for every later step?

---

Pad on the left. With `pad[b] = T - len(prompts[b])` and `cols = mx.arange(T)`:
`real = cols[None, :] >= pad[:, None]` marks the real columns. The prompt mask is
"causal and the key is real, or the key is the query itself", turned into `0` and
`-inf` with `mx.where`; the diagonal is what keeps padding queries from becoming
NaN. Positions are `cols[None, :] - pad[:, None]`, clipped at 0. For the decode
steps you need three things per row that grow with the step: the token just
generated (shape `(B, 1)`), its position `len(prompts[b]) + step`, and a mask over
*all* cached columns plus the new one, `-inf` on the row's padding columns. Keep a
Python list of finished flags; evaluate once and read the `B` tokens once per step.

---

The pieces:

```python
causal = cols[None, :] <= cols[:, None]
allowed = (causal[None] & real[:, None, :]) | mx.eye(T, dtype=mx.bool_)[None]
mask = mx.where(allowed, 0.0, -mx.inf)[:, None]  # (B, 1, T, T)
positions = mx.maximum(cols[None, :] - pad[:, None], 0)
logits, cache = model(tokens, mask=mask, positions=positions)

key_mask = mx.where(real, 0.0, -mx.inf)[:, None, None, :]  # (B, 1, 1, T)
for step in range(max_new_tokens):
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token, cache)
    for row, value in enumerate(token.tolist()):
        if not done[row]:
            out[row].append(value)
            done[row] = value == eos_id or len(out[row]) == max_new_tokens
    if all(done):
        break
    key_mask = mx.concatenate([key_mask, mx.zeros((B, 1, 1, 1))], axis=-1)
    logits, cache = model(token[:, None], cache, mask=key_mask, positions=(lengths + step)[:, None])
```

Check `all(done)` **before** building the next call, and append to a row only while
it is not done.
