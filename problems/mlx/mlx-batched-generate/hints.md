A batched row matches its solo run if three conditions hold: no real token ever attends to padding, real tokens start at position 0, and each step's next token is read from the last real token. Which side should you pad on so that the last real token sits in the same column for every row on the prompt call and stays aligned on every decode step?

---

Pad on the left. Let `pad[b] = T - len(prompts[b])` and `cols = mx.arange(T)`, so `real = cols[None, :] >= pad[:, None]` marks the real columns. For the prompt mask, allow a query to attend to a key if the pair is causal and the key is real, or if the key is the query itself, then map booleans to `0.0` and `-inf` with `mx.where`. Including the diagonal prevents padding queries from producing NaN in softmax. Set prompt positions to `cols[None, :] - pad[:, None]` clipped at 0. On each decode step, pass the newly generated tokens of shape `(B, 1)`, positions `len(prompts[b]) + step`, and a key mask covering all cached columns plus the new column (`-inf` on each row's padding columns). Track finished rows in a Python list, evaluating and reading the batch of `B` tokens once per step.

---

The core implementation:

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

Check `all(done)` **before** issuing the next model call, and append to a row only while `not done[row]`.
