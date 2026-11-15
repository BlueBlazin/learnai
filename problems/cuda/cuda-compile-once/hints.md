The compiler traces `step` once and reuses the trace as long as its guards hold. Which
properties of the two arguments does a guard look at: the shape of `tokens`? its
contents? the value of `pos` when it is a tensor? when it is a Python `int`? Count the
distinct combinations of guarded properties that your loop sends to `step` over 64
tokens, and over prompts of 5, 12, and 17 tokens.

---

`make_step` defines an inner function `step(tokens, pos)` that returns
`model(tokens, pos, cache)` and returns `torch.compile(step, fullgraph=True,
dynamic=False, backend=backend, mode=mode)`: one call, outside every loop. In
`generate`, build the padded block with `torch.zeros((batch, L), dtype=prompt.dtype,
device=device)` and copy the prompt into its first `P` columns; call `step` on it with
`torch.tensor(0, device=device)`. Keep the decode position as a 0-d tensor that starts
at `P` (not at `L`) and grows by one per step. Keep each new token as a `(batch, 1)`
tensor (`argmax(dim=-1, keepdim=True)`) and feed it straight back.

---

```python
logits = step(tokens, torch.tensor(0, device=device))
token = logits[:, n_prompt - 1].argmax(dim=-1, keepdim=True)  # the last real token
out[:, 0] = token[:, 0]
pos = torch.tensor(n_prompt, device=device)
for i in range(1, n_new):
    token = step(token, pos)[:, 0].argmax(dim=-1, keepdim=True)
    out[:, i] = token[:, 0]
    pos += 1
```

`n_prompt - 1` and `i` are Python ints used *outside* the compiled function, where
nothing guards on them. The loop starts at 1: the token written last is not fed back.
