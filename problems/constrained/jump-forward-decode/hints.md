In `constrained-greedy-decode`, you compute the permitted token set before calling the model. Consider a step where that set has only one element. What can the argmax over masked logits return, and did the forward pass contribute anything? Then ask what the model's KV cache must contain before the *next* step where it actually has a choice.

---

Keep a `pending` list of tokens the model has not seen yet, initialized to `list(prompt)`. At each step, build the `permitted` list from `constraint.allowed(state)`, appending `eos_id` if `constraint.is_accepting(state)`.

- If `permitted` is empty, stop.
- If `permitted` has one element, take it directly without calling the model.
- If `permitted` has two or more elements, call `lm(pending, cache)` once, clear `pending`, and take the argmax of `logits[-1]` over `permitted`.

In both cases, append the chosen token to the output, stop if it is `eos_id`, advance the constraint, and append the token to `pending`. Always check the size of `permitted`, not `allowed`. One allowed token plus `eos_id` is a free choice, while `eos_id` alone is forced.

---

```python
state, cache, pending, out = constraint.start(), None, list(prompt), []
while len(out) < max_new_tokens:
    permitted = allowed(state) (+ [eos_id] if is_accepting(state))
    if not permitted:
        break
    if len(permitted) == 1:
        token = permitted[0]
    else:
        logits, cache = lm(pending, cache)
        pending = []
        token = argmax of logits[-1] over the permitted ids
    out.append(token)
    if token == eos_id:
        break
    state = constraint.advance(state, token)
    pending.append(token)
return out
```

Passing `cache=None` to `lm(pending, cache)` starts a new sequence, just like `lm(pending)`. Both cost limits fall out of this structure automatically. The model is called only in the `else` branch (once per free step), `pending` holds only the tokens since the last call, and anything left in `pending` when the loop exits is never fed. Checking `len(out) < max_new_tokens` once per token ensures a forced run stops at the limit.
